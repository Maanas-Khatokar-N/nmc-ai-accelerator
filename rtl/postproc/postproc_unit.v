// ============================================================================
// Post-processing unit (inter-layer glue), one per bank, NMC-port master
//
//   OFMap (int32, one element per word)  --max-pool-->  requant + ReLU  --> IFMap (int8, low byte)
//
//   src : [channel][row][col] int32 words at src_base           (the nmc_unit's OFMap region)
//   dst : [channel][out_row][out_col] int8 in the low byte of a 32-bit word (upper 24 bits = 0),
//         dense, starting at dst_base. This is exactly what the next layer's load_generator reads.
//   src and dst regions must not overlap.
//
//   pool_k = 1 : no pooling (requant + ReLU only)
//   pool_k = 2 : 2x2 max-pool, stride 2   (any k works; rows/cols beyond out_rows*pool_k are ignored)
//
//   Order is MAX-POOL first, then requant (valid because requant is monotone, see requant.v).
//
// Datapath (everything streams, one element per cycle when the port is free):
//   read engine   : walks each pooling window element by element (1 macro request / cycle)
//   window max    : running max of the pool_k^2 words
//   param RAM     : per-channel {bias, mult, shift}, read with the window's channel
//   requant       : 3-stage pipeline -> int8
//   output FIFO   : decouples the pipeline from the write port
//   write engine  : dst words are produced in dst order, so the write address is a plain counter
//
// Memory port (same protocol as the other NMC masters; the cluster muxes it with nmc_unit):
//   write wins over read when both hit the same macro; requests lost to the core are retried.
//   Credit counter (inflight) guarantees the FIFO can never overflow.
//
// Descriptor fields marked (sw) are precomputed by software, like the conv descriptor:
//   out_rows, out_cols (sw)  src_rows / pool_k , src_cols / pool_k
//   src_ch_size        (sw)  src_rows * src_cols
//   row_step           (sw)  pool_k * src_cols
//   total_out          (sw)  channels * out_rows * out_cols
//
// Per-channel parameters are written through qp_* before start (one write per cycle).
// ============================================================================
module postproc_unit #(
    parameter ADDR_WIDTH = 17,
    parameter DIM_WIDTH  = 16,
    parameter NUM_MACROS = 8,
    parameter MACRO_AW   = 12,
    parameter MAX_OC     = 512,
    parameter FIFO_DEPTH = 8,
    parameter ACC_W      = 32,
    parameter MUL_W      = 16,
    parameter SH_W       = 5,
    parameter QA_W       = $clog2(MAX_OC)
)(
    input  wire                      clk,
    input  wire                      rst,
    input  wire                      start,
    output reg                       done,
    output wire                      busy,
    input  wire                      mem_allow,       // 1 = may use the memory port (access window)

    // ---------------- layer descriptor ----------------
    input  wire [ADDR_WIDTH-1:0]     src_base,
    input  wire [ADDR_WIDTH-1:0]     dst_base,
    input  wire [DIM_WIDTH-1:0]      src_cols,
    input  wire [DIM_WIDTH-1:0]      out_rows,
    input  wire [DIM_WIDTH-1:0]      out_cols,
    input  wire [DIM_WIDTH-1:0]      channels,
    input  wire [DIM_WIDTH-1:0]      pool_k,
    input  wire [ADDR_WIDTH-1:0]     src_ch_size,
    input  wire [ADDR_WIDTH-1:0]     row_step,
    input  wire [ADDR_WIDTH-1:0]     total_out,
    input  wire signed [7:0]         out_zp,
    input  wire                      relu_en,

    // ---------------- per-channel requant parameters ----------------
    input  wire                      qp_we,
    input  wire [QA_W-1:0]           qp_waddr,
    input  wire signed [ACC_W-1:0]   qp_bias,
    input  wire [MUL_W-1:0]          qp_mult,
    input  wire [SH_W-1:0]           qp_shift,

    // ---------------- NMC memory port ----------------
    output wire [NUM_MACROS-1:0]            nmc_req,
    output wire [NUM_MACROS-1:0]            nmc_we,
    output wire [NUM_MACROS*MACRO_AW-1:0]   nmc_addr,
    output wire [NUM_MACROS*32-1:0]         nmc_wdata,
    input  wire [NUM_MACROS-1:0]            nmc_grant,
    input  wire [NUM_MACROS*32-1:0]         nmc_rdata
);
    localparam SEL_W  = $clog2(NUM_MACROS);
    localparam FIFO_W = $clog2(FIFO_DEPTH);
    localparam QW     = ACC_W + MUL_W + SH_W;

    wire [ADDR_WIDTH-1:0] pk_a   = {{(ADDR_WIDTH-DIM_WIDTH){1'b0}}, pool_k};
    wire [ADDR_WIDTH-1:0] cols_a = {{(ADDR_WIDTH-DIM_WIDTH){1'b0}}, src_cols};

    // ------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------
    reg                   running, rd_active;
    reg [DIM_WIDTH-1:0]   oc_cnt, pr_cnt, pc_cnt, wr_cnt, wc_cnt;
    reg [ADDR_WIDTH-1:0]  el_addr, wrow_addr, win_addr, row_ptr, ch_ptr, nw;
    reg [FIFO_W:0]        inflight;
    reg [ADDR_WIDTH-1:0]  dst_ptr, wleft;

    assign busy = running;

    // ------------------------------------------------------------------
    // Output FIFO
    // ------------------------------------------------------------------
    reg [7:0]         fifo [0:FIFO_DEPTH-1];
    reg [FIFO_W-1:0]  wp, rp;
    reg [FIFO_W:0]    fcnt;
    wire [7:0]        fifo_head = fifo[rp];

    // ------------------------------------------------------------------
    // Request generation
    // ------------------------------------------------------------------
    wire first_el = (wr_cnt == 0) && (wc_cnt == 0);
    wire wc_last  = (wc_cnt + 1 >= pool_k);
    wire wr_last  = (wr_cnt + 1 >= pool_k);
    wire last_el  = wc_last && wr_last;

    wire             wr_req_v = running && (fcnt != 0) && mem_allow && (wleft != 0);
    wire [SEL_W-1:0] wr_macro = dst_ptr[SEL_W-1:0];
    wire             rd_can   = rd_active && mem_allow && (!first_el || (inflight < FIFO_DEPTH));
    wire [SEL_W-1:0] rd_macro = el_addr[SEL_W-1:0];
    wire             rd_req   = rd_can && !(wr_req_v && (wr_macro == rd_macro));

    genvar m;
    generate for (m = 0; m < NUM_MACROS; m = m + 1) begin : g_port
        wire wr_here = wr_req_v && (wr_macro == m);
        wire rd_here = rd_req   && (rd_macro == m);
        assign nmc_req[m]  = wr_here | rd_here;
        assign nmc_we[m]   = wr_here;
        assign nmc_addr[m*MACRO_AW +: MACRO_AW] = wr_here ? dst_ptr[SEL_W +: MACRO_AW]
                                                          : el_addr[SEL_W +: MACRO_AW];
        assign nmc_wdata[m*32 +: 32] = {24'b0, fifo_head};
    end endgenerate

    wire rd_fire = rd_req    && nmc_grant[rd_macro];
    wire wr_fire = wr_req_v  && nmc_grant[wr_macro];

    // ------------------------------------------------------------------
    // Read data capture: window max
    // ------------------------------------------------------------------
    reg                        rd_q, rd_first_q, rd_last_q, win_v;
    reg [SEL_W-1:0]            rd_macro_q;
    reg [DIM_WIDTH-1:0]        rd_oc_q;
    reg signed [ACC_W-1:0]     pmax;
    wire signed [ACC_W-1:0]    rdat = nmc_rdata[rd_macro_q*32 +: ACC_W];

    reg [QW-1:0] qp_mem [0:MAX_OC-1];
    reg [QW-1:0] qp_q;
    always @(posedge clk) begin
        if (qp_we) qp_mem[qp_waddr] <= {qp_bias, qp_mult, qp_shift};
        qp_q <= qp_mem[rd_oc_q[QA_W-1:0]];
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            rd_q <= 1'b0; win_v <= 1'b0;
            rd_first_q <= 1'b0; rd_last_q <= 1'b0; rd_macro_q <= 0; rd_oc_q <= 0;
        end else begin
            rd_q       <= rd_fire;
            rd_first_q <= first_el;
            rd_last_q  <= last_el;
            rd_macro_q <= rd_macro;
            rd_oc_q    <= oc_cnt;
            win_v      <= rd_q && rd_last_q;
        end
    end

    always @(posedge clk)
        if (rd_q && (rd_first_q || (rdat > pmax))) pmax <= rdat;

    // ------------------------------------------------------------------
    // Requant + ReLU
    // ------------------------------------------------------------------
    wire              rq_valid;
    wire signed [7:0] rq_y;

    requant #(.ACC_W(ACC_W), .MUL_W(MUL_W), .SH_W(SH_W)) u_rq (
        .clk(clk), .rst(rst), .in_valid(win_v), .x(pmax),
        .bias(qp_q[QW-1 -: ACC_W]), .mult(qp_q[SH_W +: MUL_W]), .shift(qp_q[SH_W-1:0]),
        .zp(out_zp), .relu_en(relu_en), .out_valid(rq_valid), .y(rq_y));

    // ------------------------------------------------------------------
    // Control: counters, FIFO, done
    // ------------------------------------------------------------------
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            running <= 1'b0; rd_active <= 1'b0; done <= 1'b0;
            oc_cnt <= 0; pr_cnt <= 0; pc_cnt <= 0; wr_cnt <= 0; wc_cnt <= 0;
            el_addr <= 0; wrow_addr <= 0; win_addr <= 0; row_ptr <= 0; ch_ptr <= 0;
            inflight <= 0; dst_ptr <= 0; wleft <= 0;
            wp <= 0; rp <= 0; fcnt <= 0;
        end else begin
            done <= 1'b0;

            if (start && !running) begin
                running   <= (total_out != 0);
                rd_active <= (total_out != 0);
                oc_cnt <= 0; pr_cnt <= 0; pc_cnt <= 0; wr_cnt <= 0; wc_cnt <= 0;
                el_addr <= src_base; wrow_addr <= src_base; win_addr <= src_base;
                row_ptr <= src_base; ch_ptr <= src_base;
                inflight <= 0; dst_ptr <= dst_base; wleft <= total_out;
                wp <= 0; rp <= 0; fcnt <= 0;
            end else if (running) begin
                // ---- FIFO ----
                if (rq_valid) begin fifo[wp] <= rq_y; wp <= wp + 1'b1; end
                if (wr_fire)  rp <= rp + 1'b1;
                fcnt <= fcnt + rq_valid - wr_fire;

                // ---- credits: one per window in flight (first element issued .. write granted) ----
                inflight <= inflight + (rd_fire && first_el) - wr_fire;

                // ---- write side ----
                if (wr_fire) begin
                    dst_ptr <= dst_ptr + 1'b1;
                    wleft   <= wleft - 1'b1;
                    if (wleft == 1) begin running <= 1'b0; done <= 1'b1; end
                end

                // ---- read side: advance the window walk on every granted element ----
                if (rd_fire) begin
                    if (!wc_last) begin
                        wc_cnt  <= wc_cnt + 1'b1;
                        el_addr <= el_addr + 1'b1;
                    end else if (!wr_last) begin
                        wc_cnt    <= 0;
                        wr_cnt    <= wr_cnt + 1'b1;
                        wrow_addr <= wrow_addr + cols_a;
                        el_addr   <= wrow_addr + cols_a;
                    end else begin
                        wc_cnt <= 0; wr_cnt <= 0;
                        if (pc_cnt + 1 < out_cols) begin
                            pc_cnt <= pc_cnt + 1'b1;
                            nw = win_addr + pk_a;
                        end else if (pr_cnt + 1 < out_rows) begin
                            pc_cnt <= 0; pr_cnt <= pr_cnt + 1'b1;
                            nw = row_ptr + row_step;
                            row_ptr <= nw;
                        end else if (oc_cnt + 1 < channels) begin
                            pc_cnt <= 0; pr_cnt <= 0; oc_cnt <= oc_cnt + 1'b1;
                            nw = ch_ptr + src_ch_size;
                            row_ptr <= nw; ch_ptr <= nw;
                        end else begin
                            rd_active <= 1'b0;
                            nw = win_addr;
                        end
                        win_addr <= nw; wrow_addr <= nw; el_addr <= nw;
                    end
                end
            end
        end
    end
endmodule