// ============================================================================
// Load generator  (address generator, part 3 of 4):  memory macros -> RF
//
// One engine, two jobs (selected by job_weights at start):
//   job_weights = 0 : load the IFMap input patch of the current tile into the
//                     IFMap RF, layout [channel][row][col], row pitch patch_cols.
//                     Partial edge tiles load only the rows/cols they need.
//   job_weights = 1 : load weights_per_oc consecutive words (one output channel's
//                     kernel, layout [ic][kr][kc]) into the weight RF at 0..N-1.
//
// Both jobs are sequences of "runs" of consecutive words. Memory words are
// low-order interleaved over NUM_MACROS macros (word w -> macro w%8, local w/8),
// so a run is fetched in aligned groups of 8 words: ONE request per macro per
// cycle, all macros with the same macro-local address. Words of a group that fall
// outside the run are simply not requested. Requests lost to the core
// (nmc_grant=0) stay pending and are retried; granted lanes are never re-read.
//
// Read data returns ONE cycle after the grant (sram_bank). The generator
// pipelines the destination RF address by one cycle and outputs the write
// enables; the caller connects  RF write_data[lane m] = nmc_rdata[lane m][7:0].
//
// done pulses when the last word is VISIBLE (valid bit set) in the RF.
// No multipliers: all pointers are incremental, the only variable additions
// are of precomputed descriptor fields.
// ============================================================================
module load_generator #(
    parameter ADDR_WIDTH = 17,
    parameter DIM_WIDTH  = 16,
    parameter RF_AW      = 8,
    parameter NUM_MACROS = 8,
    parameter MACRO_AW   = 12,
    parameter PE_ROWS    = 4,
    parameter PE_COLS    = 4,
    parameter VR_W       = $clog2(PE_ROWS + 1),
    parameter VC_W       = $clog2(PE_COLS + 1)
)(
    input  wire                              clk,
    input  wire                              rst,
    input  wire                              start,
    input  wire                              job_weights,

    // layer descriptor
    input  wire [DIM_WIDTH-1:0]              ifmap_cols,
    input  wire [ADDR_WIDTH-1:0]             ifmap_ch_size,   // ifmap_rows * ifmap_cols
    input  wire [DIM_WIDTH-1:0]              input_channels,
    input  wire [DIM_WIDTH-1:0]              kernel_rows,
    input  wire [DIM_WIDTH-1:0]              kernel_cols,
    input  wire [DIM_WIDTH-1:0]              stride,
    input  wire [RF_AW-1:0]                  patch_cols,      // RF row pitch
    input  wire [RF_AW-1:0]                  rf_ch_stride,    // patch_rows * patch_cols
    input  wire [DIM_WIDTH-1:0]              weights_per_oc,  // Cin * Kr * Kc

    // per tile / per output channel
    input  wire [ADDR_WIDTH-1:0]             ifmap_origin,    // tile's top-left input word (channel 0)
    input  wire [ADDR_WIDTH-1:0]             weight_src_addr, // first weight word of this output channel
    input  wire [VR_W-1:0]                   valid_rows,
    input  wire [VC_W-1:0]                   valid_cols,

    // NMC memory port (reads only)
    output wire [NUM_MACROS-1:0]             nmc_req,
    output wire [NUM_MACROS*MACRO_AW-1:0]    nmc_addr,
    input  wire [NUM_MACROS-1:0]             nmc_grant,

    // register-file write side (data comes from nmc_rdata, low byte)
    output wire [NUM_MACROS-1:0]             rf_wr_en,
    output wire [NUM_MACROS*RF_AW-1:0]       rf_wr_addr,
    output wire                              rf_wr_weight,    // 1: weight RF, 0: IFMap RF

    // request-time view of the destination (combinational, same cycle as nmc_req)
    output wire [NUM_MACROS*RF_AW-1:0]       rf_req_addr,     // RF word lane m WILL write
    output wire                              rf_req_weight,   // 1: weight RF, 0: IFMap RF

    output reg                               busy,
    output reg                               done
);

    localparam SW = DIM_WIDTH + 2;     // signed width for run-relative word index

    // ------------------------------------------------------------------
    // Tile load extent for partial tiles: (valid-1)*S + K, built from adds
    // ------------------------------------------------------------------
    integer k;
    reg [DIM_WIDTH-1:0] lr_acc, lc_acc;
    always @(*) begin
        lr_acc = kernel_rows;
        lc_acc = kernel_cols;
        for (k = 1; k < PE_ROWS; k = k + 1) if (k < valid_rows) lr_acc = lr_acc + stride;
        for (k = 1; k < PE_COLS; k = k + 1) if (k < valid_cols) lc_acc = lc_acc + stride;
    end

    // ------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------
    reg                       job_w;
    reg                       grp_fresh;      // pending set = lane_valid (not yet partially served)
    reg [NUM_MACROS-1:0]      pend;
    reg [ADDR_WIDTH-4:0]      g;              // macro-local address of the current group
    // Words still to fetch in the current run, counted from lane 0 of the current group, and the
    // number of lanes below the run start (first group of a run only). They replace the old signed
    // 'rel0' offset: lane_valid / last_group no longer need 18-bit adders + comparators in the
    // grant -> clock-enable path (that was the critical path on the 7z020).
    //     rel0 = lo-based:   remw = run_len - rel0,  lo = max(0, -rel0)
    reg [DIM_WIDTH:0]         remw;
    reg [2:0]                 lo;
    reg [RF_AW-1:0]           dst0;           // RF address of lane 0 (wraps for lanes below run start)
    reg [DIM_WIDTH-1:0]       run_len;

    // outer loops (patch job)
    reg [DIM_WIDTH-1:0]       lrows_m1;           // rows of the patch - 1
    reg [DIM_WIDTH-1:0]       rows_left, ch_left; // rows / channels still to come after the current run
    reg [ADDR_WIDTH-1:0]      a_ptr, ch_ptr;
    reg [RF_AW-1:0]           d_ptr, ch_dst;

    reg                       fin1;

    // write pipeline (read data arrives one cycle after grant)
    reg [NUM_MACROS-1:0]      wr_en_q;
    reg [NUM_MACROS*RF_AW-1:0] wr_addr_q;
    reg                       wr_w_q;

    // ------------------------------------------------------------------
    // Which lanes of the current group belong to the run
    // ------------------------------------------------------------------
    reg [NUM_MACROS-1:0] lane_valid;
    integer m;
    always @(*) begin
        for (m = 0; m < NUM_MACROS; m = m + 1)
            lane_valid[m] = (m >= lo) && ((|remw[DIM_WIDTH:3]) || (remw[2:0] > m));   // remw > m
    end

    // remw <= NUM_MACROS (8)
    wire last_group = (remw[DIM_WIDTH:4] == 0) && (!remw[3] || (remw[2:0] == 3'd0));

    wire [NUM_MACROS-1:0] cur_pend = grp_fresh ? lane_valid : pend;
    assign nmc_req = busy ? cur_pend : {NUM_MACROS{1'b0}};

    genvar gi;
    generate for (gi = 0; gi < NUM_MACROS; gi = gi + 1) begin : gen_addr
        assign nmc_addr[gi*MACRO_AW +: MACRO_AW] = g[MACRO_AW-1:0];
    end endgenerate

    wire [NUM_MACROS-1:0] granted = nmc_req & nmc_grant;
    wire [NUM_MACROS-1:0] rem     = cur_pend & ~nmc_grant;

    // ------------------------------------------------------------------
    // Next run (patch job only): next row of this channel, else next channel
    // ------------------------------------------------------------------
    wire more_rows = (rows_left != 0);
    wire more_ch   = (ch_left   != 0);
    wire more_runs = !job_w && (more_rows || more_ch);

    reg [ADDR_WIDTH-1:0] nx_a, nx_chptr;
    reg [RF_AW-1:0]      nx_d, nx_chdst;
    always @(*) begin
        nx_chptr = ch_ptr;
        nx_chdst = ch_dst;
        if (more_rows) begin
            nx_a   = a_ptr + ifmap_cols;
            nx_d   = d_ptr + patch_cols;
        end else begin
            nx_chptr = ch_ptr + ifmap_ch_size;
            nx_chdst = ch_dst + rf_ch_stride;
            nx_a     = nx_chptr;
            nx_d     = nx_chdst;
        end
    end

    // start-up values
    wire [ADDR_WIDTH-1:0] a0 = job_weights ? weight_src_addr : ifmap_origin;

    assign rf_wr_en     = wr_en_q;
    assign rf_wr_addr   = wr_addr_q;
    assign rf_wr_weight = wr_w_q;

    genvar gr;
    generate for (gr = 0; gr < NUM_MACROS; gr = gr + 1) begin : gen_req_dst
        assign rf_req_addr[gr*RF_AW +: RF_AW] = dst0 + gr;
    end endgenerate
    assign rf_req_weight = job_w;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            busy <= 1'b0; done <= 1'b0; fin1 <= 1'b0;
            job_w <= 1'b0; grp_fresh <= 1'b0; pend <= 0;
            g <= 0; remw <= 0; lo <= 0; dst0 <= 0; run_len <= 0;
            lrows_m1 <= 0; rows_left <= 0; ch_left <= 0;
            a_ptr <= 0; ch_ptr <= 0; d_ptr <= 0; ch_dst <= 0;
            wr_en_q <= 0; wr_addr_q <= 0; wr_w_q <= 1'b0;
        end else begin
            done    <= fin1;
            fin1    <= 1'b0;
            wr_en_q <= 0;

            if (start && !busy) begin
                busy      <= 1'b1;
                job_w     <= job_weights;
                grp_fresh <= 1'b1;
                run_len   <= job_weights ? weights_per_oc : lc_acc;
                lrows_m1  <= lr_acc - 1'b1;
                rows_left <= lr_acc - 1'b1;
                ch_left   <= input_channels - 1'b1;
                a_ptr     <= a0;
                ch_ptr    <= a0;
                d_ptr     <= 0;
                ch_dst    <= 0;
                g         <= a0[ADDR_WIDTH-1:3];
                remw      <= (job_weights ? weights_per_oc : lc_acc) + a0[2:0];
                lo        <= a0[2:0];
                dst0      <= {RF_AW{1'b0}} - a0[2:0];
            end else if (busy) begin
                wr_en_q <= granted;
                wr_w_q  <= job_w;
                for (m = 0; m < NUM_MACROS; m = m + 1)
                    wr_addr_q[m*RF_AW +: RF_AW] <= dst0 + m;

                if (rem == 0) begin
                    grp_fresh <= 1'b1;
                    if (!last_group) begin
                        g    <= g + 1'b1;
                        remw <= remw - NUM_MACROS;
                        lo   <= 3'd0;
                        dst0 <= dst0 + NUM_MACROS;
                    end else if (more_runs) begin
                        a_ptr   <= nx_a;
                        ch_ptr  <= nx_chptr;
                        d_ptr   <= nx_d;
                        ch_dst  <= nx_chdst;
                        if (more_rows) rows_left <= rows_left - 1'b1;
                        else begin
                            rows_left <= lrows_m1;
                            ch_left   <= ch_left - 1'b1;
                        end
                        g       <= nx_a[ADDR_WIDTH-1:3];
                        remw    <= run_len + nx_a[2:0];
                        lo      <= nx_a[2:0];
                        dst0    <= nx_d - nx_a[2:0];
                    end else begin
                        busy <= 1'b0;
                        fin1 <= 1'b1;
                    end
                end else begin
                    pend      <= rem;
                    grp_fresh <= 1'b0;
                end
            end
        end
    end

endmodule