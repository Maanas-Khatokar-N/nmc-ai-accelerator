// ============================================================================
// Address generator
//
// Wrapper around four generators that share ONE layer descriptor:
//   1. tile_scheduler            output tile coordinates, PE valid mask
//   2. compute_sequencer         kernel row/col/channel loop -> weight index + 16 IFMap RF read addresses
//   3. load_generator            memory macros -> IFMap RF / weight RF
//   4. store_reload_generator    PE accumulators <-> OFMap macros
//
//
//   for each output tile:
//       flush IFMap RF; load IFMap patch                 (reused by all output channels)
//       for each output channel:
//           flush weight RF; load its weights
//           compute (all input channels reduced in the PE)
//           store the 16 accumulators to the OFMap
//
// ---------------------------------------------------------------------------
// Layer descriptor.  Fields marked (sw) are precomputed by software / at
// configuration time so that the hardware needs no runtime multiplier or divider.
// All addresses are WORD addresses, one element per 32-bit word.
//
//   ifmap_cols, kernel_rows, kernel_cols, input_channels, output_channels, stride
//   ofmap_rows      (sw) (ifmap_rows - kernel_rows) / stride + 1
//   ofmap_cols      (sw) (ifmap_cols - kernel_cols) / stride + 1
//   ifmap_ch_size   (sw) ifmap_rows * ifmap_cols
//   ofmap_ch_size   (sw) ofmap_rows * ofmap_cols
//   ifmap_row_step  (sw) stride * ifmap_cols
//   patch_cols           (PE_COLS-1)*stride + kernel_cols     (computed here, shift-add)
//   rf_row_step     (sw) stride * patch_cols
//   rf_ch_stride    (sw) patch_rows * patch_cols,  patch_rows = (PE_ROWS-1)*stride + kernel_rows
//   weights_per_oc  (sw) input_channels * kernel_rows * kernel_cols
//
// Limits to respect when tiling (software):
//   input_channels * rf_ch_stride <= IFMap RF depth (256)
//   weights_per_oc                <= weight RF depth (256)
//
// ---------------------------------------------------------------------------
// Interface to the access controller / datapath:
//   compute_start (out, 1-cycle pulse): clear the PE accumulators now. The first
//        valid op appears >= 2 cycles later, so a clear applied directly or one
//        cycle later (registered) both land before the first MAC.
//   seq_valid / seq_last / ifmap_rf_addr / weight_rf_addr (out): current op.
//   seq_enable (in): the access controller consumed the op this cycle.
//   compute_done (in): all MACs of this output channel are accumulated and
//        acc_out is final (the access controller delays this by its own
//        pipeline depth + 1 cycle for the accumulator register).
//   pe_mask (out): valid PEs of the current tile.
//   ifrf_flush / wrf_flush (out): clear the D/V bits of the IFMap / weight RF.
//   ifrf_wr_en / wrf_wr_en / rf_wr_addr (out): RF write side; write_data lane m
//        = nmc_rdata[lane m][7:0] (same for both RFs).
//   nmc_*: NMC memory-port requests (load or store, never both at once);
//        st_lane_sel[m] selects which accumulator drives nmc_wdata[m].
// ============================================================================
module address_generator #(
    parameter ADDR_WIDTH = 17,
    parameter DIM_WIDTH  = 16,
    parameter RF_AW      = 8,
    parameter NUM_MACROS = 8,
    parameter MACRO_AW   = 12,
    parameter PE_ROWS    = 4,
    parameter PE_COLS    = 4,
    parameter NUM_PE     = PE_ROWS * PE_COLS,
    parameter LANE_W     = $clog2(NUM_PE)
)(
    input  wire clk,
    input  wire rst,
    input  wire start,
    output reg  done,

    // ---------------- layer descriptor ----------------
    input  wire [DIM_WIDTH-1:0]  ifmap_cols,
    input  wire [DIM_WIDTH-1:0]  kernel_rows,
    input  wire [DIM_WIDTH-1:0]  kernel_cols,
    input  wire [DIM_WIDTH-1:0]  input_channels,
    input  wire [DIM_WIDTH-1:0]  output_channels,
    input  wire [DIM_WIDTH-1:0]  stride,
    input  wire [DIM_WIDTH-1:0]  ofmap_rows,
    input  wire [DIM_WIDTH-1:0]  ofmap_cols,
    input  wire [ADDR_WIDTH-1:0] ifmap_ch_size,
    input  wire [ADDR_WIDTH-1:0] ofmap_ch_size,
    input  wire [ADDR_WIDTH-1:0] ifmap_row_step,
    input  wire [RF_AW-1:0]      rf_row_step,
    input  wire [RF_AW-1:0]      rf_ch_stride,
    input  wire [DIM_WIDTH-1:0]  weights_per_oc,
    input  wire [ADDR_WIDTH-1:0] ifmap_base_addr,
    input  wire [ADDR_WIDTH-1:0] weight_base_addr,
    input  wire [ADDR_WIDTH-1:0] ofmap_base_addr,

    // ---------------- compute side (to access controller) ----------------
    output wire                    compute_start,
    input  wire                    compute_done,
    input  wire                    seq_enable,
    output wire                    seq_valid,
    output wire                    seq_last,
    output wire [NUM_PE*RF_AW-1:0] ifmap_rf_addr,
    output wire [RF_AW-1:0]        weight_rf_addr,
    output wire [NUM_PE-1:0]       pe_mask,          // mask of the tile being COMPUTED

    // ---------------- register files (ping-pong) ----------------
    output wire                        ifrf_flush,
    output wire                        ifrf_flush_sel,
    output wire                        ifrf_wr_sel,
    output wire                        ifrf_rd_sel,
    output wire                        wrf_flush,
    output wire                        wrf_flush_sel,
    output wire                        wrf_wr_sel,
    output wire                        wrf_rd_sel,
    output wire [NUM_MACROS-1:0]       ifrf_wr_en,
    output wire [NUM_MACROS-1:0]       wrf_wr_en,
    output wire [NUM_MACROS*RF_AW-1:0] rf_wr_addr,

    // ---------------- OFMap shadow ----------------
    output wire                        st_capture,   // acc_out -> shadow register (datapath)

    // ---------------- NMC memory port ----------------
    output wire [NUM_MACROS-1:0]          nmc_req,
    output wire [NUM_MACROS-1:0]          nmc_we,
    output wire [NUM_MACROS*MACRO_AW-1:0] nmc_addr,
    input  wire [NUM_MACROS-1:0]          nmc_grant,
    output wire [NUM_MACROS*LANE_W-1:0]   st_lane_sel,

    output wire                           ph_load,
    output wire                           ph_compute,
    output wire                           ph_store
);

    localparam VR_W = $clog2(PE_ROWS + 1);
    localparam VC_W = $clog2(PE_COLS + 1);

    wire [DIM_WIDTH-1:0] patch_cols = (PE_COLS - 1) * stride + kernel_cols;

    reg  layer_busy;
    wire layer_start = start && !layer_busy;

    // ------------------------------------------------------------------
    // Loader FSM (producer): [flush half, load patch] per tile,
    //                        [flush half, load weights] per output channel
    // ------------------------------------------------------------------
    localparam [3:0] L_IDLE     = 4'd0,  L_WAIT_IF  = 4'd1,  L_IF_FLUSH = 4'd2,
                     L_LDP_GO   = 4'd3,  L_LDP_WAIT = 4'd4,  L_WAIT_W   = 4'd5,
                     L_W_FLUSH  = 4'd6,  L_LDW_GO   = 4'd7,  L_LDW_WAIT = 4'd8,
                     L_NEXT_OC  = 4'd9,  L_CHECK    = 4'd10;
    reg [3:0]            lstate;
    reg                  ld_ifb, ld_wb;          // half the loader is filling
    reg [DIM_WIDTH-1:0]  ld_oc;
    reg [ADDR_WIDTH-1:0] oc_w_ptr;
    reg                  no_more_tiles;

    // ------------------------------------------------------------------
    // Compute engine FSM (consumer)
    // ------------------------------------------------------------------
    localparam [2:0] C_IDLE = 3'd0, C_WAIT = 3'd1, C_GO = 3'd2,
                     C_RUN  = 3'd3, C_CAP  = 3'd4, C_FIN = 3'd5;
    reg [2:0]            cstate;
    reg                  cp_ifb, cp_wb;          // half the compute engine reads
    reg [DIM_WIDTH-1:0]  cmp_oc;
    reg [ADDR_WIDTH-1:0] cp_of_ptr;
    reg                  sh_full;                // shadow holds data not yet stored

    reg [1:0] if_full, w_full;                   // half is loaded and not yet released

    // per-tile metadata, indexed by IFMap half
    reg [NUM_PE-1:0]     meta_mask [0:1];
    reg [VR_W-1:0]       meta_vr   [0:1];
    reg [VC_W-1:0]       meta_vc   [0:1];
    reg [ADDR_WIDTH-1:0] meta_of   [0:1];

    // ------------------------------------------------------------------
    // 1. Tile scheduler (now a LOAD pointer: runs ahead of compute)
    // ------------------------------------------------------------------
    wire                  ts_valid;
    wire [DIM_WIDTH-1:0]  ts_row, ts_col;
    wire [ADDR_WIDTH-1:0] ts_ifmap_origin, ts_ofmap_origin;
    wire [VR_W-1:0]       ts_vrows;
    wire [VC_W-1:0]       ts_vcols;
    wire [NUM_PE-1:0]     ts_mask;
    wire                  ts_advance = (lstate == L_LDP_GO);

    tile_scheduler #(
        .ADDR_WIDTH(ADDR_WIDTH), .DIM_WIDTH(DIM_WIDTH),
        .PE_ROWS(PE_ROWS), .PE_COLS(PE_COLS)
    ) u_tile (
        .clk(clk), .rst(rst),
        .start(layer_start), .advance(ts_advance),
        .ofmap_rows(ofmap_rows), .ofmap_cols(ofmap_cols), .stride(stride),
        .ifmap_row_step(ifmap_row_step),
        .ifmap_base_addr(ifmap_base_addr), .ofmap_base_addr(ofmap_base_addr),
        .tile_valid(ts_valid), .tile_row(ts_row), .tile_col(ts_col),
        .ifmap_origin(ts_ifmap_origin), .ofmap_origin(ts_ofmap_origin),
        .valid_rows(ts_vrows), .valid_cols(ts_vcols), .pe_mask(ts_mask)
    );

    // latch tile metadata when its patch load is launched (tile pointer advances on the same edge)
    always @(posedge clk) begin
        if (lstate == L_LDP_GO) begin
            meta_mask[ld_ifb] <= ts_mask;
            meta_vr  [ld_ifb] <= ts_vrows;
            meta_vc  [ld_ifb] <= ts_vcols;
            meta_of  [ld_ifb] <= ts_ofmap_origin;
        end
    end

    assign pe_mask = meta_mask[cp_ifb];

    // ------------------------------------------------------------------
    // 2. Compute sequencer
    // ------------------------------------------------------------------
    assign compute_start = (cstate == C_GO);

    wire seq_done_unused;
    compute_sequencer #(
        .DIM_WIDTH(DIM_WIDTH), .RF_AW(RF_AW),
        .NUM_PE(NUM_PE), .PE_ROWS(PE_ROWS), .PE_COLS(PE_COLS)
    ) u_seq (
        .clk(clk), .rst(rst),
        .start(compute_start), .enable(seq_enable),
        .kernel_rows(kernel_rows), .kernel_cols(kernel_cols),
        .input_channels(input_channels), .stride(stride),
        .patch_cols(patch_cols[RF_AW-1:0]),
        .rf_row_step(rf_row_step), .rf_ch_stride(rf_ch_stride),
        .ifmap_rf_addr(ifmap_rf_addr), .weight_rf_addr(weight_rf_addr),
        .valid(seq_valid), .last(seq_last), .done(seq_done_unused)
    );

    // ------------------------------------------------------------------
    // 3. Load generator
    // ------------------------------------------------------------------
    wire ld_start = (lstate == L_LDP_GO) || (lstate == L_LDW_GO);
    wire ld_job_w = (lstate == L_LDW_GO);
    wire ld_busy, ld_done;
    wire [NUM_MACROS-1:0]          ld_req;
    wire [NUM_MACROS*MACRO_AW-1:0] ld_addr;
    wire                           ld_wr_weight;
    wire [NUM_MACROS-1:0]          ld_wr_en;
    wire [NUM_MACROS-1:0]          ld_grant;

    load_generator #(
        .ADDR_WIDTH(ADDR_WIDTH), .DIM_WIDTH(DIM_WIDTH), .RF_AW(RF_AW),
        .NUM_MACROS(NUM_MACROS), .MACRO_AW(MACRO_AW),
        .PE_ROWS(PE_ROWS), .PE_COLS(PE_COLS)
    ) u_load (
        .clk(clk), .rst(rst),
        .start(ld_start), .job_weights(ld_job_w),
        .ifmap_cols(ifmap_cols), .ifmap_ch_size(ifmap_ch_size),
        .input_channels(input_channels),
        .kernel_rows(kernel_rows), .kernel_cols(kernel_cols), .stride(stride),
        .patch_cols(patch_cols[RF_AW-1:0]), .rf_ch_stride(rf_ch_stride),
        .weights_per_oc(weights_per_oc),
        .ifmap_origin(ts_ifmap_origin), .weight_src_addr(oc_w_ptr),
        .valid_rows(ts_vrows), .valid_cols(ts_vcols),
        .nmc_req(ld_req), .nmc_addr(ld_addr), .nmc_grant(ld_grant),
        .rf_wr_en(ld_wr_en), .rf_wr_addr(rf_wr_addr), .rf_wr_weight(ld_wr_weight),
        .busy(ld_busy), .done(ld_done)
    );

    assign ifrf_wr_en = ld_wr_en & {NUM_MACROS{~ld_wr_weight}};
    assign wrf_wr_en  = ld_wr_en & {NUM_MACROS{ ld_wr_weight}};

    // ------------------------------------------------------------------
    // 4. Store generator (reads the OFMap shadow, not the live accumulators)
    // ------------------------------------------------------------------
    wire cmp_oc_last = (cmp_oc + 1 >= output_channels);
    wire [ADDR_WIDTH-1:0] st_addr_now = (cmp_oc == 0) ? meta_of[cp_ifb] : cp_of_ptr;
    wire cap_fire = (cstate == C_CAP) && !sh_full;

    assign st_capture = cap_fire;

    wire st_busy, st_done;
    wire [NUM_MACROS-1:0]          st_req, st_we, st_grant;
    wire [NUM_MACROS*MACRO_AW-1:0] st_addr;

    store_reload_generator #(
        .ADDR_WIDTH(ADDR_WIDTH), .DIM_WIDTH(DIM_WIDTH),
        .NUM_MACROS(NUM_MACROS), .MACRO_AW(MACRO_AW),
        .PE_ROWS(PE_ROWS), .PE_COLS(PE_COLS)
    ) u_store (
        .clk(clk), .rst(rst),
        .start(cap_fire), .reload(1'b0),
        .ofmap_cols(ofmap_cols), .ofmap_tile_addr(st_addr_now),
        .valid_rows(meta_vr[cp_ifb]), .valid_cols(meta_vc[cp_ifb]),
        .nmc_req(st_req), .nmc_we(st_we), .nmc_addr(st_addr), .nmc_grant(st_grant),
        .st_lane_sel(st_lane_sel),
        .rl_wr_en(), .rl_wr_lane(),
        .busy(st_busy), .done(st_done)
    );

    // ------------------------------------------------------------------
    // NMC port arbiter: per macro, store wins over load; loser retries
    // ------------------------------------------------------------------
    assign nmc_req   = ld_req | st_req;
    assign nmc_we    = st_req & st_we;
    assign st_grant  = nmc_grant & st_req;
    assign ld_grant  = nmc_grant & ld_req & ~st_req;

    genvar am;
    generate for (am = 0; am < NUM_MACROS; am = am + 1) begin : g_arb
        assign nmc_addr[am*MACRO_AW +: MACRO_AW] = st_req[am] ? st_addr[am*MACRO_AW +: MACRO_AW]
                                                              : ld_addr[am*MACRO_AW +: MACRO_AW];
    end endgenerate

    // ------------------------------------------------------------------
    // RF half selects / flushes
    // ------------------------------------------------------------------
    assign ifrf_flush     = (lstate == L_IF_FLUSH);
    assign ifrf_flush_sel = ld_ifb;
    assign ifrf_wr_sel    = ld_ifb;
    assign ifrf_rd_sel    = cp_ifb;
    assign wrf_flush      = (lstate == L_W_FLUSH);
    assign wrf_flush_sel  = ld_wb;
    assign wrf_wr_sel     = ld_wb;
    assign wrf_rd_sel     = cp_wb;

    // perf phases (they now OVERLAP, so load+compute+store can exceed total)
    assign ph_load    = (lstate == L_LDP_GO) || (lstate == L_LDP_WAIT) ||
                        (lstate == L_LDW_GO) || (lstate == L_LDW_WAIT);
    assign ph_compute = (cstate == C_GO) || (cstate == C_RUN) || (cstate == C_CAP);
    assign ph_store   = sh_full;

    // ------------------------------------------------------------------
    // Half occupancy flags
    // ------------------------------------------------------------------
    wire if_set = (lstate == L_LDP_WAIT) && ld_done;
    wire w_set  = (lstate == L_LDW_WAIT) && ld_done;
    wire if_clr = cap_fire && cmp_oc_last;
    wire w_clr  = cap_fire;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            if_full <= 2'b00;  w_full <= 2'b00;
        end else if (layer_start) begin
            if_full <= 2'b00;  w_full <= 2'b00;
        end else begin
            if (if_set) if_full[ld_ifb] <= 1'b1;
            if (if_clr) if_full[cp_ifb] <= 1'b0;
            if (w_set)  w_full[ld_wb]   <= 1'b1;
            if (w_clr)  w_full[cp_wb]   <= 1'b0;
        end
    end

    // ------------------------------------------------------------------
    // Loader FSM
    // ------------------------------------------------------------------
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            lstate <= L_IDLE; ld_ifb <= 1'b0; ld_wb <= 1'b0;
            ld_oc <= 0; oc_w_ptr <= 0; no_more_tiles <= 1'b0;
        end else begin
            case (lstate)
                L_IDLE: if (layer_start) begin
                    ld_ifb <= 1'b0; ld_wb <= 1'b0; no_more_tiles <= 1'b0;
                    lstate <= L_WAIT_IF;
                end

                L_WAIT_IF:  if (!if_full[ld_ifb]) lstate <= L_IF_FLUSH;
                L_IF_FLUSH: begin
                    ld_oc    <= 0;
                    oc_w_ptr <= weight_base_addr;
                    lstate   <= L_LDP_GO;
                end
                L_LDP_GO:   lstate <= L_LDP_WAIT;
                L_LDP_WAIT: if (ld_done) lstate <= L_WAIT_W;

                L_WAIT_W:   if (!w_full[ld_wb]) lstate <= L_W_FLUSH;
                L_W_FLUSH:  lstate <= L_LDW_GO;
                L_LDW_GO:   lstate <= L_LDW_WAIT;
                L_LDW_WAIT: if (ld_done) begin
                    ld_wb  <= ~ld_wb;
                    lstate <= L_NEXT_OC;
                end

                L_NEXT_OC: begin
                    if (ld_oc + 1 < output_channels) begin
                        ld_oc    <= ld_oc + 1'b1;
                        oc_w_ptr <= oc_w_ptr + weights_per_oc;
                        lstate   <= L_WAIT_W;
                    end else begin
                        ld_ifb <= ~ld_ifb;
                        lstate <= L_CHECK;
                    end
                end

                L_CHECK: begin
                    no_more_tiles <= !ts_valid;
                    lstate        <= ts_valid ? L_WAIT_IF : L_IDLE;
                end

                default: lstate <= L_IDLE;
            endcase
        end
    end

    // ------------------------------------------------------------------
    // Compute engine FSM
    // ------------------------------------------------------------------
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            cstate <= C_IDLE; cp_ifb <= 1'b0; cp_wb <= 1'b0;
            cmp_oc <= 0; cp_of_ptr <= 0; sh_full <= 1'b0;
            done <= 1'b0; layer_busy <= 1'b0;
        end else begin
            done <= 1'b0;
            if (layer_start) layer_busy <= 1'b1;

            if (cap_fire)      sh_full <= 1'b1;
            else if (st_done)  sh_full <= 1'b0;

            case (cstate)
                C_IDLE: if (layer_start) begin
                    cp_ifb <= 1'b0; cp_wb <= 1'b0; cmp_oc <= 0;
                    cstate <= C_WAIT;
                end

                C_WAIT: if (if_full[cp_ifb] && w_full[cp_wb]) cstate <= C_GO;
                C_GO:   cstate <= C_RUN;
                C_RUN:  if (compute_done) cstate <= C_CAP;

                // accumulators stay put until the shadow is free; capture + store start + release
                C_CAP: if (!sh_full) begin
                    cp_of_ptr <= st_addr_now + ofmap_ch_size;
                    cp_wb     <= ~cp_wb;
                    if (cmp_oc_last) begin
                        cmp_oc <= 0;
                        cp_ifb <= ~cp_ifb;
                        cstate <= no_more_tiles ? C_FIN : C_WAIT;
                    end else begin
                        cmp_oc <= cmp_oc + 1'b1;
                        cstate <= C_WAIT;
                    end
                end

                C_FIN: if (!sh_full) begin
                    done       <= 1'b1;
                    layer_busy <= 1'b0;
                    cstate     <= C_IDLE;
                end

                default: cstate <= C_IDLE;
            endcase
        end
    end

endmodule