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
    input  wire start,          // start one layer
    output reg  done,           // layer finished (1-cycle pulse)

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
    output wire [NUM_PE-1:0]       pe_mask,

    // ---------------- register files ----------------
    output wire                        ifrf_flush,
    output wire                        wrf_flush,
    output wire [NUM_MACROS-1:0]       ifrf_wr_en,
    output wire [NUM_MACROS-1:0]       wrf_wr_en,
    output wire [NUM_MACROS*RF_AW-1:0] rf_wr_addr,

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

    // patch_cols = (PE_COLS-1)*stride + kernel_cols  (constant multiplier = shift-add)
    wire [DIM_WIDTH-1:0] patch_cols = (PE_COLS - 1) * stride + kernel_cols;

    // ------------------------------------------------------------------
    // FSM
    // ------------------------------------------------------------------
    localparam [3:0] S_IDLE       = 4'd0,
                     S_TILE_FLUSH = 4'd1,
                     S_LDP_GO     = 4'd2,
                     S_LDP_WAIT   = 4'd3,
                     S_OC_FLUSH   = 4'd4,
                     S_LDW_GO     = 4'd5,
                     S_LDW_WAIT   = 4'd6,
                     S_CMP_GO     = 4'd7,
                     S_CMP_WAIT   = 4'd8,
                     S_ST_GO      = 4'd9,
                     S_ST_WAIT    = 4'd10,
                     S_NEXT_OC    = 4'd11,
                     S_NEXT_TILE  = 4'd12,
                     S_CHECK      = 4'd13,
                     S_FINISH     = 4'd14;

    reg [3:0] state;

    reg [DIM_WIDTH-1:0]  oc;
    reg [ADDR_WIDTH-1:0] oc_w_ptr;     // weight words of the current output channel
    reg [ADDR_WIDTH-1:0] oc_of_ptr;    // OFMap tile origin of the current output channel

    assign ph_load    = (state == S_LDP_GO) || (state == S_LDP_WAIT) ||
                        (state == S_LDW_GO) || (state == S_LDW_WAIT);
    assign ph_compute = (state == S_CMP_GO) || (state == S_CMP_WAIT);
    assign ph_store   = (state == S_ST_GO)  || (state == S_ST_WAIT);

    // ------------------------------------------------------------------
    // 1. Tile scheduler
    // ------------------------------------------------------------------
    wire                  ts_start   = (state == S_IDLE) && start;
    wire                  ts_advance = (state == S_NEXT_TILE);
    wire                  ts_valid;
    wire [DIM_WIDTH-1:0]  ts_row, ts_col;
    wire [ADDR_WIDTH-1:0] ts_ifmap_origin, ts_ofmap_origin;
    wire [VR_W-1:0]       ts_vrows;
    wire [VC_W-1:0]       ts_vcols;

    tile_scheduler #(
        .ADDR_WIDTH(ADDR_WIDTH), .DIM_WIDTH(DIM_WIDTH),
        .PE_ROWS(PE_ROWS), .PE_COLS(PE_COLS)
    ) u_tile (
        .clk(clk), .rst(rst),
        .start(ts_start), .advance(ts_advance),
        .ofmap_rows(ofmap_rows), .ofmap_cols(ofmap_cols), .stride(stride),
        .ifmap_row_step(ifmap_row_step),
        .ifmap_base_addr(ifmap_base_addr), .ofmap_base_addr(ofmap_base_addr),
        .tile_valid(ts_valid), .tile_row(ts_row), .tile_col(ts_col),
        .ifmap_origin(ts_ifmap_origin), .ofmap_origin(ts_ofmap_origin),
        .valid_rows(ts_vrows), .valid_cols(ts_vcols), .pe_mask(pe_mask)
    );

    // ------------------------------------------------------------------
    // 2. Compute sequencer
    // ------------------------------------------------------------------
    assign compute_start = (state == S_CMP_GO);

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
    wire ld_start = (state == S_LDP_GO) || (state == S_LDW_GO);
    wire ld_job_w = (state == S_LDW_GO);
    wire ld_busy, ld_done;
    wire [NUM_MACROS-1:0]          ld_req;
    wire [NUM_MACROS*MACRO_AW-1:0] ld_addr;
    wire                           ld_wr_weight;
    wire [NUM_MACROS-1:0]          ld_wr_en;

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
        .nmc_req(ld_req), .nmc_addr(ld_addr), .nmc_grant(nmc_grant),
        .rf_wr_en(ld_wr_en), .rf_wr_addr(rf_wr_addr), .rf_wr_weight(ld_wr_weight),
        .busy(ld_busy), .done(ld_done)
    );

    assign ifrf_wr_en = ld_wr_en & {NUM_MACROS{~ld_wr_weight}};
    assign wrf_wr_en  = ld_wr_en & {NUM_MACROS{ ld_wr_weight}};

    // ------------------------------------------------------------------
    // 4. Store / reload generator
    //    (reload mode is wired once the loop order becomes configurable)
    // ------------------------------------------------------------------
    wire st_start = (state == S_ST_GO);
    wire st_busy, st_done;
    wire [NUM_MACROS-1:0]          st_req, st_we;
    wire [NUM_MACROS*MACRO_AW-1:0] st_addr;

    store_reload_generator #(
        .ADDR_WIDTH(ADDR_WIDTH), .DIM_WIDTH(DIM_WIDTH),
        .NUM_MACROS(NUM_MACROS), .MACRO_AW(MACRO_AW),
        .PE_ROWS(PE_ROWS), .PE_COLS(PE_COLS)
    ) u_store (
        .clk(clk), .rst(rst),
        .start(st_start), .reload(1'b0),
        .ofmap_cols(ofmap_cols), .ofmap_tile_addr(oc_of_ptr),
        .valid_rows(ts_vrows), .valid_cols(ts_vcols),
        .nmc_req(st_req), .nmc_we(st_we), .nmc_addr(st_addr), .nmc_grant(nmc_grant),
        .st_lane_sel(st_lane_sel),
        .rl_wr_en(), .rl_wr_lane(),
        .busy(st_busy), .done(st_done)
    );

    // ------------------------------------------------------------------
    // NMC port: only one of load / store is ever busy
    // ------------------------------------------------------------------
    assign nmc_req  = ld_req | st_req;
    assign nmc_we   = st_busy ? st_we   : {NUM_MACROS{1'b0}};
    assign nmc_addr = st_busy ? st_addr : ld_addr;

    // ------------------------------------------------------------------
    // Flushes
    // ------------------------------------------------------------------
    assign ifrf_flush = (state == S_TILE_FLUSH);
    assign wrf_flush  = (state == S_OC_FLUSH);

    // ------------------------------------------------------------------
    // Master FSM
    // ------------------------------------------------------------------
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S_IDLE; done <= 1'b0;
            oc <= 0; oc_w_ptr <= 0; oc_of_ptr <= 0;
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE:       if (start) state <= S_TILE_FLUSH;

                S_TILE_FLUSH: begin
                    oc        <= 0;
                    oc_w_ptr  <= weight_base_addr;
                    oc_of_ptr <= ts_ofmap_origin;
                    state     <= S_LDP_GO;
                end

                S_LDP_GO:     state <= S_LDP_WAIT;
                S_LDP_WAIT:   if (ld_done) state <= S_OC_FLUSH;

                S_OC_FLUSH:   state <= S_LDW_GO;
                S_LDW_GO:     state <= S_LDW_WAIT;
                S_LDW_WAIT:   if (ld_done) state <= S_CMP_GO;

                S_CMP_GO:     state <= S_CMP_WAIT;
                S_CMP_WAIT:   if (compute_done) state <= S_ST_GO;

                S_ST_GO:      state <= S_ST_WAIT;
                S_ST_WAIT:    if (st_done) state <= S_NEXT_OC;

                S_NEXT_OC: begin
                    if (oc + 1 < output_channels) begin
                        oc        <= oc + 1'b1;
                        oc_w_ptr  <= oc_w_ptr  + weights_per_oc;
                        oc_of_ptr <= oc_of_ptr + ofmap_ch_size;
                        state     <= S_OC_FLUSH;
                    end else begin
                        state     <= S_NEXT_TILE;
                    end
                end

                S_NEXT_TILE:  state <= S_CHECK;
                S_CHECK:      state <= ts_valid ? S_TILE_FLUSH : S_FINISH;

                S_FINISH: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default:      state <= S_IDLE;
            endcase
        end
    end

endmodule