// ============================================================================
// NMC unit  (one per memory bank)
//
//   layer descriptor --> address_generator --+--> RF write side (IFMap RF, weight RF)
//                                            +--> access_controller --> PE array
//                                            +--> NMC memory port (loads / stores)
//
// Contents:
//   - address_generator      tile scheduler, compute sequencer, load + store generators
//   - register_file x2       IFMap RF (8-bit, 16 read ports) and weight RF (8-bit, 1 read port)
//                            (paper: two multi-port RFs; the psum RF joins when reload mode is wired)
//   - access_controller      MAC / reload handshake
//   - pe_array               16 PEs, weight-broadcast, output-stationary
//   - store data mux         nmc_wdata[m] = accumulator selected by st_lane_sel[m]
//
// The memory side is the per-macro NMC port of sram_bank:
//   nmc_req/we/addr/wdata out, nmc_grant/rdata in. nmc_rdata[m][7:0] feeds the RFs.
//
// dbg_* outputs expose internal strobes for testbenches / future perf counters.
// ============================================================================
module nmc_unit #(
    parameter ADDR_WIDTH = 17,
    parameter DIM_WIDTH  = 16,
    parameter RF_AW      = 8,
    parameter NUM_MACROS = 8,
    parameter MACRO_AW   = 12,
    parameter PE_ROWS    = 4,
    parameter PE_COLS    = 4,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32,
    parameter NUM_PE     = PE_ROWS * PE_COLS,
    parameter LANE_W     = $clog2(NUM_PE)
)(
    input  wire clk,
    input  wire rst,
    input  wire start,
    output wire done,

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

    // ---------------- NMC memory port (to sram_bank) ----------------
    output wire [NUM_MACROS-1:0]            nmc_req,
    output wire [NUM_MACROS-1:0]            nmc_we,
    output wire [NUM_MACROS*MACRO_AW-1:0]   nmc_addr,
    output wire [NUM_MACROS*32-1:0]         nmc_wdata,
    input  wire [NUM_MACROS-1:0]            nmc_grant,
    input  wire [NUM_MACROS*32-1:0]         nmc_rdata,

    // ---------------- debug / observability ----------------
    output wire                             dbg_compute_start,
    output wire                             dbg_compute_done,
    output wire                             dbg_mac_en,
    output wire                             dbg_stall,
    output wire [NUM_PE*ACC_WIDTH-1:0]      dbg_acc_out
);

    // ---------------- address generator ----------------
    wire                    compute_start, compute_done, seq_enable, seq_valid, seq_last;
    wire [NUM_PE*RF_AW-1:0] ifmap_rf_addr;
    wire [RF_AW-1:0]        weight_rf_addr;
    wire [NUM_PE-1:0]       pe_mask;
    wire                    ifrf_flush, wrf_flush;
    wire [NUM_MACROS-1:0]   ifrf_wr_en, wrf_wr_en;
    wire [NUM_MACROS*RF_AW-1:0] rf_wr_addr;
    wire [NUM_MACROS*LANE_W-1:0] st_lane_sel;

    address_generator #(
        .ADDR_WIDTH(ADDR_WIDTH), .DIM_WIDTH(DIM_WIDTH), .RF_AW(RF_AW),
        .NUM_MACROS(NUM_MACROS), .MACRO_AW(MACRO_AW),
        .PE_ROWS(PE_ROWS), .PE_COLS(PE_COLS)
    ) u_ag (
        .clk(clk), .rst(rst), .start(start), .done(done),
        .ifmap_cols(ifmap_cols), .kernel_rows(kernel_rows), .kernel_cols(kernel_cols),
        .input_channels(input_channels), .output_channels(output_channels), .stride(stride),
        .ofmap_rows(ofmap_rows), .ofmap_cols(ofmap_cols),
        .ifmap_ch_size(ifmap_ch_size), .ofmap_ch_size(ofmap_ch_size), .ifmap_row_step(ifmap_row_step),
        .rf_row_step(rf_row_step), .rf_ch_stride(rf_ch_stride), .weights_per_oc(weights_per_oc),
        .ifmap_base_addr(ifmap_base_addr), .weight_base_addr(weight_base_addr), .ofmap_base_addr(ofmap_base_addr),
        .compute_start(compute_start), .compute_done(compute_done),
        .seq_enable(seq_enable), .seq_valid(seq_valid), .seq_last(seq_last),
        .ifmap_rf_addr(ifmap_rf_addr), .weight_rf_addr(weight_rf_addr), .pe_mask(pe_mask),
        .ifrf_flush(ifrf_flush), .wrf_flush(wrf_flush),
        .ifrf_wr_en(ifrf_wr_en), .wrf_wr_en(wrf_wr_en), .rf_wr_addr(rf_wr_addr),
        .nmc_req(nmc_req), .nmc_we(nmc_we), .nmc_addr(nmc_addr), .nmc_grant(nmc_grant),
        .st_lane_sel(st_lane_sel)
    );

    // ---------------- register files (RF write data = low byte of the macro read data) ----------------
    wire [NUM_MACROS*DATA_WIDTH-1:0] rf_wdata;
    genvar gi;
    generate for (gi = 0; gi < NUM_MACROS; gi = gi + 1) begin : g_wd
        assign rf_wdata[gi*DATA_WIDTH +: DATA_WIDTH] = nmc_rdata[gi*32 +: DATA_WIDTH];
    end endgenerate

    wire [NUM_PE-1:0]                ifrf_read_en;
    wire                             wrf_read_en;
    wire [NUM_PE*DATA_WIDTH-1:0]     if_rd;
    wire [NUM_PE-1:0]                if_rv;
    wire [NUM_MACROS-1:0]            if_blk, w_blk;
    wire [DATA_WIDTH-1:0]            w_rd;
    wire [0:0]                       w_rv;

    register_file #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(1<<RF_AW),
                    .NUM_READ_PORTS(NUM_PE), .NUM_WRITE_PORTS(NUM_MACROS)) u_ifrf (
        .clk(clk), .rst(rst), .flush(ifrf_flush),
        .write_en(ifrf_wr_en), .write_addr(rf_wr_addr), .write_data(rf_wdata), .write_blocked(if_blk),
        .read_en(ifrf_read_en), .read_last({NUM_PE{1'b0}}),
        .read_addr(ifmap_rf_addr), .read_data(if_rd), .read_valid(if_rv));

    register_file #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(1<<RF_AW),
                    .NUM_READ_PORTS(1), .NUM_WRITE_PORTS(NUM_MACROS)) u_wrf (
        .clk(clk), .rst(rst), .flush(wrf_flush),
        .write_en(wrf_wr_en), .write_addr(rf_wr_addr), .write_data(rf_wdata), .write_blocked(w_blk),
        .read_en(wrf_read_en), .read_last(1'b0),
        .read_addr(weight_rf_addr), .read_data(w_rd), .read_valid(w_rv));

    // ---------------- access controller + PE array ----------------
    wire pe_clear, pe_mac_en, pe_psum_load, ac_stall, psum_ack;
    wire [NUM_PE-1:0] psrf_read_en, psrf_read_last;

    access_controller #(.NUM_PE(NUM_PE)) u_ac (
        .clk(clk), .rst(rst),
        .compute_start(compute_start), .seq_valid(seq_valid), .seq_last(seq_last), .pe_mask(pe_mask),
        .seq_enable(seq_enable), .compute_done(compute_done),
        .ifrf_read_valid(if_rv), .wrf_read_valid(w_rv[0]),
        .ifrf_read_en(ifrf_read_en), .wrf_read_en(wrf_read_en),
        .psum_req(1'b0), .psrf_read_valid({NUM_PE{1'b0}}),     // reload path: psum RF not wired yet
        .psrf_read_en(psrf_read_en), .psrf_read_last(psrf_read_last), .psum_ack(psum_ack),
        .pe_clear(pe_clear), .pe_mac_en(pe_mac_en), .pe_psum_load(pe_psum_load), .stall(ac_stall));

    wire signed [NUM_PE*ACC_WIDTH-1:0] acc_out;
    pe_array #(.NUM_PE(NUM_PE), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH), .WEIGHT_BROADCAST(1)) u_pea (
        .clk(clk), .rst(rst), .clear(pe_clear), .pe_mask(pe_mask),
        .psum_load(pe_psum_load), .mac_en(pe_mac_en), .psum_in({NUM_PE*ACC_WIDTH{1'b0}}),
        .ifmap_in(if_rd), .weight_in(w_rd), .acc_out(acc_out));

    // ---------------- store data mux ----------------
    generate for (gi = 0; gi < NUM_MACROS; gi = gi + 1) begin : g_st
        wire [LANE_W-1:0] sel = st_lane_sel[gi*LANE_W +: LANE_W];
        assign nmc_wdata[gi*32 +: 32] = acc_out[sel*ACC_WIDTH +: ACC_WIDTH];
    end endgenerate

    // ---------------- debug ----------------
    assign dbg_compute_start = compute_start;
    assign dbg_compute_done  = compute_done;
    assign dbg_mac_en        = pe_mac_en;
    assign dbg_stall         = ac_stall;
    assign dbg_acc_out       = acc_out;

endmodule