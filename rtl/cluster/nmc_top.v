// ============================================================================
// NMC subsystem top  (nmc_regs + nmc_cluster behind ONE bus slave port)
//
//   SoC interconnect --bus--> nmc_top --+--> nmc_regs     (control, descriptors, perf)
//                                       +--> nmc_cluster  (banked SRAM, host data port;
//                                                          4 x nmc_unit + postproc)
//                             irq_o  <--+
//
// One 32-bit slave port, single cycle, no wait states, byte addresses:
//   bus_addr[MEM_BYTE_AW]   = 0 : SRAM window, byte offset = {bank, word-in-bank, 2'b00}
//                                 (default 4 x 128 KB = 512 KB at 0x00000 - 0x7FFFF)
//                           = 1 : register block (nmc_regs, 4 KB) at 0x80000
//   bus_rvalid  : 1 cycle after every request
//   bus_rdata   : read data (0 for writes)
//   bus_err     : register-block errors (unmapped / read-only / reserved)
//
// The RISC-V (or a DMA) accesses SRAM here exactly like the host core port of
// the cluster; NMC units and the host contend per macro inside sram_bank (host wins).
// irq_o is level: connect to the core's interrupt / event input, clear through
// IRQ_STATUS (W1C).
// ============================================================================
module nmc_top #(
    parameter NUM_BANKS       = 4,
    parameter ADDR_WIDTH      = 17,
    parameter DIM_WIDTH       = 16,
    parameter RF_AW           = 8,
    parameter NUM_MACROS      = 8,
    parameter DEPTH_PER_MACRO = 4096,
    parameter CNT_W           = 48,
    parameter MAX_OC          = 512,
    parameter MACRO_AW        = $clog2(DEPTH_PER_MACRO),
    parameter QA_W            = $clog2(MAX_OC),
    parameter BANK_W          = $clog2(NUM_BANKS),
    parameter CORE_AW         = $clog2(NUM_MACROS) + MACRO_AW,
    parameter MEM_BYTE_AW     = BANK_W + CORE_AW + 2,      // 19 -> 512 KB
    parameter REG_AW          = 12,
    parameter TOP_AW          = MEM_BYTE_AW + 1
)(
    input  wire                 clk,
    input  wire                 rst,

    input  wire                 bus_req,
    input  wire                 bus_we,
    input  wire [TOP_AW-1:0]    bus_addr,
    input  wire [31:0]          bus_wdata,
    output reg                  bus_rvalid,
    output wire [31:0]          bus_rdata,
    output wire                 bus_err,

    output wire                 irq_o
);

    wire is_reg = bus_addr[MEM_BYTE_AW];

    // ---------------- request routing ----------------
    wire                      reg_req  = bus_req &  is_reg;
    wire                      core_req = bus_req & ~is_reg;
    wire [BANK_W+CORE_AW-1:0] core_addr = bus_addr[2 +: BANK_W+CORE_AW];

    // ---------------- response mux (both targets answer one cycle later) ----------------
    reg  sel_q, we_q;
    always @(posedge clk or posedge rst)
        if (rst) begin bus_rvalid <= 1'b0; sel_q <= 1'b0; we_q <= 1'b0; end
        else     begin bus_rvalid <= bus_req; sel_q <= is_reg; we_q <= bus_we; end

    wire [31:0] reg_rdata, core_rdata;
    wire        reg_rvalid, reg_err, core_rvalid;     // reg_rvalid / core_rvalid unused (own rvalid above)
    assign bus_rdata = we_q ? 32'h0 : (sel_q ? reg_rdata : core_rdata);
    assign bus_err   = sel_q & reg_err;

    // ---------------- regs <-> cluster wires (names match the ports) ----------------
    wire [NUM_BANKS-1:0] unit_mask, accumulate, unit_busy, pp_mask, pp_busy, pp_relu_en, pp_qp_we, perf_valid;
    wire start, irq, pp_start, pp_irq;
    wire [NUM_BANKS*DIM_WIDTH-1:0]  ifmap_cols, kernel_rows, kernel_cols, input_channels, output_channels, stride,
                                    ofmap_rows, ofmap_cols, weights_per_oc,
                                    pp_src_cols, pp_out_rows, pp_out_cols, pp_channels, pp_pool_k;
    wire [NUM_BANKS*ADDR_WIDTH-1:0] ifmap_ch_size, ofmap_ch_size, ifmap_row_step, ifmap_base_addr, weight_base_addr,
                                    ofmap_base_addr, pp_src_base, pp_dst_base, pp_src_ch_size, pp_row_step, pp_total_out;
    wire [NUM_BANKS*RF_AW-1:0]      rf_row_step, rf_ch_stride;
    wire [NUM_BANKS*8-1:0]          pp_out_zp;
    wire [QA_W-1:0]                 pp_qp_waddr;
    wire signed [31:0]              pp_qp_bias;
    wire [15:0]                     pp_qp_mult;
    wire [4:0]                      pp_qp_shift;
    wire [NUM_BANKS*CNT_W-1:0]      perf_cyc_total, perf_cyc_mac, perf_cyc_stall, perf_cyc_mem_blocked, perf_mac_ops;

    nmc_regs #(.NUM_BANKS(NUM_BANKS), .ADDR_WIDTH(ADDR_WIDTH), .DIM_WIDTH(DIM_WIDTH), .RF_AW(RF_AW),
               .CNT_W(CNT_W), .MAX_OC(MAX_OC), .BUS_AW(REG_AW)) u_regs (
        .clk(clk), .rst(rst),
        .reg_req(reg_req), .reg_we(bus_we), .reg_addr(bus_addr[REG_AW-1:0]), .reg_wdata(bus_wdata),
        .reg_rvalid(reg_rvalid), .reg_rdata(reg_rdata), .reg_err(reg_err), .irq_o(irq_o),
        .*);

    nmc_cluster #(.NUM_BANKS(NUM_BANKS), .ADDR_WIDTH(ADDR_WIDTH), .DIM_WIDTH(DIM_WIDTH), .RF_AW(RF_AW),
                  .NUM_MACROS(NUM_MACROS), .DEPTH_PER_MACRO(DEPTH_PER_MACRO), .CNT_W(CNT_W),
                  .MAX_OC(MAX_OC)) u_cluster (
        .clk(clk), .rst(rst),
        .core_req(core_req), .core_we(bus_we), .core_addr(core_addr), .core_wdata(bus_wdata),
        .core_rdata(core_rdata), .core_rvalid(core_rvalid),
        .*);

endmodule