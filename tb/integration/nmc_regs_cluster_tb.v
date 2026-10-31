`timescale 1ns/1ps
// ============================================================================
// nmc_regs + nmc_cluster, driven ONLY through the register bus (data goes in/out
// through the cluster's host core port, like the other cluster testbenches).
//
//   1. conv round on bank 0: 6x6 IFMap, 3x3 kernel -> 4x4 OFMap (milestone-1 layer)
//      checks OFMap words, irq_o, STATUS, perf counters read over the bus
//   2. post-proc round on bank 0: 2x2 max-pool + requant (bias 0, mult 1, shift 0)
//      + ReLU, requant params written through QP_* registers; checks int8 output
//
// Run (from the repo root, like the other tests):
//   iverilog -g2012 -o sim/build/nmc_regs_cluster.vvp tb/integration/nmc_regs_cluster_tb.v \
//            $(find rtl -name '*.v') && vvp sim/build/nmc_regs_cluster.vvp
// ============================================================================
module nmc_regs_cluster_tb;
    localparam NB = 4, ADDR_WIDTH = 17, DIM_WIDTH = 16, RF_AW = 8, NM = 8;
    localparam DEPTH_PM = 4096, MAW = 12, CORE_AW = 3 + MAW, BANK_W = 2, CNT_W = 48, QA_W = 9;

    localparam IFM_BASE = 0, W_BASE = 3000, OFM_BASE = 5000, DST_BASE = 8000;
    localparam IFM = 6, K = 3, OFM = 4;

    reg clk = 0, rst = 1;  always #5 clk = ~clk;

    // ---------------- register bus ----------------
    reg         reg_req = 0, reg_we = 0;
    reg  [11:0] reg_addr = 0;
    reg  [31:0] reg_wdata = 0;
    wire        reg_rvalid, reg_err, irq_o;
    wire [31:0] reg_rdata;

    // ---------------- regs <-> cluster wires (names match the ports) ----------------
    wire [NB-1:0] unit_mask, accumulate, unit_busy, pp_mask, pp_busy, pp_relu_en, pp_qp_we, perf_valid;
    wire start, irq, pp_start, pp_irq;
    wire [NB*DIM_WIDTH-1:0]  ifmap_cols, kernel_rows, kernel_cols, input_channels, output_channels, stride,
                             ofmap_rows, ofmap_cols, weights_per_oc,
                             pp_src_cols, pp_out_rows, pp_out_cols, pp_channels, pp_pool_k;
    wire [NB*ADDR_WIDTH-1:0] ifmap_ch_size, ofmap_ch_size, ifmap_row_step, ifmap_base_addr, weight_base_addr,
                             ofmap_base_addr, pp_src_base, pp_dst_base, pp_src_ch_size, pp_row_step, pp_total_out;
    wire [NB*RF_AW-1:0]      rf_row_step, rf_ch_stride;
    wire [NB*8-1:0]          pp_out_zp;
    wire [QA_W-1:0]          pp_qp_waddr;
    wire signed [31:0]       pp_qp_bias;
    wire [15:0]              pp_qp_mult;
    wire [4:0]               pp_qp_shift;
    wire [NB*CNT_W-1:0]      perf_cyc_total, perf_cyc_mac, perf_cyc_stall, perf_cyc_mem_blocked, perf_mac_ops;

    nmc_regs #(.NUM_BANKS(NB), .ADDR_WIDTH(ADDR_WIDTH), .DIM_WIDTH(DIM_WIDTH), .RF_AW(RF_AW),
               .CNT_W(CNT_W), .MAX_OC(512)) u_regs (
        .clk(clk), .rst(rst),
        .reg_req(reg_req), .reg_we(reg_we), .reg_addr(reg_addr), .reg_wdata(reg_wdata),
        .reg_rvalid(reg_rvalid), .reg_rdata(reg_rdata), .reg_err(reg_err), .irq_o(irq_o),
        .*);

    // host data port (core_*) straight from the testbench
    reg                      core_req = 0, core_we = 0;
    reg [BANK_W+CORE_AW-1:0] core_addr = 0;
    reg [31:0]               core_wdata = 0;
    wire [31:0]              core_rdata;
    wire                     core_rvalid;

    nmc_cluster #(.NUM_BANKS(NB), .ADDR_WIDTH(ADDR_WIDTH), .DIM_WIDTH(DIM_WIDTH), .RF_AW(RF_AW),
                  .NUM_MACROS(NM), .DEPTH_PER_MACRO(DEPTH_PM), .CNT_W(CNT_W), .MAX_OC(512)) u_cluster (
        .clk(clk), .rst(rst),
        .core_req(core_req), .core_we(core_we), .core_addr(core_addr), .core_wdata(core_wdata),
        .core_rdata(core_rdata), .core_rvalid(core_rvalid),
        .*);

    // ---------------- helpers ----------------
    integer errors = 0;
    reg [31:0] rd;

    task check(input [255:0] name, input [63:0] got, input [63:0] exp);
        if (got !== exp) begin $display("FAIL %0s: got %0h expected %0h", name, got, exp); errors = errors + 1; end
    endtask

    function [11:0] BA(input integer b, input integer f);   // word f of bank b's register block
        BA = 12'h100 * (b + 1) + f * 4;
    endfunction

    task bus_write(input [11:0] a, input [31:0] d);
        begin
            @(negedge clk); reg_req = 1; reg_we = 1; reg_addr = a; reg_wdata = d;
            @(negedge clk); reg_req = 0; reg_we = 0;
            if (reg_err) begin $display("FAIL bus error on write %0h", a); errors = errors + 1; end
        end
    endtask
    task bus_read(input [11:0] a, output [31:0] d);
        begin
            @(negedge clk); reg_req = 1; reg_we = 0; reg_addr = a;
            @(negedge clk); reg_req = 0;
            d = reg_rdata;
            if (reg_err) begin $display("FAIL bus error on read %0h", a); errors = errors + 1; end
        end
    endtask
    task wait_irq;
        integer t;
        begin
            t = 0;
            while (!irq_o && t < 2000000) begin @(negedge clk); t = t + 1; end
            if (!irq_o) begin $display("TIMEOUT waiting for irq_o"); errors = errors + 1; $finish; end
            @(negedge clk);
        end
    endtask

    task core_write(input [BANK_W+CORE_AW-1:0] a, input [31:0] d);
        begin
            @(negedge clk); core_req = 1; core_we = 1; core_addr = a; core_wdata = d;
            @(negedge clk); core_req = 0; core_we = 0;
        end
    endtask
    task core_read(input [BANK_W+CORE_AW-1:0] a, output [31:0] d);
        begin
            @(negedge clk); core_req = 1; core_we = 0; core_addr = a;
            @(negedge clk); core_req = 0;
            d = core_rdata;
        end
    endtask

    // ---------------- layer data ----------------
    reg signed [7:0]  ifm [0:IFM*IFM-1];
    reg signed [7:0]  wgt [0:K*K-1];
    integer           exp_o [0:OFM*OFM-1];
    integer           exp_y [0:3];
    integer i, r, c, ki, kj, m;

    initial begin
        $dumpfile("sim/waveform/nmc_regs_cluster_wave.vcd");
        $dumpvars(0, nmc_regs_cluster_tb);
        #22 rst = 0;

        for (i = 0; i < IFM*IFM; i = i + 1) ifm[i] = ($random & 15) - 8;
        for (i = 0; i < K*K;     i = i + 1) wgt[i] = ($random & 7) - 4;
        for (r = 0; r < OFM; r = r + 1) for (c = 0; c < OFM; c = c + 1) begin
            exp_o[r*OFM+c] = 0;
            for (ki = 0; ki < K; ki = ki + 1) for (kj = 0; kj < K; kj = kj + 1)
                exp_o[r*OFM+c] = exp_o[r*OFM+c] + ifm[(r+ki)*IFM + (c+kj)] * wgt[ki*K+kj];
        end
        // expected post-proc: 2x2 max-pool, then clamp(0..127) (bias 0, mult 1, shift 0, zp 0, ReLU)
        for (r = 0; r < 2; r = r + 1) for (c = 0; c < 2; c = c + 1) begin
            m = exp_o[(2*r)*OFM + 2*c];
            if (exp_o[(2*r)*OFM + 2*c+1]   > m) m = exp_o[(2*r)*OFM + 2*c+1];
            if (exp_o[(2*r+1)*OFM + 2*c]   > m) m = exp_o[(2*r+1)*OFM + 2*c];
            if (exp_o[(2*r+1)*OFM + 2*c+1] > m) m = exp_o[(2*r+1)*OFM + 2*c+1];
            exp_y[r*2+c] = (m < 0) ? 0 : (m > 127) ? 127 : m;
        end

        // ---- host fills bank 0 over the data port ----
        for (i = 0; i < IFM*IFM; i = i + 1) core_write(IFM_BASE + i, {24'b0, ifm[i]});
        for (i = 0; i < K*K;     i = i + 1) core_write(W_BASE + i,   {24'b0, wgt[i]});
        for (i = 0; i < OFM*OFM + 16; i = i + 1) core_write(OFM_BASE + i, 32'hDEADBEEF);
        for (i = 0; i < 4 + 16;       i = i + 1) core_write(DST_BASE + i, 32'hDEADBEEF);

        bus_read(12'h000, rd);  check("ID", rd, 32'h4E4D0001);

        // ================= 1. conv round, all configuration over the bus =================
        bus_write(BA(0, 0),  IFM);            // IFMAP_COLS
        bus_write(BA(0, 1),  K);              // KERNEL_ROWS
        bus_write(BA(0, 2),  K);              // KERNEL_COLS
        bus_write(BA(0, 3),  1);              // INPUT_CHANNELS
        bus_write(BA(0, 4),  1);              // OUTPUT_CHANNELS
        bus_write(BA(0, 5),  1);              // STRIDE
        bus_write(BA(0, 6),  OFM);            // OFMAP_ROWS
        bus_write(BA(0, 7),  OFM);            // OFMAP_COLS
        bus_write(BA(0, 8),  K*K);            // WEIGHTS_PER_OC
        bus_write(BA(0, 9),  IFM*IFM);        // IFMAP_CH_SIZE
        bus_write(BA(0, 10), OFM*OFM);        // OFMAP_CH_SIZE
        bus_write(BA(0, 11), IFM);            // IFMAP_ROW_STEP
        bus_write(BA(0, 12), IFM_BASE);       // IFMAP_BASE
        bus_write(BA(0, 13), W_BASE);         // WEIGHT_BASE
        bus_write(BA(0, 14), OFM_BASE);       // OFMAP_BASE
        bus_write(BA(0, 15), IFM);            // RF_ROW_STEP  (stride * patch_cols = 6)
        bus_write(BA(0, 16), IFM*IFM);        // RF_CH_STRIDE (patch 6x6)
        bus_write(BA(0, 17), 0);              // ACCUMULATE
        bus_write(12'h08, 32'h1);             // UNIT_MASK = bank 0
        bus_write(12'h24, 32'h3);             // IRQ_ENABLE = conv done, pp done
        bus_write(12'h04, 32'h1);             // CTRL.CONV_START
        bus_read (12'h10, rd);  check("STATUS conv active", rd, 1);
        wait_irq;
        bus_read (12'h10, rd);  check("STATUS idle after irq", rd, 0);
        bus_read (12'h20, rd);  check("IRQ_STATUS conv done", rd, 1);
        bus_write(12'h20, 32'h1);
        check("irq_o cleared", irq_o, 0);
        bus_read (12'h1C, rd);  check("PERF_VALID bank 0", rd, 1);
        bus_read (BA(0, 34), rd); check("MAC cycles (9)", rd, 9);
        bus_read (BA(0, 40), rd); check("MAC ops (9*16)", rd, 144);
        bus_read (BA(0, 36), rd); check("stalls", rd, 0);
        bus_read (BA(0, 32), rd); $display("      bank0 total cycles over the bus = %0d", rd);

        for (i = 0; i < OFM*OFM; i = i + 1) begin
            core_read(OFM_BASE + i, rd);
            check("OFMap word", rd, exp_o[i] & 32'hFFFF_FFFF);
        end
        for (i = 0; i < 16; i = i + 1) begin
            core_read(OFM_BASE + OFM*OFM + i, rd);
            check("OFMap guard", rd, 32'hDEADBEEF);
        end

        // ================= 2. post-proc round (pool 2, requant, ReLU) =================
        bus_write(BA(0, 20), OFM_BASE);       // PP_SRC_BASE
        bus_write(BA(0, 21), DST_BASE);       // PP_DST_BASE
        bus_write(BA(0, 22), OFM);            // PP_SRC_COLS
        bus_write(BA(0, 23), 2);              // PP_OUT_ROWS
        bus_write(BA(0, 24), 2);              // PP_OUT_COLS
        bus_write(BA(0, 25), 1);              // PP_CHANNELS
        bus_write(BA(0, 26), 2);              // PP_POOL_K
        bus_write(BA(0, 27), OFM*OFM);        // PP_SRC_CH_SIZE
        bus_write(BA(0, 28), 2*OFM);          // PP_ROW_STEP (pool_k * src_cols)
        bus_write(BA(0, 29), 4);              // PP_TOTAL_OUT
        bus_write(BA(0, 30), 0);              // PP_OUT_ZP
        bus_write(BA(0, 31), 1);              // PP_RELU_EN
        bus_write(12'h40, 32'h1);             // QP_BANK_MASK = bank 0
        bus_write(12'h44, 0);                 // QP_ADDR = channel 0
        bus_write(12'h48, 0);                 // QP_BIAS
        bus_write(12'h4C, 1);                 // QP_MULT
        bus_write(12'h50, 0);                 // QP_SHIFT (commit)
        bus_write(12'h0C, 32'h1);             // PP_MASK = bank 0
        bus_write(12'h04, 32'h2);             // CTRL.PP_START
        bus_read (12'h10, rd);  check("STATUS pp active", rd, 2);
        wait_irq;
        bus_read (12'h20, rd);  check("IRQ_STATUS pp done", rd, 2);
        bus_write(12'h20, 32'hF);
        bus_read (12'h10, rd);  check("STATUS idle after pp irq", rd, 0);

        for (i = 0; i < 4; i = i + 1) begin
            core_read(DST_BASE + i, rd);
            check("pp output (int8 in low byte)", rd, exp_y[i]);
        end
        for (i = 0; i < 16; i = i + 1) begin
            core_read(DST_BASE + 4 + i, rd);
            check("pp guard", rd, 32'hDEADBEEF);
        end

        if (errors == 0) $display("\nnmc_regs + cluster: ALL TESTS PASSED");
        else             $display("\nnmc_regs + cluster: %0d ERRORS", errors);
        $finish;
    end
endmodule