`timescale 1ns/1ps
// ============================================================================
// Self-checking unit test for nmc_regs against a behavioural cluster stub.
// The stub copies the cluster's round semantics (go = start && !running &&
// !pp_running && |mask, irq pulse at the end, perf_valid, per-bank requant RAM)
// but computes nothing. Checks: ID/CONFIG, descriptor write/readback/masking and
// fan-out, start rules, blocking while busy, W1C interrupts, 48-bit perf
// readout, requant commit + auto-increment, bus error responses.
// ============================================================================
module cluster_stub #(parameter NB = 4, CNT = 48, QA = 9, LAT = 40) (
    input  wire clk, rst,
    input  wire [NB-1:0] unit_mask, input wire start,
    output reg  irq, output wire [NB-1:0] unit_busy,
    input  wire [NB-1:0] pp_mask, input wire pp_start,
    output reg  pp_irq, output wire [NB-1:0] pp_busy,
    input  wire [NB-1:0] pp_qp_we, input wire [QA-1:0] pp_qp_waddr,
    input  wire signed [31:0] pp_qp_bias, input wire [15:0] pp_qp_mult, input wire [4:0] pp_qp_shift,
    output reg  [NB-1:0] perf_valid,
    output wire [NB*CNT-1:0] perf_cyc_total, perf_cyc_mac, perf_cyc_stall, perf_cyc_mem_blocked, perf_mac_ops
);
    reg running, pp_running;
    reg [15:0] cnt;
    reg [NB-1:0] bmask;
    wire go  = start    && !running && !pp_running && (|unit_mask);
    wire pgo = pp_start && !running && !pp_running && (|pp_mask);
    assign unit_busy = running    ? bmask : {NB{1'b0}};
    assign pp_busy   = pp_running ? bmask : {NB{1'b0}};

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            running <= 0; pp_running <= 0; irq <= 0; pp_irq <= 0; perf_valid <= 0; cnt <= 0; bmask <= 0;
        end else begin
            irq <= 0; pp_irq <= 0;
            if (go) begin running <= 1; bmask <= unit_mask; cnt <= LAT; perf_valid <= 0; end
            else if (pgo) begin pp_running <= 1; bmask <= pp_mask; cnt <= LAT; end
            else if (running || pp_running) begin
                if (cnt == 0) begin
                    if (running) begin irq <= 1; perf_valid <= bmask; end else pp_irq <= 1;
                    running <= 0; pp_running <= 0;
                end else cnt <= cnt - 1;
            end
        end
    end

    reg [52:0] qpm [0:NB*(1<<QA)-1];
    integer q;
    initial for (q = 0; q < NB*(1<<QA); q = q + 1) qpm[q] = 0;
    always @(posedge clk) for (q = 0; q < NB; q = q + 1)
        if (pp_qp_we[q]) qpm[q*(1<<QA) + pp_qp_waddr] <= {pp_qp_bias, pp_qp_mult, pp_qp_shift};

    genvar b;
    generate for (b = 0; b < NB; b = b + 1) begin : g_perf
        assign perf_cyc_total      [b*CNT +: CNT] = (b == 1) ? 48'h0001_0000_0009 : b*1000 + 7;
        assign perf_cyc_mac        [b*CNT +: CNT] = b*10 + 9;
        assign perf_cyc_stall      [b*CNT +: CNT] = b;
        assign perf_cyc_mem_blocked[b*CNT +: CNT] = 2*b;
        assign perf_mac_ops        [b*CNT +: CNT] = b*144;
    end endgenerate
endmodule

module nmc_regs_tb;
    localparam NB = 4, DW = 16, AW = 17, RFAW = 8, CNT = 48, QA = 9;

    reg clk = 0, rst = 1;  always #5 clk = ~clk;

    reg         reg_req = 0, reg_we = 0;
    reg  [11:0] reg_addr = 0;
    reg  [31:0] reg_wdata = 0;
    wire        reg_rvalid, reg_err, irq_o;
    wire [31:0] reg_rdata;

    wire [NB-1:0] unit_mask, accumulate, pp_mask, pp_relu_en, pp_qp_we, unit_busy, pp_busy, perf_valid;
    wire start, pp_start, irq, pp_irq;
    wire [NB*DW-1:0]  ifmap_cols, input_channels, pp_pool_k;
    wire [NB*AW-1:0]  ifmap_base_addr, pp_total_out;
    wire [NB*RFAW-1:0] rf_ch_stride;
    wire [NB*8-1:0]   pp_out_zp;
    wire [QA-1:0]     pp_qp_waddr;
    wire signed [31:0] pp_qp_bias;
    wire [15:0]       pp_qp_mult;
    wire [4:0]        pp_qp_shift;
    wire [NB*CNT-1:0] p_total, p_mac, p_stall, p_blk, p_ops;

    nmc_regs #(.NUM_BANKS(NB), .ADDR_WIDTH(AW), .DIM_WIDTH(DW), .RF_AW(RFAW), .CNT_W(CNT)) dut (
        .clk(clk), .rst(rst),
        .reg_req(reg_req), .reg_we(reg_we), .reg_addr(reg_addr), .reg_wdata(reg_wdata),
        .reg_rvalid(reg_rvalid), .reg_rdata(reg_rdata), .reg_err(reg_err), .irq_o(irq_o),
        .unit_mask(unit_mask), .start(start), .accumulate(accumulate), .irq(irq), .unit_busy(unit_busy),
        .ifmap_cols(ifmap_cols), .input_channels(input_channels),
        .ifmap_base_addr(ifmap_base_addr), .rf_ch_stride(rf_ch_stride),
        .pp_mask(pp_mask), .pp_start(pp_start), .pp_irq(pp_irq), .pp_busy(pp_busy),
        .pp_pool_k(pp_pool_k), .pp_total_out(pp_total_out), .pp_out_zp(pp_out_zp), .pp_relu_en(pp_relu_en),
        .pp_qp_we(pp_qp_we), .pp_qp_waddr(pp_qp_waddr), .pp_qp_bias(pp_qp_bias),
        .pp_qp_mult(pp_qp_mult), .pp_qp_shift(pp_qp_shift),
        .perf_valid(perf_valid), .perf_cyc_total(p_total), .perf_cyc_mac(p_mac),
        .perf_cyc_stall(p_stall), .perf_cyc_mem_blocked(p_blk), .perf_mac_ops(p_ops));

    cluster_stub #(.NB(NB), .CNT(CNT), .QA(QA)) stub (
        .clk(clk), .rst(rst), .unit_mask(unit_mask), .start(start), .irq(irq), .unit_busy(unit_busy),
        .pp_mask(pp_mask), .pp_start(pp_start), .pp_irq(pp_irq), .pp_busy(pp_busy),
        .pp_qp_we(pp_qp_we), .pp_qp_waddr(pp_qp_waddr), .pp_qp_bias(pp_qp_bias),
        .pp_qp_mult(pp_qp_mult), .pp_qp_shift(pp_qp_shift), .perf_valid(perf_valid),
        .perf_cyc_total(p_total), .perf_cyc_mac(p_mac), .perf_cyc_stall(p_stall),
        .perf_cyc_mem_blocked(p_blk), .perf_mac_ops(p_ops));

    // start pulse counters (a rejected start must never reach the cluster)
    integer n_start = 0, n_pp_start = 0;
    always @(posedge clk) begin
        if (start)    n_start    = n_start + 1;
        if (pp_start) n_pp_start = n_pp_start + 1;
    end

    integer errors = 0;
    reg [31:0] rd;
    reg        last_err;

    task check(input [255:0] name, input [63:0] got, input [63:0] exp);
        if (got !== exp) begin $display("FAIL %0s: got %0h expected %0h", name, got, exp); errors = errors + 1; end
    endtask

    // bank-block address: word f of bank b
    function [11:0] BA(input integer b, input integer f);
        BA = 12'h100 * (b + 1) + f * 4;
    endfunction

    task bus_write(input [11:0] a, input [31:0] d);
        begin
            @(negedge clk); reg_req = 1; reg_we = 1; reg_addr = a; reg_wdata = d;
            @(negedge clk); reg_req = 0; reg_we = 0;
            if (!reg_rvalid) begin $display("FAIL no rvalid on write %0h", a); errors = errors + 1; end
            last_err = reg_err;
        end
    endtask

    task bus_read(input [11:0] a, output [31:0] d);
        begin
            @(negedge clk); reg_req = 1; reg_we = 0; reg_addr = a;
            @(negedge clk); reg_req = 0;
            if (!reg_rvalid) begin $display("FAIL no rvalid on read %0h", a); errors = errors + 1; end
            d = reg_rdata; last_err = reg_err;
        end
    endtask

    task wait_irq;
        integer t;
        begin
            t = 0;
            while (!irq_o && t < 1000) begin @(negedge clk); t = t + 1; end
            if (!irq_o) begin $display("FAIL timeout waiting for irq_o"); errors = errors + 1; end
            @(negedge clk);
        end
    endtask

    localparam G_CTRL = 12'h04, G_UMASK = 12'h08, G_PMASK = 12'h0C, G_STATUS = 12'h10, G_UBUSY = 12'h14,
               G_PBUSY = 12'h18, G_PVALID = 12'h1C, G_IRQST = 12'h20, G_IRQEN = 12'h24,
               G_QPMASK = 12'h40, G_QPADDR = 12'h44, G_QPBIAS = 12'h48, G_QPMULT = 12'h4C, G_QPSHIFT = 12'h50;
    // bank word indices
    localparam F_IFMAP_COLS = 0, F_IN_CH = 3, F_IFMAP_BASE = 12, F_RF_CH = 16, F_ACCUM = 17, F_RSVD = 18,
               F_PP_POOL_K = 26, F_PP_TOTAL = 29, F_PP_ZP = 30, F_PP_RELU = 31;

    initial begin
        $dumpfile("sim/waveform/nmc_regs_wave.vcd");
        $dumpvars(0, nmc_regs_tb);
        #22 rst = 0;

        // ---- identification ----
        bus_read(12'h000, rd); check("ID", rd, 32'h4E4D0001); check("ID err", last_err, 0);
        bus_read(12'h028, rd); check("CONFIG", rd, {16'd512, 8'd48, 8'd4});
        bus_write(12'h000, 32'h1); check("write RO ID -> err", last_err, 1);

        // ---- descriptor write / readback / masking / fan-out ----
        bus_write(BA(0, F_IFMAP_COLS), 32'hFFFF_1234);
        bus_read (BA(0, F_IFMAP_COLS), rd);   check("ifmap_cols mask", rd, 32'h1234);
        check("ifmap_cols port b0", ifmap_cols[0*DW +: DW], 16'h1234);
        bus_write(BA(2, F_IN_CH), 64);
        check("in_ch port b2", input_channels[2*DW +: DW], 64);
        check("in_ch port b0 untouched", input_channels[0*DW +: DW], 0);
        bus_write(BA(3, F_IFMAP_BASE), 32'hFFFF_FFFF);
        bus_read (BA(3, F_IFMAP_BASE), rd);   check("base mask 17b", rd, 32'h0001_FFFF);
        check("base port b3", ifmap_base_addr[3*AW +: AW], 17'h1FFFF);
        bus_write(BA(1, F_RF_CH), 32'h1FF);
        check("rf_ch port b1 (8b)", rf_ch_stride[1*RFAW +: RFAW], 8'hFF);
        bus_write(BA(2, F_ACCUM), 32'h3);
        check("accumulate", accumulate, 4'b0100);
        bus_write(BA(0, F_PP_ZP), 32'h1F0);
        check("zp port b0", pp_out_zp[7:0], 8'hF0);
        bus_write(BA(3, F_PP_RELU), 1);
        check("relu", pp_relu_en, 4'b1000);
        bus_write(BA(1, F_PP_TOTAL), 100);
        check("total_out b1", pp_total_out[1*AW +: AW], 100);
        bus_write(BA(0, F_RSVD), 5);   check("reserved word write -> err", last_err, 1);
        bus_read (BA(0, F_RSVD), rd);  check("reserved word read -> err",  last_err, 1);
        bus_write(12'h500, 5);         check("bank 4 (unmapped) write -> err", last_err, 1);
        bus_read (12'h500, rd);        check("bank 4 (unmapped) read -> err",  last_err, 1);
        bus_write(BA(0, 32), 5);       check("write perf RO -> err", last_err, 1);

        // ---- start rules ----
        bus_write(G_CTRL, 1);                                    // mask == 0
        bus_read (G_IRQST, rd);  check("start w/ empty mask rejected", rd, 4);
        bus_write(G_IRQST, 4);
        bus_read (G_IRQST, rd);  check("W1C clears", rd, 0);
        check("no start reached cluster", n_start, 0);

        // ---- conv round on banks 0 and 2 ----
        bus_write(G_UMASK, 32'hFFFF_FFF5);
        bus_read (G_UMASK, rd);  check("umask trimmed to NB bits", rd, 4'b0101);
        bus_write(G_IRQEN, 4'b0011);
        bus_write(G_CTRL, 1);
        bus_read (G_STATUS, rd); check("STATUS conv active", rd, 1);
        check("1 start pulse", n_start, 1);
        bus_read (G_UBUSY, rd);  check("UNIT_BUSY", rd, 4'b0101);
        bus_read (G_PBUSY, rd);  check("PP_BUSY", rd, 0);

        bus_write(G_CTRL, 1);                                    // second start while running
        bus_read (G_IRQST, rd);  check("2nd start flagged", rd & 4, 4);
        check("2nd start not forwarded", n_start, 1);
        bus_write(BA(0, F_IFMAP_COLS), 32'h7777);                // descriptor write while running
        bus_read (BA(0, F_IFMAP_COLS), rd); check("desc write blocked", rd, 32'h1234);
        bus_read (G_IRQST, rd);  check("blocked write flagged", rd & 8, 8);
        bus_write(G_UMASK, 4'b1111);
        bus_read (G_UMASK, rd);  check("umask write blocked", rd, 4'b0101);
        bus_write(G_PMASK, 4'b0010);                             // PP_MASK is NOT locked by a conv round
        bus_read (G_PMASK, rd);  check("pmask writable during conv", rd, 4'b0010);
        bus_write(G_CTRL, 2);                                    // pp start while conv in flight
        check("pp start rejected during conv", n_pp_start, 0);

        wait_irq;
        bus_read (G_IRQST, rd);  check("irq status", rd, 4'b1101);   // done + rejected + blocked
        bus_read (G_STATUS, rd); check("STATUS idle", rd, 0);
        bus_read (G_PVALID, rd); check("PERF_VALID", rd, 4'b0101);
        bus_write(G_IRQST, 32'hF);
        bus_read (G_IRQST, rd);  check("irq status cleared", rd, 0);
        check("irq_o low", irq_o, 0);

        // ---- perf counters (48-bit LO/HI) ----
        bus_read(BA(1, 32), rd); check("b1 total LO", rd, 32'h0000_0009);
        bus_read(BA(1, 33), rd); check("b1 total HI", rd, 32'h0000_0001);
        bus_read(BA(0, 32), rd); check("b0 total LO", rd, 7);
        bus_read(BA(2, 32), rd); check("b2 total LO", rd, 2007);
        bus_read(BA(2, 34), rd); check("b2 mac",      rd, 29);
        bus_read(BA(3, 36), rd); check("b3 stall",    rd, 3);
        bus_read(BA(3, 38), rd); check("b3 blocked",  rd, 6);
        bus_read(BA(2, 40), rd); check("b2 mac_ops",  rd, 288);
        bus_read(BA(2, 41), rd); check("b2 mac_ops HI", rd, 0);
        bus_read(BA(0, 42), rd); check("perf word 10 -> err", last_err, 1);

        // ---- requant parameters ----
        bus_write(G_QPMASK, 4'b0010);
        bus_write(G_QPADDR, 5);
        bus_write(G_QPBIAS, 32'hFFFF_FF9C);                      // -100
        bus_write(G_QPMULT, 300);
        bus_write(G_QPSHIFT, 7);                                 // commit -> ch 5
        bus_write(G_QPBIAS, 55);
        bus_write(G_QPMULT, 32'hFFFF);
        bus_write(G_QPSHIFT, 31);                                // commit -> ch 6 (auto-increment)
        repeat (3) @(negedge clk);
        check("qp bank1 ch5", stub.qpm[1*512 + 5], {32'hFFFF_FF9C, 16'd300, 5'd7});
        check("qp bank1 ch6", stub.qpm[1*512 + 6], {32'd55, 16'hFFFF, 5'd31});
        check("qp bank0 untouched", stub.qpm[0*512 + 5], 0);
        bus_read(G_QPADDR, rd);  check("qp addr auto-inc", rd, 7);

        bus_write(G_QPMASK, 4'b0101);                            // two banks at once, wrap at MAX_OC
        bus_write(G_QPADDR, 511);
        bus_write(G_QPBIAS, 1);  bus_write(G_QPMULT, 2);  bus_write(G_QPSHIFT, 3);
        repeat (3) @(negedge clk);
        check("qp bank0 ch511", stub.qpm[0*512 + 511], {32'd1, 16'd2, 5'd3});
        check("qp bank2 ch511", stub.qpm[2*512 + 511], {32'd1, 16'd2, 5'd3});
        check("qp bank1 ch511 untouched", stub.qpm[1*512 + 511], 0);
        bus_read(G_QPADDR, rd);  check("qp addr wraps", rd, 0);

        // ---- pp round on bank 1 ----
        bus_write(G_PMASK, 0);
        bus_write(G_CTRL, 2);
        bus_read (G_IRQST, rd);  check("pp start w/ empty mask rejected", rd, 4);
        bus_write(G_IRQST, 4);
        bus_write(G_PMASK, 4'b0010);
        bus_write(G_CTRL, 2);
        bus_read (G_STATUS, rd); check("STATUS pp active", rd, 2);
        check("1 pp start pulse", n_pp_start, 1);
        bus_read (G_PBUSY, rd);  check("PP_BUSY", rd, 4'b0010);
        bus_write(G_QPBIAS, 32'h99);                             // QP locked during a pp round
        bus_read (G_QPBIAS, rd); check("qp write blocked", rd, 1);
        bus_write(BA(1, F_PP_POOL_K), 3);
        check("pp desc blocked", pp_pool_k[1*DW +: DW], 0);
        bus_write(BA(1, F_IN_CH), 9);                            // conv desc is NOT locked by a pp round
        check("conv desc writable during pp", input_channels[1*DW +: DW], 9);
        bus_write(G_CTRL, 1);                                    // conv start during pp round
        check("conv start rejected during pp", n_start, 1);
        wait_irq;
        bus_read (G_IRQST, rd);  check("pp irq status", rd, 4'b1110);   // PP_DONE + rejected conv start + blocked writes
        bus_write(G_IRQST, 32'hF);

        // ---- both start bits in one write: conv wins, pp rejected ----
        bus_write(G_UMASK, 4'b0001);
        bus_write(G_CTRL, 3);
        @(negedge clk);                                          // start pulse is registered: lands 1 cycle after the write
        check("conv wins: conv started", n_start, 2);
        check("conv wins: pp not started", n_pp_start, 1);
        bus_read (G_IRQST, rd);  check("pp rejected flagged", rd & 4, 4);
        wait_irq;
        bus_write(G_IRQST, 32'hF);
        check("irq_o low at end", irq_o, 0);

        if (errors == 0) $display("ALL TESTS PASSED"); else $display("%0d ERRORS", errors);
        $finish;
    end
endmodule