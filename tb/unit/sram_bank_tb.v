`timescale 1ns/1ps
// Self-checking testbench for sram_bank.
// Tests: interleaving, 8-macro-per-cycle NMC read (Eq. 1), core-vs-NMC
// conflict on one macro, simultaneous NMC writes, read latency / rvalid.
module sram_bank_tb;

    parameter DW = 32, M = 8, DEPTH = 64;
    localparam MAW = $clog2(DEPTH), SW = $clog2(M), CAW = SW + MAW;

    reg clk = 0, rst = 1;
    always #5 clk = ~clk;

    reg                 core_req = 0, core_we = 0;
    reg  [CAW-1:0]      core_addr = 0;
    reg  [DW-1:0]       core_wdata = 0;
    wire [DW-1:0]       core_rdata;
    wire                core_rvalid;

    reg  [M-1:0]        nmc_req = 0, nmc_we = 0;
    reg  [M*MAW-1:0]    nmc_addr = 0;
    reg  [M*DW-1:0]     nmc_wdata = 0;
    wire [M-1:0]        nmc_grant, nmc_blocked, nmc_rvalid;
    wire [M*DW-1:0]     nmc_rdata;

    sram_bank #(.DATA_WIDTH(DW), .NUM_MACROS(M), .DEPTH_PER_MACRO(DEPTH)) dut (
        .clk(clk), .rst(rst),
        .core_req(core_req), .core_we(core_we), .core_addr(core_addr),
        .core_wdata(core_wdata), .core_rdata(core_rdata), .core_rvalid(core_rvalid),
        .nmc_req(nmc_req), .nmc_we(nmc_we), .nmc_addr(nmc_addr), .nmc_wdata(nmc_wdata),
        .nmc_grant(nmc_grant), .nmc_blocked(nmc_blocked),
        .nmc_rdata(nmc_rdata), .nmc_rvalid(nmc_rvalid));

    integer errors = 0, i;

    task check(input [255:0] name, input [DW-1:0] got, input [DW-1:0] exp);
        if (got !== exp) begin
            $display("FAIL %0s: got %0h expected %0h", name, got, exp);
            errors = errors + 1;
        end
    endtask

    task core_write(input [CAW-1:0] a, input [DW-1:0] d);
        begin
            @(negedge clk); core_req = 1; core_we = 1; core_addr = a; core_wdata = d;
            @(negedge clk); core_req = 0; core_we = 0;
        end
    endtask

    // read a word through the core port; data valid one cycle later
    task core_read(input [CAW-1:0] a, output [DW-1:0] d);
        begin
            @(negedge clk); core_req = 1; core_we = 0; core_addr = a;
            @(negedge clk); core_req = 0;
            if (!core_rvalid) begin
                $display("FAIL core_rvalid not set"); errors = errors + 1;
            end
            d = core_rdata;
        end
    endtask

    // set every NMC lane to the same macro-local address
    task set_nmc_addr_all(input [MAW-1:0] a);
        integer k;
        begin
            for (k = 0; k < M; k = k + 1) nmc_addr[k*MAW +: MAW] = a;
        end
    endtask

    reg [DW-1:0] rd;

    initial begin
        $dumpfile("sim/waveform/sram_bank_wave.vcd");
        $dumpvars(0, sram_bank_tb);

        #12 rst = 0;

        // ---- Test 1: core writes land interleaved: word w -> macro w%8, local w/8
        for (i = 0; i < M; i = i + 1) core_write(i,     32'hA0 + i);   // local 0
        for (i = 0; i < M; i = i + 1) core_write(M + i, 32'hB0 + i);   // local 1

        core_read(11, rd); check("core read word 11 (B3)", rd, 32'hB3);
        core_read(5,  rd); check("core read word 5  (A5)", rd, 32'hA5);

        // ---- Test 2: NMC reads ALL 8 macros in ONE cycle (Eq. 1) ----
        @(negedge clk);
        nmc_req = {M{1'b1}}; nmc_we = 0; set_nmc_addr_all(0);
        #1;
        check("nmc grant all 8", nmc_grant,   {M{1'b1}});
        check("nmc blocked none", nmc_blocked, 0);
        check("no rvalid same cycle", nmc_rvalid, 0);
        @(negedge clk); nmc_req = 0;
        check("nmc rvalid all 8", nmc_rvalid, {M{1'b1}});
        for (i = 0; i < M; i = i + 1)
            check("nmc lane data local0", nmc_rdata[i*DW +: DW], 32'hA0 + i);
        @(negedge clk);
        check("rvalid drops when idle", nmc_rvalid, 0);

        // ---- Test 3: NMC reads different local addr per macro in one cycle ----
        @(negedge clk);
        nmc_req = {M{1'b1}}; nmc_we = 0;
        for (i = 0; i < M; i = i + 1) nmc_addr[i*MAW +: MAW] = i[0] ? 1 : 0;  // odd->B, even->A
        @(negedge clk); nmc_req = 0;
        for (i = 0; i < M; i = i + 1)
            check("nmc per-macro addr", nmc_rdata[i*DW +: DW], (i[0] ? 32'hB0 : 32'hA0) + i);

        // ---- Test 4: conflict. core reads macro 3 while NMC reads all 8 ----
        @(negedge clk);
        core_req = 1; core_we = 0; core_addr = 11;          // macro 3, local 1 -> B3
        nmc_req = {M{1'b1}}; nmc_we = 0; set_nmc_addr_all(0);
        #1;
        check("conflict grant",   nmc_grant,   8'hF7);
        check("conflict blocked", nmc_blocked, 8'h08);
        @(negedge clk); core_req = 0; nmc_req = 0;
        check("core data wins",      core_rdata, 32'hB3);
        check("core rvalid",         core_rvalid, 1);
        check("nmc rvalid 7 lanes",  nmc_rvalid, 8'hF7);
        for (i = 0; i < M; i = i + 1)
            if (i != 3) check("nmc data during conflict", nmc_rdata[i*DW +: DW], 32'hA0 + i);

        // ---- Test 5: core hits a DIFFERENT macro than the NMC request -> no conflict
        @(negedge clk);
        core_req = 1; core_we = 0; core_addr = 8*0 + 6;     // macro 6
        nmc_req = 8'h0F; nmc_we = 0; set_nmc_addr_all(1);   // macros 0..3
        #1;
        check("no-conflict grant",   nmc_grant,   8'h0F);
        check("no-conflict blocked", nmc_blocked, 0);
        @(negedge clk); core_req = 0; nmc_req = 0;
        check("core data macro 6", core_rdata, 32'hA6);

        // ---- Test 6: NMC writes 8 macros in one cycle, core write to macro 3 wins ----
        @(negedge clk);
        nmc_req = {M{1'b1}}; nmc_we = {M{1'b1}}; set_nmc_addr_all(10);
        for (i = 0; i < M; i = i + 1) nmc_wdata[i*DW +: DW] = 32'hD0 + i;
        core_req = 1; core_we = 1; core_addr = 10*M + 3; core_wdata = 32'hEE;
        #1;
        check("write conflict grant", nmc_grant, 8'hF7);
        @(negedge clk); core_req = 0; core_we = 0; nmc_req = 0; nmc_we = 0;
        check("writes give no nmc_rvalid", nmc_rvalid, 0);
        check("writes give no core_rvalid", core_rvalid, 0);

        @(negedge clk);
        nmc_req = {M{1'b1}}; nmc_we = 0; set_nmc_addr_all(10);
        @(negedge clk); nmc_req = 0;
        for (i = 0; i < M; i = i + 1)
            check("post-write data", nmc_rdata[i*DW +: DW], (i == 3) ? 32'hEE : 32'hD0 + i);

        // ---- Test 7: core and NMC write concurrently to different macros ----
        @(negedge clk);
        core_req = 1; core_we = 1; core_addr = 20*M + 0; core_wdata = 32'h1111;   // macro 0
        nmc_req = 8'hFE; nmc_we = 8'hFE; set_nmc_addr_all(20);                    // macros 1..7
        for (i = 0; i < M; i = i + 1) nmc_wdata[i*DW +: DW] = 32'h2000 + i;
        #1; check("concurrent write grants", nmc_grant, 8'hFE);
        @(negedge clk); core_req = 0; core_we = 0; nmc_req = 0; nmc_we = 0;
        core_read(20*M + 0, rd); check("concurrent: core word",  rd, 32'h1111);
        core_read(20*M + 7, rd); check("concurrent: nmc word 7", rd, 32'h2007);

        if (errors == 0) $display("ALL TESTS PASSED");
        else             $display("%0d ERRORS", errors);
        $finish;
    end
endmodule