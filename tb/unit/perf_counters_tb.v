`timescale 1ns/1ps
// Self-checking unit test for perf_counters
module perf_counters_tb;
    localparam N = 16, W = 48;
    reg clk = 0, rst = 1;  always #5 clk = ~clk;

    reg start = 0, done = 0, mac_en = 0, stall = 0, mem_blocked = 0;
    reg ph_load = 0, ph_compute = 0, ph_store = 0;
    reg [N-1:0] pe_mask = {N{1'b1}};
    wire running, perf_valid;
    wire [W-1:0] c_total, c_load, c_comp, c_store, c_mac, c_stall, c_blk, ops;

    perf_counters #(.NUM_PE(N), .CNT_W(W)) dut (
        .clk(clk), .rst(rst), .start(start), .done(done), .mac_en(mac_en), .pe_mask(pe_mask),
        .stall(stall), .mem_blocked(mem_blocked), .ph_load(ph_load), .ph_compute(ph_compute),
        .ph_store(ph_store), .running(running), .perf_valid(perf_valid),
        .cyc_total(c_total), .cyc_load(c_load), .cyc_compute(c_comp), .cyc_store(c_store),
        .cyc_mac(c_mac), .cyc_stall(c_stall), .cyc_mem_blocked(c_blk), .mac_ops(ops));

    integer errors = 0, i;
    task check(input [255:0] name, input integer got, input integer exp);
        if (got !== exp) begin $display("FAIL %0s: got %0d expected %0d", name, got, exp); errors = errors + 1; end
    endtask
    task tick; begin @(negedge clk); end endtask

    initial begin
        $dumpfile("sim/waveform/perf_counters_wave.vcd");
        $dumpvars(0, perf_counters_tb);
        #12 rst = 0; tick;

        // activity before start must not be counted
        mac_en = 1; tick; mac_en = 0;
        check("idle: total", c_total, 0); check("idle: running", running, 0);

        // ---- layer 1 ----
        start = 1; tick; start = 0;                       // accepted at this edge
        check("running after start", running, 1);
        ph_load = 1;                 for (i = 0; i < 4; i = i + 1) tick;   // 4 load cycles
        ph_load = 0; ph_compute = 1;
        mac_en = 1;                  for (i = 0; i < 9; i = i + 1) tick;   // 9 full MACs
        pe_mask = 16'h000F;          for (i = 0; i < 2; i = i + 1) tick;   // 2 MACs, 4 PEs
        mac_en = 0; stall = 1;       tick; tick;                           // 2 stalls
        stall = 0; ph_compute = 0; ph_store = 1; mem_blocked = 1; tick;    // 1 blocked
        mem_blocked = 0; tick; tick;                                       // 3 store total
        ph_store = 0; done = 1; tick; done = 0;           // done cycle is counted
        // total = 4 + 9 + 2 + 2 + 3 + 1(done cycle) = 21
        check("L1 total",   c_total, 21);
        check("L1 load",    c_load, 4);
        check("L1 compute", c_comp, 13);
        check("L1 store",   c_store, 3);
        check("L1 mac",     c_mac, 11);
        check("L1 stall",   c_stall, 2);
        check("L1 blocked", c_blk, 1);
        check("L1 mac_ops", ops, 9*16 + 2*4);
        check("L1 valid",   perf_valid, 1);
        check("L1 running", running, 0);

        // counters hold after done
        mac_en = 1; tick; tick; mac_en = 0;
        check("hold: total", c_total, 21); check("hold: ops", ops, 152);

        // ---- layer 2: counters restart ----
        pe_mask = {N{1'b1}};
        start = 1; tick; start = 0;
        check("L2 cleared total", c_total, 0); check("L2 cleared ops", ops, 0);
        check("L2 valid dropped", perf_valid, 0);
        start = 1; mac_en = 1; tick; start = 0;           // start while running is ignored
        mac_en = 0; done = 1; tick; done = 0;
        check("L2 total", c_total, 2); check("L2 mac_ops", ops, 16);

        if (errors == 0) $display("ALL TESTS PASSED"); else $display("%0d ERRORS", errors);
        $finish;
    end
endmodule