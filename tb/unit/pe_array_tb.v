`timescale 1ns/1ps
// Self-checking testbench for the PE array.
// Tests: broadcast MAC, masking, psum load, clear, per-PE weight mode.
module pe_array_tb;

    parameter NUM_PE = 16, DATA_WIDTH = 8, ACC_WIDTH = 32;

    reg clk = 0, rst = 1, clear = 0, psum_load = 0, mac_en = 0;
    reg [NUM_PE-1:0] pe_mask = {NUM_PE{1'b1}};
    reg  signed [NUM_PE*ACC_WIDTH-1:0]  psum_in  = 0;
    reg  signed [NUM_PE*DATA_WIDTH-1:0] ifmap_in = 0;
    reg  signed [DATA_WIDTH-1:0]        weight_b = 0;           // broadcast weight
    reg  signed [NUM_PE*DATA_WIDTH-1:0] weight_u = 0;           // per-PE weights
    wire signed [NUM_PE*ACC_WIDTH-1:0]  acc_b, acc_u;

    // DUT 1: broadcast weight
    pe_array #(.WEIGHT_BROADCAST(1)) dut_b (
        .clk(clk), .rst(rst), .clear(clear), .pe_mask(pe_mask),
        .psum_load(psum_load), .mac_en(mac_en), .psum_in(psum_in),
        .ifmap_in(ifmap_in), .weight_in(weight_b), .acc_out(acc_b));

    // DUT 2: per-PE weights
    pe_array #(.WEIGHT_BROADCAST(0)) dut_u (
        .clk(clk), .rst(rst), .clear(clear), .pe_mask(pe_mask),
        .psum_load(psum_load), .mac_en(mac_en), .psum_in(psum_in),
        .ifmap_in(ifmap_in), .weight_in(weight_u), .acc_out(acc_u));

    always #5 clk = ~clk;

    integer i, errors = 0;
    reg signed [ACC_WIDTH-1:0] got;

    task check_b(input integer pe, input signed [ACC_WIDTH-1:0] exp, input [255:0] name);
        begin
            got = acc_b[pe*ACC_WIDTH +: ACC_WIDTH];
            if (got !== exp) begin
                $display("FAIL %0s: PE%0d got %0d expected %0d", name, pe, got, exp);
                errors = errors + 1;
            end
        end
    endtask

    // Drive inputs just after a clock edge, then wait for next edge
    task step; begin @(posedge clk); #1; end endtask

    initial begin
        $dumpfile("sim/waveform/pe_array_wave.vcd");
        $dumpvars(0, pe_array_tb);

        #12 rst = 0; step;

        // ---- Test 1: broadcast weight. PE i sees pixel (i+1), weight = 3 ----
        for (i = 0; i < NUM_PE; i = i + 1) ifmap_in[i*DATA_WIDTH +: DATA_WIDTH] = i + 1;
        weight_b = 3; mac_en = 1; step;          // acc = 3*(i+1)
        weight_b = -2; step;                     // acc += -2*(i+1)  -> 1*(i+1)
        mac_en = 0; step;
        for (i = 0; i < NUM_PE; i = i + 1) check_b(i, i + 1, "broadcast");

        // ---- Test 2: mask lower 8 PEs; they must hold, upper 8 accumulate ----
        pe_mask = 16'hFF00;
        weight_b = 1; mac_en = 1; step; mac_en = 0; step;
        for (i = 0; i < 8; i = i + 1)  check_b(i, i + 1,           "mask-hold");
        for (i = 8; i < 16; i = i + 1) check_b(i, 2*(i + 1),       "mask-active");

        // ---- Test 3: psum load (masked PEs must ignore it) ----
        for (i = 0; i < NUM_PE; i = i + 1) psum_in[i*ACC_WIDTH +: ACC_WIDTH] = 1000 + i;
        psum_load = 1; step; psum_load = 0; step;
        for (i = 0; i < 8; i = i + 1)  check_b(i, i + 1,    "load-masked");
        for (i = 8; i < 16; i = i + 1) check_b(i, 1000 + i, "load-active");

        // ---- Test 4: load then accumulate continues from loaded psum ----
        pe_mask = {NUM_PE{1'b1}};
        psum_load = 1; step; psum_load = 0;
        weight_b = 2; mac_en = 1; step; mac_en = 0; step;
        for (i = 0; i < NUM_PE; i = i + 1) check_b(i, 1000 + i + 2*(i + 1), "load+mac");

        // ---- Test 5: clear works even on masked PEs ----
        pe_mask = 16'h0000; clear = 1; step; clear = 0; step;
        for (i = 0; i < NUM_PE; i = i + 1) check_b(i, 0, "clear");

        // ---- Test 6: per-PE weight mode (weights = i, pixel = i+1) ----
        pe_mask = {NUM_PE{1'b1}};
        for (i = 0; i < NUM_PE; i = i + 1) weight_u[i*DATA_WIDTH +: DATA_WIDTH] = i;
        clear = 1; step; clear = 0;
        mac_en = 1; step; mac_en = 0; step;
        for (i = 0; i < NUM_PE; i = i + 1) begin
            got = acc_u[i*ACC_WIDTH +: ACC_WIDTH];
            if (got !== i*(i+1)) begin
                $display("FAIL unicast: PE%0d got %0d expected %0d", i, got, i*(i+1));
                errors = errors + 1;
            end
        end

        if (errors == 0) $display("ALL TESTS PASSED");
        else             $display("%0d ERRORS", errors);
        $finish;
    end
endmodule