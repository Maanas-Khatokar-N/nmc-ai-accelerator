`timescale 1ns/1ps
// Self-checking unit test for access_controller (pure handshake logic).
module access_controller_tb;
    localparam N = 16;
    reg clk = 0, rst = 1;  always #5 clk = ~clk;

    reg compute_start = 0, seq_valid = 0, seq_last = 0;
    reg [N-1:0] pe_mask = {N{1'b1}}, ifv = {N{1'b1}}, psv = {N{1'b1}};
    reg wv = 1, psum_req = 0;
    wire seq_enable, compute_done, wrf_en, psum_ack, pe_clear, mac_en, psum_load, stall;
    wire [N-1:0] if_en, ps_en, ps_last;

    access_controller #(.NUM_PE(N)) dut (
        .clk(clk), .rst(rst), .compute_start(compute_start), .seq_valid(seq_valid), .seq_last(seq_last),
        .pe_mask(pe_mask), .seq_enable(seq_enable), .compute_done(compute_done),
        .ifrf_read_valid(ifv), .wrf_read_valid(wv), .ifrf_read_en(if_en), .wrf_read_en(wrf_en),
        .psum_req(psum_req), .psrf_read_valid(psv), .psrf_read_en(ps_en), .psrf_read_last(ps_last),
        .psum_ack(psum_ack), .pe_clear(pe_clear), .pe_mac_en(mac_en), .pe_psum_load(psum_load), .stall(stall));

    integer errors = 0;
    task check(input [255:0] name, input integer got, input integer exp);
        if (got !== exp) begin $display("FAIL %0s: got %0h expected %0h", name, got, exp); errors = errors + 1; end
    endtask

    initial begin
        $dumpfile("sim/waveform/access_controller_wave.vcd");
        $dumpvars(0, access_controller_tb);
        #12 rst = 0; @(negedge clk);

        // 1. everything valid -> MAC fires, all lanes consumed
        seq_valid = 1; #1;
        check("fire: mac_en", mac_en, 1);  check("fire: seq_enable", seq_enable, 1);
        check("fire: ifrf_read_en", if_en, 16'hFFFF); check("fire: wrf_read_en", wrf_en, 1);
        check("fire: stall", stall, 0);

        // 2. one active IFMap lane invalid -> stall, nothing consumed
        ifv = 16'hFFFF & ~(1 << 5); #1;
        check("stall lane: mac_en", mac_en, 0); check("stall lane: seq_enable", seq_enable, 0);
        check("stall lane: ifrf_read_en", if_en, 0); check("stall lane: wrf_read_en", wrf_en, 0);
        check("stall lane: stall", stall, 1);

        // 3. same lane but masked -> fires, masked lane not read
        pe_mask = 16'hFFFF & ~(1 << 5); #1;
        check("masked: mac_en", mac_en, 1);
        check("masked: ifrf_read_en", if_en, 16'hFFDF);

        // 4. weight invalid -> stall
        ifv = {N{1'b1}}; pe_mask = {N{1'b1}}; wv = 0; #1;
        check("weight stall: mac_en", mac_en, 0); check("weight stall: stall", stall, 1);
        wv = 1;

        // 5. no op pending
        seq_valid = 0; #1;
        check("idle: mac_en", mac_en, 0); check("idle: stall", stall, 0); check("idle: read_en", if_en, 0);

        // 6. compute_done: only after the LAST op actually fires, one cycle later
        seq_valid = 1; seq_last = 0; @(negedge clk);
        check("done not after non-last", compute_done, 0);
        seq_last = 1; ifv = 0; @(negedge clk);              // last op but stalled
        check("done not while stalled", compute_done, 0);
        ifv = {N{1'b1}}; #1; check("last fires", mac_en, 1);
        @(negedge clk);
        check("done 1 cycle after last MAC", compute_done, 1);
        seq_valid = 0; seq_last = 0; @(negedge clk);
        check("done is a 1-cycle pulse", compute_done, 0);

        // 7. pe_clear is compute_start delayed 1 cycle
        compute_start = 1; @(negedge clk); compute_start = 0;
        check("clear delayed", pe_clear, 1); @(negedge clk);
        check("clear 1-cycle", pe_clear, 0);

        // 8. psum reload: valid -> load, active lanes consumed with read_last, MAC blocked
        seq_valid = 1; psum_req = 1; #1;
        check("psum: load", psum_load, 1); check("psum: ack", psum_ack, 1);
        check("psum: read_en", ps_en, 16'hFFFF); check("psum: read_last", ps_last, 16'hFFFF);
        check("psum: mac blocked", mac_en, 0);
        // 9. psum lane invalid & active -> wait, nothing consumed
        psv = 16'hFFFF & ~(1 << 3); #1;
        check("psum wait: load", psum_load, 0); check("psum wait: read_en", ps_en, 0);
        // 10. that lane masked -> load proceeds, masked lane untouched
        pe_mask = 16'hFFFF & ~(1 << 3); #1;
        check("psum masked: load", psum_load, 1); check("psum masked: read_en", ps_en, 16'hFFF7);
        psum_req = 0; pe_mask = {N{1'b1}}; psv = {N{1'b1}}; #1;
        check("psum off: mac resumes", mac_en, 1); check("psum off: no load", psum_load, 0);

        if (errors == 0) $display("ALL TESTS PASSED"); else $display("%0d ERRORS", errors);
        $finish;
    end
endmodule