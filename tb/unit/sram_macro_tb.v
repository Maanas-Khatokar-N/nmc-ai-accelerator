`timescale 1ns/1ps
// Self-checking testbench for sram_macro
module sram_macro_tb;

    parameter DATA_WIDTH = 16, DEPTH = 32, ADDR_WIDTH = $clog2(DEPTH);

    reg clk = 0, en = 0, we = 0;
    reg  [ADDR_WIDTH-1:0] addr  = 0;
    reg  [DATA_WIDTH-1:0] wdata = 0;
    wire [DATA_WIDTH-1:0] rdata;

    sram_macro #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(DEPTH)) dut (
        .clk(clk), .en(en), .we(we), .addr(addr), .wdata(wdata), .rdata(rdata));

    always #5 clk = ~clk;

    integer errors = 0, i;

    task check(input [255:0] name, input [DATA_WIDTH-1:0] got, input [DATA_WIDTH-1:0] exp);
        if (got !== exp) begin
            $display("FAIL %0s: got %0h expected %0h", name, got, exp);
            errors = errors + 1;
        end
    endtask

    initial begin
        $dumpfile("sim/waveform/sram_macro_wave.vcd");
        $dumpvars(0, sram_macro_tb);

        // write 32 words: mem[i] = 0x100 + i
        for (i = 0; i < DEPTH; i = i + 1) begin
            @(negedge clk); en = 1; we = 1; addr = i; wdata = 16'h100 + i;
        end
        @(negedge clk); en = 0; we = 0;

        // read: data appears one cycle after the request, not the same cycle
        @(negedge clk); en = 1; we = 0; addr = 1;
        @(negedge clk); en = 0;
        check("read addr 1", rdata, 16'h101);
        @(negedge clk); en = 1; we = 0; addr = 5;
        #1; check("no combinational read", rdata, 16'h101);   // still old value
        @(negedge clk); en = 0;
        check("read latency 1", rdata, 16'h105);

        // hold when idle
        addr = 9;
        @(negedge clk);
        check("hold when en=0", rdata, 16'h105);

        // read-before-write: write returns OLD data on rdata
        en = 1; we = 1; addr = 7; wdata = 16'hBEEF;
        @(negedge clk); en = 0; we = 0;
        check("read-before-write old data", rdata, 16'h107);
        en = 1; we = 0; addr = 7;
        @(negedge clk); en = 0;
        check("write took effect", rdata, 16'hBEEF);

        // we without en must not write
        en = 0; we = 1; addr = 8; wdata = 16'hDEAD;
        @(negedge clk); we = 0;
        en = 1; addr = 8;
        @(negedge clk); en = 0;
        check("we ignored when en=0", rdata, 16'h108);

        if (errors == 0) $display("ALL TESTS PASSED");
        else             $display("%0d ERRORS", errors);
        $finish;
    end
endmodule