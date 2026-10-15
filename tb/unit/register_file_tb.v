`timescale 1ns/1ps

module register_file_tb #(
    parameter DATA_WIDTH      = 32,
    parameter DEPTH           = 256,
    parameter NUM_READ_PORTS  = 16,
    parameter NUM_WRITE_PORTS = 16,
    parameter ADDR_WIDTH      = $clog2(DEPTH)
);

    reg clk, rst, flush;

    reg  [NUM_WRITE_PORTS-1:0]            write_en;
    reg  [ADDR_WIDTH*NUM_WRITE_PORTS-1:0] write_addr;
    reg  [DATA_WIDTH*NUM_WRITE_PORTS-1:0] write_data;
    wire [NUM_WRITE_PORTS-1:0]            write_blocked;

    reg  [NUM_READ_PORTS-1:0]             read_en, read_last;
    reg  [ADDR_WIDTH*NUM_READ_PORTS-1:0]  read_addr;
    wire [DATA_WIDTH*NUM_READ_PORTS-1:0]  read_data;
    wire [NUM_READ_PORTS-1:0]             read_valid;

    register_file #(
        .DATA_WIDTH(DATA_WIDTH), .DEPTH(DEPTH),
        .NUM_READ_PORTS(NUM_READ_PORTS), .NUM_WRITE_PORTS(NUM_WRITE_PORTS),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) dut (
        .clk(clk), .rst(rst), .flush(flush),
        .write_en(write_en), .write_addr(write_addr), .write_data(write_data),
        .write_blocked(write_blocked),
        .read_en(read_en), .read_last(read_last), .read_addr(read_addr),
        .read_data(read_data), .read_valid(read_valid)
    );

    always #5 clk = ~clk;

    integer errors = 0;
    integer i;

    // helper: DV of a word as {D,V}
    function [1:0] dv(input integer a);
        dv = {dut.d_bit[a], dut.v_bit[a]};
    endfunction

    task check(input [255:0] name, input integer got, input integer exp);
        if (got !== exp) begin
            $display("FAIL  %0s: got %0d expected %0d", name, got, exp);
            errors = errors + 1;
        end
    endtask

    task idle;
        begin
            write_en = 0; read_en = 0; read_last = 0; flush = 0;
        end
    endtask

    initial begin
        clk = 0; rst = 1; idle;
        write_addr = 0; write_data = 0; read_addr = 0;
        #12 rst = 0;

        // ---------- reset state: everything 00, reads invalid ----------
        @(negedge clk);
        for (i = 0; i < NUM_READ_PORTS; i = i + 1)
            read_addr[i*ADDR_WIDTH +: ADDR_WIDTH] = i;
        #1;
        check("reset: read_valid", read_valid, 0);
        check("reset: read_data",  read_data, 0);
        check("reset: DV[0]", dv(0), 2'b00);

        // ---------- write 16 words (addr 0..15, data 10..160) ----------
        @(negedge clk);
        write_en = {NUM_WRITE_PORTS{1'b1}};
        for (i = 0; i < NUM_WRITE_PORTS; i = i + 1) begin
            write_addr[i*ADDR_WIDTH +: ADDR_WIDTH] = i;
            write_data[i*DATA_WIDTH +: DATA_WIDTH] = (i + 1) * 10;
        end
        @(negedge clk);
        idle;
        check("after write: DV[3]", dv(3), 2'b01);
        check("after write: read_valid", read_valid, 16'hFFFF);
        check("after write: data port 2", read_data[2*DATA_WIDTH +: DATA_WIDTH], 30);

        // ---------- read, not last: 01 -> 11 ----------
        read_en = {NUM_READ_PORTS{1'b1}};
        @(negedge clk);
        read_en = 0;
        check("read: DV[3]", dv(3), 2'b11);
        check("read: still valid", read_valid, 16'hFFFF);

        // ---------- write to live word must be blocked ----------
        write_en = 16'h0001;
        write_addr[0 +: ADDR_WIDTH] = 0;
        write_data[0 +: DATA_WIDTH] = 999;
        #1; check("live write blocked flag", write_blocked[0], 1);
        @(negedge clk);
        idle;
        check("live data kept", read_data[0 +: DATA_WIDTH], 10);

        // ---------- last read on ports 0..7: 11 -> 10 ----------
        read_en   = 16'h00FF;
        read_last = 16'h00FF;
        @(negedge clk);
        read_en = 0; read_last = 0;
        check("last: DV[0]", dv(0), 2'b10);
        check("last: DV[8]", dv(8), 2'b11);
        check("last: read_valid", read_valid, 16'hFF00);
        check("last: masked data", read_data[0 +: DATA_WIDTH], 0);

        // ---------- replace consumed word: 10 -> 01 ----------
        write_en = 16'h0001;
        write_addr[0 +: ADDR_WIDTH] = 0;
        write_data[0 +: DATA_WIDTH] = 777;
        #1; check("replace allowed", write_blocked[0], 0);
        @(negedge clk);
        idle;
        check("replace: DV[0]", dv(0), 2'b01);
        check("replace: data", read_data[0 +: DATA_WIDTH], 777);

        // ---------- read of an empty word does not change state ----------
        read_addr[15*ADDR_WIDTH +: ADDR_WIDTH] = 200;
        read_en = 16'h8000; read_last = 16'h8000;
        @(negedge clk);
        read_en = 0; read_last = 0;
        check("empty read: DV[200]", dv(200), 2'b00);

        // ---------- two readers same word, one last -> consumed ----------
        read_addr[0*ADDR_WIDTH +: ADDR_WIDTH] = 12;
        read_addr[1*ADDR_WIDTH +: ADDR_WIDTH] = 12;
        read_en = 16'h0003; read_last = 16'h0002;
        @(negedge clk);
        read_en = 0; read_last = 0;
        check("shared read: DV[12]", dv(12), 2'b10);

        // ---------- flush ----------
        flush = 1;
        @(negedge clk);
        idle;
        check("flush: DV[9]", dv(9), 2'b00);
        check("flush: read_valid", read_valid[8], 0);

        if (errors == 0) $display("ALL TESTS PASSED");
        else             $display("%0d TEST(S) FAILED", errors);
        $finish;
    end

    initial begin
        $dumpfile("sim/waveform/register_file_wave.vcd");
        $dumpvars(0, register_file_tb);
    end

endmodule