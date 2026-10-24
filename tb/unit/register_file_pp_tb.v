`timescale 1ns/1ps
module register_file_pp_tb;
    reg clk = 0, rst = 1, flush = 0, flush_sel = 0, wr_sel = 0, rd_sel = 0;
    reg [1:0]  wen = 0, ren = 0, rlast = 0;
    reg [3:0]  wa = 0, ra = 0;
    reg [15:0] wd = 0;
    wire [1:0] blk, rv;  wire [15:0] rd;
    always #5 clk = ~clk;

    register_file_pp #(.DATA_WIDTH(8), .DEPTH(4), .NUM_READ_PORTS(2), .NUM_WRITE_PORTS(2)) dut (
        .clk(clk), .rst(rst), .flush(flush), .flush_sel(flush_sel), .wr_sel(wr_sel), .rd_sel(rd_sel),
        .write_en(wen), .write_addr(wa), .write_data(wd), .write_blocked(blk),
        .read_en(ren), .read_last(rlast), .read_addr(ra), .read_data(rd), .read_valid(rv));

    integer errors = 0;
    task check(input [255:0] name, input integer got, input integer exp);
        if (got !== exp) begin $display("FAIL %0s: got %0h exp %0h", name, got, exp); errors = errors + 1; end
    endtask

    initial begin
        $dumpfile("sim/waveform/register_file_pp_wave.vcd");
        $dumpvars(0, register_file_pp_tb);
        #12 rst = 0; @(negedge clk);
        wen = 2'b01; wa[1:0] = 0;
        wr_sel = 0; wd[7:0] = 8'h11; @(negedge clk);       // half A <- 11
        wr_sel = 1; wd[7:0] = 8'h22; @(negedge clk);       // half B <- 22
        wen = 0; ra[1:0] = 0;
        rd_sel = 0; #1; check("A data", rd[7:0], 8'h11); check("A valid", rv[0], 1);
        rd_sel = 1; #1; check("B data", rd[7:0], 8'h22); check("B valid", rv[0], 1);

        // consuming A must not touch B
        rd_sel = 0; ren = 2'b01; rlast = 2'b01; @(negedge clk); ren = 0; rlast = 0; #1;
        check("A consumed", rv[0], 0);
        rd_sel = 1; #1; check("B untouched", rv[0], 1);

        // flush B only
        flush = 1; flush_sel = 1; @(negedge clk); flush = 0; #1;
        check("B flushed", rv[0], 0);

        // refill A while B is being read-selected: write goes to A only
        wr_sel = 0; wen = 2'b01; wd[7:0] = 8'h33; @(negedge clk); wen = 0;
        rd_sel = 1; #1; check("B still empty", rv[0], 0);
        rd_sel = 0; #1; check("A refilled", rd[7:0], 8'h33);

        if (errors == 0) $display("ALL TESTS PASSED"); else $display("%0d ERRORS", errors);
        $finish;
    end
endmodule