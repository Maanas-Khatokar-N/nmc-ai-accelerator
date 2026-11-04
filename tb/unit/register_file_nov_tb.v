`timescale 1ns/1ps
// WORD_VALID=0 vs the original model, under the protocol the unit guarantees:
// writes only go to words that are free in the reference (never overwrite live data), reads only
// matter where the reference says the word is valid. Checks data equality there, constants elsewhere.
module register_file_nov_tb;
    parameter DW = 8, DEPTH = 64, NR = 16, NW = 8, AW = $clog2(DEPTH), CYCLES = 20000;
    reg clk = 0, rst = 1, flush = 0;  always #5 clk = ~clk;
    reg [NW-1:0] wen = 0;  reg [AW*NW-1:0] waddr = 0;  reg [DW*NW-1:0] wdata = 0;
    reg [NR-1:0] ren = 0;  reg [AW*NR-1:0] raddr = 0;
    wire [NR-1:0] z = {NR{1'b0}};
    wire [NW-1:0] blk_r, cf_r, blk_d, cf_d;  wire [DW*NR-1:0] rd_r, rd_d;  wire [NR-1:0] rv_r, rv_d;
    reg [NW-1:0] wen_req;

    register_file_ref #(.DATA_WIDTH(DW), .DEPTH(DEPTH), .NUM_READ_PORTS(NR), .NUM_WRITE_PORTS(NW)) r_i (
        .clk(clk), .rst(rst), .flush(flush), .write_en(wen), .write_addr(waddr), .write_data(wdata), .write_blocked(blk_r),
        .chk_addr(waddr), .chk_free(cf_r), .read_en(ren), .read_last(z), .read_addr(raddr), .read_data(rd_r), .read_valid(rv_r));
    register_file #(.DATA_WIDTH(DW), .DEPTH(DEPTH), .NUM_READ_PORTS(NR), .NUM_WRITE_PORTS(NW),
                    .TRACK_READS(0), .WORD_VALID(0)) d_i (
        .clk(clk), .rst(rst), .flush(flush), .write_en(wen), .write_addr(waddr), .write_data(wdata), .write_blocked(blk_d),
        .chk_addr(waddr), .chk_free(cf_d), .read_en(ren), .read_last(z), .read_addr(raddr), .read_data(rd_d), .read_valid(rv_d));

    integer cyc, p, base, errors = 0, nchecked = 0;
    initial begin
        #22 rst = 0;
        for (cyc = 0; cyc < CYCLES; cyc = cyc + 1) begin
            @(negedge clk);
            flush = ($random & 15) == 0;
            base = $random & (DEPTH - 1);
            for (p = 0; p < NW; p = p + 1) begin
                waddr[p*AW +: AW] = (base + p) & (DEPTH - 1);
                wdata[p*DW +: DW] = $random;
            end
            for (p = 0; p < NR; p = p + 1) begin ren[p] = $random; raddr[p*AW +: AW] = $random; end
            wen = 0; #1;
            for (p = 0; p < NW; p = p + 1) wen[p] = ($random & 1) && cf_r[p];   // free words only
            #1;
            if (blk_r !== 0) begin errors = errors + 1; $display("ref blocked (tb bug)"); end
            if (rv_d !== {NR{1'b1}} || blk_d !== 0 || cf_d !== {NW{1'b1}}) begin errors = errors + 1; $display("constants wrong cyc %0d", cyc); end
            for (p = 0; p < NR; p = p + 1)
                if (rv_r[p]) begin
                    nchecked = nchecked + 1;
                    if (rd_d[p*DW +: DW] !== rd_r[p*DW +: DW]) begin errors = errors + 1; if (errors < 10) $display("data mismatch cyc %0d port %0d", cyc, p); end
                end
        end
        $display("checked %0d valid reads", nchecked);
        if (errors == 0) $display("ALL TESTS PASSED"); else $display("%0d ERRORS", errors);
        $finish;
    end
endmodule
