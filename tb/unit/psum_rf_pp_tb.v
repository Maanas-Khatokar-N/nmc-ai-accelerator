`timescale 1ns/1ps
// psum_rf_pp vs the register_file_pp configuration it replaces (32b, 16 words, 16 read, 8 write ports).
// Writes follow the reload generator contract: one PE row per cycle, lane = row*4 + (m - w0lo) mod 8,
// only words that are free in the reference. Compared every cycle: read_valid, chk_free always,
// read_data where the reference word is valid (the old file zero-masks invalid words, the new one does not).
//   iverilog -g2012 -o sim/build/psum_rf.vvp tb/unit/psum_rf_pp_tb.v rtl/regfile/psum_rf_pp.v rtl/regfile/register_file_pp.v rtl/regfile/register_file.v && vvp sim/build/psum_rf.vvp
module psum_rf_pp_tb;
    localparam NM = 8, NP = 16, DW = 32, LW = 4;
    reg clk = 0, rst = 1;  always #5 clk = ~clk;
    reg flush = 0, flush_sel = 0, wr_sel = 0, rd_sel = 0;
    reg [NM-1:0] wen = 0;  reg [NM*LW-1:0] waddr = 0, caddr = 0;  reg [NM*DW-1:0] wdata = 0;
    reg [NP-1:0] ren = 0, rlast = 0;
    wire [NP*LW-1:0] raddr;
    genvar g;  generate for (g = 0; g < NP; g = g + 1) begin : a assign raddr[g*LW +: LW] = g; end endgenerate

    wire [NM-1:0] blk_r, cf_r, cf_n;  wire [NP*DW-1:0] rd_r, rd_n;  wire [NP-1:0] rv_r, rv_n;
    reg  [NM-1:0] wen_want;

    register_file_pp #(.DATA_WIDTH(DW), .DEPTH(NP), .NUM_READ_PORTS(NP), .NUM_WRITE_PORTS(NM),
                       .BANKED_STORAGE(0), .TRACK_READS(1), .WORD_VALID(1)) ref_i (
        .clk(clk), .rst(rst), .flush(flush), .flush_sel(flush_sel), .wr_sel(wr_sel), .rd_sel(rd_sel),
        .write_en(wen), .write_addr(waddr), .write_data(wdata), .write_blocked(blk_r),
        .chk_addr(caddr), .chk_free(cf_r), .read_en(ren), .read_last(rlast), .read_addr(raddr),
        .read_data(rd_r), .read_valid(rv_r));

    psum_rf_pp dut (
        .clk(clk), .rst(rst), .flush(flush), .flush_sel(flush_sel), .wr_sel(wr_sel), .rd_sel(rd_sel),
        .write_en(wen), .write_addr(waddr), .write_data(wdata),
        .chk_addr(caddr), .chk_free(cf_n), .read_en(ren), .read_last(rlast),
        .read_data(rd_n), .read_valid(rv_n));

    integer cyc, mm, c, row, w0, vc, p, errors = 0, nwr = 0, nvalid = 0;
    initial begin
        // kick the always @(*) blocks once before the first clock edge (Verilog does not run them at t=0)
        #1 wen = 8'h01;  #1 wen = 8'h00;
        #20 rst = 0;
        for (cyc = 0; cyc < 40000; cyc = cyc + 1) begin
            @(negedge clk);
            flush = ($random & 15) == 0;  flush_sel = $random;
            if (($random & 7) == 0) wr_sel = $random;
            if (($random & 7) == 0) rd_sel = $random;
            row = $random & 3;  w0 = $random & 7;  vc = 1 + ($random & 3);
            for (mm = 0; mm < NM; mm = mm + 1) begin
                c = (mm - w0) & 7;
                waddr[mm*LW +: LW] = row*4 + (c & 3);
                caddr[mm*LW +: LW] = row*4 + (c & 3);
                wdata[mm*DW +: DW] = $random;
                wen_want[mm] = (c < vc);
            end
            for (p = 0; p < NP; p = p + 1) begin ren[p] = $random; rlast[p] = (($random & 3) == 0); end
            wen = 0; #1;
            wen = wen_want & {NM{($random & 3) != 0}} & cf_r;          // free words only (reload retries)
            #1;
            if (cf_n !== cf_r) begin errors = errors + 1; if (errors < 10) $display("chk_free mismatch cyc %0d ref %b new %b wr_sel %b caddr %h flush %b", cyc, cf_r, cf_n, wr_sel, caddr, flush); end
            if (rv_n !== rv_r) begin errors = errors + 1; if (errors < 10) $display("read_valid mismatch cyc %0d", cyc); end
            for (p = 0; p < NP; p = p + 1)
                if (rv_r[p]) begin
                    nvalid = nvalid + 1;
                    if (rd_n[p*DW +: DW] !== rd_r[p*DW +: DW]) begin errors = errors + 1; if (errors < 10) $display("data mismatch cyc %0d word %0d", cyc, p); end
                end
            nwr = nwr + (|wen);
            if (|blk_r) begin errors = errors + 1; $display("tb bug: ref blocked"); end
        end
        $display("write cycles=%0d valid-word reads checked=%0d", nwr, nvalid);
        if (errors == 0) $display("ALL TESTS PASSED"); else $display("%0d ERRORS", errors);
        $finish;
    end
endmodule
