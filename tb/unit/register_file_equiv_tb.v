`timescale 1ns/1ps
// Randomised equivalence: new register_file (banked / flat, TRACK_READS 1 / 0) vs the original model.
// Writes obey the banked contract (consecutive addresses in one cycle); reads, chk, flush are random.
//   iverilog -g2012 -o sim/build/rf_equiv.vvp tb/unit/register_file_equiv_tb.v tb/models/register_file_ref.v rtl/regfile/register_file.v && vvp sim/build/rf_equiv.vvp
module register_file_equiv_tb;
    parameter DW = 8, DEPTH = 64, NR = 16, NW = 8, AW = $clog2(DEPTH), CYCLES = 30000;

    reg clk = 0, rst = 1, flush = 0;  always #5 clk = ~clk;
    reg [NW-1:0]     wen = 0;
    reg [AW*NW-1:0]  waddr = 0, caddr = 0;
    reg [DW*NW-1:0]  wdata = 0;
    reg [NR-1:0]     ren = 0, rlast = 0;
    reg [AW*NR-1:0]  raddr = 0;
    wire [NR-1:0]    zero_last = {NR{1'b0}};

    // outputs: ref (full tracking), ref0 (read_last = 0), A banked+track, B banked no-track, C flat+track
    wire [NW-1:0] blk_r, blk_r0, blk_a, blk_b, blk_c, cf_r, cf_r0, cf_a, cf_b, cf_c;
    wire [DW*NR-1:0] rd_r, rd_r0, rd_a, rd_b, rd_c;
    wire [NR-1:0]    rv_r, rv_r0, rv_a, rv_b, rv_c;

    `define PORTS(blk, cf, rd, rv, last) .clk(clk), .rst(rst), .flush(flush), .write_en(wen), .write_addr(waddr), \
        .write_data(wdata), .write_blocked(blk), .chk_addr(caddr), .chk_free(cf), .read_en(ren), .read_last(last), \
        .read_addr(raddr), .read_data(rd), .read_valid(rv)

    register_file_ref #(.DATA_WIDTH(DW), .DEPTH(DEPTH), .NUM_READ_PORTS(NR), .NUM_WRITE_PORTS(NW)) ref_i (`PORTS(blk_r,  cf_r,  rd_r,  rv_r,  rlast));
    register_file_ref #(.DATA_WIDTH(DW), .DEPTH(DEPTH), .NUM_READ_PORTS(NR), .NUM_WRITE_PORTS(NW)) ref0_i(`PORTS(blk_r0, cf_r0, rd_r0, rv_r0, zero_last));
    register_file #(.DATA_WIDTH(DW), .DEPTH(DEPTH), .NUM_READ_PORTS(NR), .NUM_WRITE_PORTS(NW),
                    .BANKED_STORAGE(1), .TRACK_READS(1)) a_i (`PORTS(blk_a, cf_a, rd_a, rv_a, rlast));
    register_file #(.DATA_WIDTH(DW), .DEPTH(DEPTH), .NUM_READ_PORTS(NR), .NUM_WRITE_PORTS(NW),
                    .BANKED_STORAGE(1), .TRACK_READS(0)) b_i (`PORTS(blk_b, cf_b, rd_b, rv_b, rlast));
    register_file #(.DATA_WIDTH(DW), .DEPTH(DEPTH), .NUM_READ_PORTS(NR), .NUM_WRITE_PORTS(NW),
                    .BANKED_STORAGE(0), .TRACK_READS(1)) c_i (`PORTS(blk_c, cf_c, rd_c, rv_c, rlast));

    integer errors = 0, cyc, p, base, nlive_max = 0, nblk = 0, nlast = 0;

    task cmp(input [127:0] n, input [DW*NR-1:0] g1, input [DW*NR-1:0] e1, input [NR-1:0] g2, input [NR-1:0] e2,
             input [NW-1:0] g3, input [NW-1:0] e3, input [NW-1:0] g4, input [NW-1:0] e4);
        if (g1 !== e1 || g2 !== e2 || g3 !== e3 || g4 !== e4) begin
            errors = errors + 1;
            if (errors < 10) $display("MISMATCH %0s cyc %0d: data %b/%b valid %b/%b blk %b/%b free %b/%b", n, cyc, g1 === e1, 1'b1, g2, e2, g3, e3, g4, e4);
        end
    endtask

    initial begin
        #22 rst = 0;
        for (cyc = 0; cyc < CYCLES; cyc = cyc + 1) begin
            @(negedge clk);
            flush = ($random & 31) == 0;
            base  = $random & (DEPTH - 1);
            for (p = 0; p < NW; p = p + 1) begin
                wen[p] = ($random & 3) != 0;
                waddr[p*AW +: AW] = (base + p) & (DEPTH - 1);          // consecutive window
                wdata[p*DW +: DW] = $random;
                caddr[p*AW +: AW] = $random;
            end
            for (p = 0; p < NR; p = p + 1) begin
                ren[p] = $random; rlast[p] = (($random & 3) == 0);
                raddr[p*AW +: AW] = $random;
            end
            #1;
            cmp("A banked+track", rd_a, rd_r,  rv_a, rv_r,  blk_a, blk_r,  cf_a, cf_r);
            cmp("C flat+track",   rd_c, rd_r,  rv_c, rv_r,  blk_c, blk_r,  cf_c, cf_r);
            cmp("B banked/notrk", rd_b, rd_r0, rv_b, rv_r0, blk_b, blk_r0, cf_b, cf_r0);
            nblk = nblk + (|blk_r);  nlast = nlast + (|(rlast & ren));
        end
        $display("cycles=%0d blocked-write cycles=%0d  read_last cycles=%0d", CYCLES, nblk, nlast);
        if (errors == 0) $display("ALL TESTS PASSED"); else $display("%0d ERRORS", errors);
        $finish;
    end
endmodule