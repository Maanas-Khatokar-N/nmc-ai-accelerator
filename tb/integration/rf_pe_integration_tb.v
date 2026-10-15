`timescale 1ns/1ps
// Integration: 8-bit IFMap RF (16 rd) + 8-bit weight RF (1 rd) + 32-bit psum RF -> pe_array
module rf_pe_integration_tb;
    localparam N=16, AW=8;
    reg clk=0, rst=1, flush=0, clear=0, psum_load=0, mac_en=0;
    always #5 clk=~clk;

    // IFMap RF
    reg  [N-1:0] i_wen=0; reg [AW*N-1:0] i_wa=0; reg [8*N-1:0] i_wd=0; wire [N-1:0] i_blk;
    reg  [N-1:0] i_ren=0, i_rlast=0; reg [AW*N-1:0] i_ra=0; wire [8*N-1:0] i_rd; wire [N-1:0] i_rv;
    register_file #(.DATA_WIDTH(8),.DEPTH(256),.NUM_READ_PORTS(N),.NUM_WRITE_PORTS(N)) ifrf(
        .clk(clk),.rst(rst),.flush(flush),.write_en(i_wen),.write_addr(i_wa),.write_data(i_wd),.write_blocked(i_blk),
        .read_en(i_ren),.read_last(i_rlast),.read_addr(i_ra),.read_data(i_rd),.read_valid(i_rv));

    // Weight RF (1 read port, 1 write port)
    reg w_wen=0; reg [AW-1:0] w_wa=0; reg [7:0] w_wd=0; wire w_blk;
    reg w_ren=0, w_rlast=0; reg [AW-1:0] w_ra=0; wire [7:0] w_rd; wire w_rv;
    register_file #(.DATA_WIDTH(8),.DEPTH(256),.NUM_READ_PORTS(1),.NUM_WRITE_PORTS(1)) wrf(
        .clk(clk),.rst(rst),.flush(flush),.write_en(w_wen),.write_addr(w_wa),.write_data(w_wd),.write_blocked(w_blk),
        .read_en(w_ren),.read_last(w_rlast),.read_addr(w_ra),.read_data(w_rd),.read_valid(w_rv));

    // Psum RF (32-bit)
    reg  [N-1:0] p_wen=0; reg [AW*N-1:0] p_wa=0; reg [32*N-1:0] p_wd=0; wire [N-1:0] p_blk;
    reg  [N-1:0] p_ren=0, p_rlast=0; reg [AW*N-1:0] p_ra=0; wire [32*N-1:0] p_rd; wire [N-1:0] p_rv;
    register_file #(.DATA_WIDTH(32),.DEPTH(256),.NUM_READ_PORTS(N),.NUM_WRITE_PORTS(N)) psrf(
        .clk(clk),.rst(rst),.flush(flush),.write_en(p_wen),.write_addr(p_wa),.write_data(p_wd),.write_blocked(p_blk),
        .read_en(p_ren),.read_last(p_rlast),.read_addr(p_ra),.read_data(p_rd),.read_valid(p_rv));

    wire signed [32*N-1:0] acc;
    pe_array #(.WEIGHT_BROADCAST(1)) pea(.clk(clk),.rst(rst),.clear(clear),.pe_mask({N{1'b1}}),
        .psum_load(psum_load),.mac_en(mac_en),.psum_in(p_rd),.ifmap_in(i_rd),.weight_in(w_rd),.acc_out(acc));

    integer i, errors=0;
    initial begin
        #12 rst=0;
        @(negedge clk);
        // load pixels (i+1) at addr i, weight 3 at addr 0, psums 1000+i at addr i
        i_wen={N{1'b1}}; p_wen={N{1'b1}}; w_wen=1; w_wa=0; w_wd=3;
        for (i=0;i<N;i=i+1) begin
            i_wa[i*AW+:AW]=i; i_wd[i*8+:8]=i+1;
            p_wa[i*AW+:AW]=i; p_wd[i*32+:32]=1000+i;
            i_ra[i*AW+:AW]=i; p_ra[i*AW+:AW]=i;
        end
        @(negedge clk); i_wen=0; p_wen=0; w_wen=0;

        // reload psums into PEs (consume psum, last use)
        p_ren={N{1'b1}}; p_rlast={N{1'b1}}; psum_load=1;
        @(negedge clk); psum_load=0; p_ren=0; p_rlast=0;

        // MAC: gate mac_en on all operands valid; read pixels (not last), weight (not last)
        if (&i_rv && w_rv) begin mac_en=1; i_ren={N{1'b1}}; w_ren=1; end
        @(negedge clk); mac_en=0; i_ren=0; w_ren=0;
        for (i=0;i<N;i=i+1)
            if (acc[i*32+:32] !== 1000+i+3*(i+1)) begin errors=errors+1; $display("FAIL PE%0d got %0d exp %0d",i,acc[i*32+:32],1000+i+3*(i+1)); end
        // psum RF now consumed: reading must be invalid
        if (p_rv !== 16'h0000) begin errors=errors+1; $display("FAIL psum RF not consumed %b",p_rv); end
        // ifmap still live (11)
        if (i_rv !== 16'hFFFF) begin errors=errors+1; $display("FAIL ifmap should stay live"); end
        if (errors==0) $display("INTEGRATION PASSED"); else $display("%0d ERRORS",errors);
        $finish;
    end
endmodule