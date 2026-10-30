`timescale 1ns/1ps
// ============================================================================
// Cluster-level post-processing testbench driven by scripts/cluster_pp_golden.py
//   host (core port) --> nmc_cluster { 4 x ( sram_bank + nmc_unit + postproc_unit ) }
// Program sim/data/pp_prog.hex, REC words per bank:
//   0 en 1 src_cols 2 out_rows 3 out_cols 4 ch 5 pool 6 src_ch_size 7 row_step 8 total
//   9 zp 10 relu 11 src_base 12 dst_base 13 n_src_words
// Data  sim/data/ppc_b<B>_src.hex , ppc_b<B>_qp.hex (3 lines / channel)
// Out   sim/data/ppc_b<B>_dst.hex
// Plusargs: +NOISE [+VCD]
// ============================================================================
module cluster_pp_tb;
    localparam NB = 4, ADDR_WIDTH = 17, DIM_WIDTH = 16, RF_AW = 8, NM = 8;
    localparam DEPTH_PM = 4096, MAW = 12, CORE_AW = 3 + MAW, BANK_W = 2, CNT_W = 48, QA_W = 9;
    localparam REC = 16;

    reg clk = 0, rst = 1;  always #5 clk = ~clk;

    reg                      start = 0, pp_start = 0;
    reg  [NB-1:0]            mask = 0, pp_mask = 0, accumulate = 0;
    wire                     irq, pp_irq;
    wire [NB-1:0]            busy, pp_busy;

    reg [NB*DIM_WIDTH-1:0]  p_cols = 0, p_orows = 0, p_ocols = 0, p_ch = 0, p_pk = 0;
    reg [NB*ADDR_WIDTH-1:0] p_chsz = 0, p_rstep = 0, p_total = 0, p_sb = 0, p_db = 0;
    reg [NB*8-1:0]          p_zp = 0;
    reg [NB-1:0]            p_relu = 0, p_qwe = 0;
    reg [QA_W-1:0]          p_qa = 0;
    reg signed [31:0]       p_qb = 0;  reg [15:0] p_qm = 0;  reg [4:0] p_qs = 0;

    reg                      core_req = 0, core_we = 0;
    reg [BANK_W+CORE_AW-1:0] core_addr = 0;
    reg [31:0]               core_wdata = 0;
    wire [31:0]              core_rdata;
    wire                     core_rvalid;
    wire [NB-1:0]            perf_valid;
    wire [NB*CNT_W-1:0]      c_total, c_mac, c_stall, c_blk, c_ops;

    nmc_cluster #(.NUM_BANKS(NB), .ADDR_WIDTH(ADDR_WIDTH), .DIM_WIDTH(DIM_WIDTH), .RF_AW(RF_AW),
                  .NUM_MACROS(NM), .DEPTH_PER_MACRO(DEPTH_PM), .CNT_W(CNT_W)) dut (
        .clk(clk), .rst(rst), .unit_mask(mask), .start(start), .accumulate(accumulate),
        .irq(irq), .unit_busy(busy),
        .ifmap_cols({NB*DIM_WIDTH{1'b0}}), .kernel_rows({NB*DIM_WIDTH{1'b0}}), .kernel_cols({NB*DIM_WIDTH{1'b0}}),
        .input_channels({NB*DIM_WIDTH{1'b0}}), .output_channels({NB*DIM_WIDTH{1'b0}}), .stride({NB*DIM_WIDTH{1'b0}}),
        .ofmap_rows({NB*DIM_WIDTH{1'b0}}), .ofmap_cols({NB*DIM_WIDTH{1'b0}}), .weights_per_oc({NB*DIM_WIDTH{1'b0}}),
        .ifmap_ch_size({NB*ADDR_WIDTH{1'b0}}), .ofmap_ch_size({NB*ADDR_WIDTH{1'b0}}), .ifmap_row_step({NB*ADDR_WIDTH{1'b0}}),
        .ifmap_base_addr({NB*ADDR_WIDTH{1'b0}}), .weight_base_addr({NB*ADDR_WIDTH{1'b0}}), .ofmap_base_addr({NB*ADDR_WIDTH{1'b0}}),
        .rf_row_step({NB*RF_AW{1'b0}}), .rf_ch_stride({NB*RF_AW{1'b0}}),
        .pp_mask(pp_mask), .pp_start(pp_start), .pp_irq(pp_irq), .pp_busy(pp_busy),
        .pp_src_base(p_sb), .pp_dst_base(p_db), .pp_src_cols(p_cols), .pp_out_rows(p_orows),
        .pp_out_cols(p_ocols), .pp_channels(p_ch), .pp_pool_k(p_pk), .pp_src_ch_size(p_chsz),
        .pp_row_step(p_rstep), .pp_total_out(p_total), .pp_out_zp(p_zp), .pp_relu_en(p_relu),
        .pp_qp_we(p_qwe), .pp_qp_waddr(p_qa), .pp_qp_bias(p_qb), .pp_qp_mult(p_qm), .pp_qp_shift(p_qs),
        .core_req(core_req), .core_we(core_we), .core_addr(core_addr), .core_wdata(core_wdata),
        .core_rdata(core_rdata), .core_rvalid(core_rvalid),
        .perf_valid(perf_valid), .perf_cyc_total(c_total), .perf_cyc_mac(c_mac),
        .perf_cyc_stall(c_stall), .perf_cyc_mem_blocked(c_blk), .perf_mac_ops(c_ops));

    reg noise_en = 0;
    always @(negedge clk) if (noise_en) begin
        core_req = $random & 1; core_we = 1'b0; core_addr = $random;
    end

    reg [31:0] prog [0:NB*REC-1];
    reg [31:0] dmem [0:16383];
    reg [8*64-1:0] fname;
    integer noise;

    function [31:0] F(input integer b, input integer f);  F = prog[b*REC + f];  endfunction

    task fill(input integer b, input integer base, input integer n);
        integer i;
        begin
            for (i = 0; i < n; i = i + 1) begin
                @(negedge clk); core_req = 1; core_we = 1;
                core_addr = {b[BANK_W-1:0], base[CORE_AW-1:0]} + i; core_wdata = dmem[i];
            end
            @(negedge clk); core_req = 0; core_we = 0;
        end
    endtask
    task drain(input integer b, input integer base, input integer n, input integer fd);
        integer i;
        begin
            for (i = 0; i <= n; i = i + 1) begin
                @(negedge clk);
                if (i > 0) $fdisplay(fd, "%08x", core_rdata);
                if (i < n) begin core_req = 1; core_we = 0; core_addr = {b[BANK_W-1:0], base[CORE_AW-1:0]} + i; end
                else       begin core_req = 0; end
            end
        end
    endtask

    // concurrency + conv-isolation checks
    integer overlap_cyc = 0, max_busy = 0, nb_busy, wall = 0, conv_busy_seen = 0;
    always @(posedge clk) begin
        nb_busy = pp_busy[0] + pp_busy[1] + pp_busy[2] + pp_busy[3];
        if (nb_busy > 1) overlap_cyc = overlap_cyc + 1;
        if (nb_busy > max_busy) max_busy = nb_busy;
        if (nb_busy > 0) wall = wall + 1;
        if (|busy) conv_busy_seen = 1;
    end

    integer b, i, t, fd, n_dst;
    initial begin
        if (!$value$plusargs("NOISE=%d", noise)) noise = 0;
        if ($test$plusargs("VCD")) begin $dumpfile("sim/waveform/cluster_pp_wave.vcd"); $dumpvars(0, cluster_pp_tb); end
        $readmemh("sim/data/pp_prog.hex", prog);
        #22 rst = 0;

        // ---- host loads src OFMaps, guard words, descriptors, requant params ----
        for (b = 0; b < NB; b = b + 1) if (F(b,0)) begin
            pp_mask[b] = 1'b1;
            $sformat(fname, "sim/data/ppc_b%0d_src.hex", b);
            $readmemh(fname, dmem, 0, F(b,13)-1);
            fill(b, F(b,11), F(b,13));
            for (i = 0; i < 16384; i = i + 1) dmem[i] = 32'hDEADBEEF;
            fill(b, F(b,12), F(b,8) + 16);

            p_cols [b*DIM_WIDTH +: DIM_WIDTH] = F(b,1);  p_orows[b*DIM_WIDTH +: DIM_WIDTH] = F(b,2);
            p_ocols[b*DIM_WIDTH +: DIM_WIDTH] = F(b,3);  p_ch   [b*DIM_WIDTH +: DIM_WIDTH] = F(b,4);
            p_pk   [b*DIM_WIDTH +: DIM_WIDTH] = F(b,5);
            p_chsz [b*ADDR_WIDTH +: ADDR_WIDTH] = F(b,6); p_rstep[b*ADDR_WIDTH +: ADDR_WIDTH] = F(b,7);
            p_total[b*ADDR_WIDTH +: ADDR_WIDTH] = F(b,8);
            p_zp   [b*8 +: 8] = F(b,9);  p_relu[b] = F(b,10);
            p_sb   [b*ADDR_WIDTH +: ADDR_WIDTH] = F(b,11); p_db[b*ADDR_WIDTH +: ADDR_WIDTH] = F(b,12);

            $sformat(fname, "sim/data/ppc_b%0d_qp.hex", b);
            $readmemh(fname, dmem, 0, 3*F(b,4)-1);
            for (i = 0; i < F(b,4); i = i + 1) begin
                @(negedge clk); p_qwe = 0; p_qwe[b] = 1'b1; p_qa = i;
                p_qb = dmem[3*i]; p_qm = dmem[3*i+1]; p_qs = dmem[3*i+2];
            end
            @(negedge clk); p_qwe = 0;
        end

        // ---- start the round; a conv start while post-proc runs must be ignored ----
        noise_en = noise;
        @(negedge clk); pp_start = 1;
        @(negedge clk); pp_start = 0;
        mask = 4'hF;
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        t = 0;
        while (!pp_irq && t < 5000000) begin @(negedge clk); t = t + 1; end
        if (!pp_irq) begin $display("TIMEOUT"); $finish; end
        noise_en = 0; @(negedge clk); core_req = 0; core_we = 0;
        mask = 0;
        repeat (4) @(negedge clk);
        if (pp_busy !== 4'b0000) $display("ERROR pp_busy still set after irq");

        for (b = 0; b < NB; b = b + 1) if (F(b,0)) begin
            $sformat(fname, "sim/data/ppc_b%0d_dst.hex", b);
            fd = $fopen(fname, "w");
            drain(b, F(b,12), F(b,8) + 16, fd);
            $fclose(fd);
        end
        $display("CLUSTERPP max_busy=%0d overlap_cycles=%0d wall=%0d conv_busy=%0d irq_cycles=%0d",
                 max_busy, overlap_cyc, wall, conv_busy_seen, t);
        $finish;
    end
endmodule