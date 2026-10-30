`timescale 1ns/1ps
// ============================================================================
// Cluster testbench driven by scripts/tiled_layer_golden.py
//
//   host (core port) --> nmc_cluster { 4 x ( sram_bank + nmc_unit ) }
//
// Program = sim/data/prog.hex, REC words per (round, pass, bank):
//   0 en  1 ifmap_cols 2 kr 3 kc 4 cin 5 cout 6 stride 7 ofr 8 ofc 9 if_ch 10 of_ch
//   11 if_rstep 12 rf_rstep 13 rf_ch 14 wpo 15 if_base 16 w_base 17 ofm_base
//   18 accumulate 19 n_if_words 20 n_w_words 21 drain_words
// Data files  sim/data/r<R>_p<P>_b<B>_{if,w}.hex  (loaded when n_*_words > 0)
// Output      sim/data/r<R>_p<P>_b<B>_of.hex      (when drain_words > 0)
// Plusargs: +NROUNDS +NPASS [+NOISE] [+VCD]
// ============================================================================
module cluster_tb;
    localparam NB = 4, ADDR_WIDTH = 17, DIM_WIDTH = 16, RF_AW = 8, NM = 8;
    localparam DEPTH_PM = 4096, MAW = 12, CORE_AW = 3 + MAW, BANK_W = 2, CNT_W = 48;
    localparam REC = 24;

    reg clk = 0, rst = 1;  always #5 clk = ~clk;

    reg                      start = 0;
    reg  [NB-1:0]            mask = 0, accumulate = 0;
    wire                     irq;
    wire [NB-1:0]            busy;

    reg [NB*DIM_WIDTH-1:0]  d_ifc = 0, d_kr = 0, d_kc = 0, d_cin = 0, d_cout = 0, d_s = 0, d_ofr = 0, d_ofc = 0, d_wpo = 0;
    reg [NB*ADDR_WIDTH-1:0] d_ifch = 0, d_ofch = 0, d_ifrs = 0, d_ifb = 0, d_wb = 0, d_ofb = 0;
    reg [NB*RF_AW-1:0]      d_rfrs = 0, d_rfch = 0;

    reg                     core_req = 0, core_we = 0;
    reg [BANK_W+CORE_AW-1:0] core_addr = 0;
    reg [31:0]              core_wdata = 0;
    wire [31:0]             core_rdata;
    wire                    core_rvalid;

    wire [NB-1:0]           perf_valid;
    wire [NB*CNT_W-1:0]     p_total, p_mac, p_stall, p_blk, p_ops;

    nmc_cluster #(.NUM_BANKS(NB), .ADDR_WIDTH(ADDR_WIDTH), .DIM_WIDTH(DIM_WIDTH), .RF_AW(RF_AW),
                  .NUM_MACROS(NM), .DEPTH_PER_MACRO(DEPTH_PM), .CNT_W(CNT_W)) dut (
        .clk(clk), .rst(rst), .unit_mask(mask), .start(start), .accumulate(accumulate),
        .irq(irq), .unit_busy(busy),
        .ifmap_cols(d_ifc), .kernel_rows(d_kr), .kernel_cols(d_kc), .input_channels(d_cin),
        .output_channels(d_cout), .stride(d_s), .ofmap_rows(d_ofr), .ofmap_cols(d_ofc),
        .weights_per_oc(d_wpo), .ifmap_ch_size(d_ifch), .ofmap_ch_size(d_ofch),
        .ifmap_row_step(d_ifrs), .ifmap_base_addr(d_ifb), .weight_base_addr(d_wb),
        .ofmap_base_addr(d_ofb), .rf_row_step(d_rfrs), .rf_ch_stride(d_rfch),
        .core_req(core_req), .core_we(core_we), .core_addr(core_addr), .core_wdata(core_wdata),
        .core_rdata(core_rdata), .core_rvalid(core_rvalid),
        .perf_valid(perf_valid), .perf_cyc_total(p_total), .perf_cyc_mac(p_mac),
        .perf_cyc_stall(p_stall), .perf_cyc_mem_blocked(p_blk), .perf_mac_ops(p_ops),
        .pp_mask({NB{1'b0}}), .pp_start(1'b0), .pp_irq(), .pp_busy(),
        .pp_src_base({NB*ADDR_WIDTH{1'b0}}), .pp_dst_base({NB*ADDR_WIDTH{1'b0}}),
        .pp_src_cols({NB*DIM_WIDTH{1'b0}}), .pp_out_rows({NB*DIM_WIDTH{1'b0}}),
        .pp_out_cols({NB*DIM_WIDTH{1'b0}}), .pp_channels({NB*DIM_WIDTH{1'b0}}),
        .pp_pool_k({NB*DIM_WIDTH{1'b0}}), .pp_src_ch_size({NB*ADDR_WIDTH{1'b0}}),
        .pp_row_step({NB*ADDR_WIDTH{1'b0}}), .pp_total_out({NB*ADDR_WIDTH{1'b0}}),
        .pp_out_zp({NB*8{1'b0}}), .pp_relu_en({NB{1'b0}}),
        .pp_qp_we({NB{1'b0}}), .pp_qp_waddr(9'd0), .pp_qp_bias(32'd0),
        .pp_qp_mult(16'd0), .pp_qp_shift(5'd0));

    // ---------------- host-port noise: random reads on random banks ----------------
    reg noise_en = 0;
    always @(negedge clk) if (noise_en) begin
        core_req = $random & 1; core_we = 1'b0; core_addr = $random;
    end

    // ---------------- program + data ----------------
    reg [31:0] prog [0:(1<<20)-1];
    reg [31:0] dmem [0:(1<<15)-1];
    integer nrounds, npass, noise;
    reg [8*80-1:0] fname;

    function [31:0] F(input integer r, input integer p, input integer b, input integer f);
        F = prog[((r*npass + p)*NB + b)*REC + f];
    endfunction

    // 1 word / cycle burst write of dmem[0..n-1] into bank b at word offset base
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

    // pipelined burst read of n words of bank b -> file
    task drain(input integer b, input integer base, input integer n, input integer fd);
        integer i;
        begin
            for (i = 0; i <= n; i = i + 1) begin
                @(negedge clk);
                if (i > 0) $fdisplay(fd, "%08x", core_rdata);     // data of the request issued one edge ago
                if (i < n) begin core_req = 1; core_we = 0; core_addr = {b[BANK_W-1:0], base[CORE_AW-1:0]} + i; end
                else       begin core_req = 0; end
            end
        end
    endtask

    // ---------------- concurrency + perf accounting ----------------
    integer overlap_cyc = 0, max_busy = 0, nb_busy;
    integer wall_nmc = 0;
    always @(posedge clk) begin
        nb_busy = busy[0] + busy[1] + busy[2] + busy[3];
        if (nb_busy > 1) overlap_cyc = overlap_cyc + 1;
        if (nb_busy > max_busy) max_busy = nb_busy;
        if (nb_busy > 0) wall_nmc = wall_nmc + 1;
    end

    integer r, p, b, i, t, fd, errors = 0;
    reg [63:0] s_ops = 0, s_mac = 0, s_stall = 0, s_blk = 0, s_busy_cyc = 0, s_crit = 0;
    reg [63:0] mx, v;
    integer units;

    initial begin
        if (!$value$plusargs("NROUNDS=%d", nrounds)) nrounds = 1;
        if (!$value$plusargs("NPASS=%d",   npass))   npass = 1;
        if (!$value$plusargs("NOISE=%d",   noise))   noise = 0;
        if ($test$plusargs("VCD")) begin $dumpfile("sim/waveform/cluster_wave.vcd"); $dumpvars(0, cluster_tb); end
        $readmemh("sim/data/prog.hex", prog);
        #22 rst = 0;

        for (r = 0; r < nrounds; r = r + 1) begin
            for (p = 0; p < npass; p = p + 1) begin
                // ---- host loads the banks (units are idle) ----
                mask = 0; noise_en = 0;
                for (b = 0; b < NB; b = b + 1) if (F(r,p,b,0)) begin
                    mask[b] = 1'b1;
                    if (F(r,p,b,19) != 0) begin
                        $sformat(fname, "sim/data/r%0d_p%0d_b%0d_if.hex", r, p, b);
                        $readmemh(fname, dmem, 0, F(r,p,b,19)-1);
                        fill(b, F(r,p,b,15), F(r,p,b,19));
                    end
                    if (F(r,p,b,20) != 0) begin
                        $sformat(fname, "sim/data/r%0d_p%0d_b%0d_w.hex", r, p, b);
                        $readmemh(fname, dmem, 0, F(r,p,b,20)-1);
                        fill(b, F(r,p,b,16), F(r,p,b,20));
                    end
                    d_ifc[b*DIM_WIDTH +: DIM_WIDTH] = F(r,p,b,1);  d_kr[b*DIM_WIDTH +: DIM_WIDTH] = F(r,p,b,2);
                    d_kc[b*DIM_WIDTH +: DIM_WIDTH]  = F(r,p,b,3);  d_cin[b*DIM_WIDTH +: DIM_WIDTH] = F(r,p,b,4);
                    d_cout[b*DIM_WIDTH +: DIM_WIDTH] = F(r,p,b,5); d_s[b*DIM_WIDTH +: DIM_WIDTH] = F(r,p,b,6);
                    d_ofr[b*DIM_WIDTH +: DIM_WIDTH] = F(r,p,b,7);  d_ofc[b*DIM_WIDTH +: DIM_WIDTH] = F(r,p,b,8);
                    d_ifch[b*ADDR_WIDTH +: ADDR_WIDTH] = F(r,p,b,9); d_ofch[b*ADDR_WIDTH +: ADDR_WIDTH] = F(r,p,b,10);
                    d_ifrs[b*ADDR_WIDTH +: ADDR_WIDTH] = F(r,p,b,11);
                    d_rfrs[b*RF_AW +: RF_AW] = F(r,p,b,12);        d_rfch[b*RF_AW +: RF_AW] = F(r,p,b,13);
                    d_wpo[b*DIM_WIDTH +: DIM_WIDTH] = F(r,p,b,14);
                    d_ifb[b*ADDR_WIDTH +: ADDR_WIDTH] = F(r,p,b,15); d_wb[b*ADDR_WIDTH +: ADDR_WIDTH] = F(r,p,b,16);
                    d_ofb[b*ADDR_WIDTH +: ADDR_WIDTH] = F(r,p,b,17);
                    accumulate[b] = F(r,p,b,18);
                end

                // ---- all enabled units start together; wait for the cluster irq ----
                noise_en = noise;
                @(negedge clk); start = 1;
                @(negedge clk); start = 0;
                t = 0;
                while (!irq && t < 20000000) begin @(negedge clk); t = t + 1; end
                if (!irq) begin $display("TIMEOUT r%0d p%0d", r, p); errors = errors + 1; $finish; end
                noise_en = 0; @(negedge clk); core_req = 0; core_we = 0;

                // ---- perf: banks run concurrently, so the pass costs max(), not sum() ----
                mx = 0; units = 0;
                for (b = 0; b < NB; b = b + 1) if (mask[b]) begin
                    units = units + 1;
                    v = p_total[b*CNT_W +: CNT_W]; if (v > mx) mx = v;
                    s_busy_cyc = s_busy_cyc + v;
                    s_ops = s_ops + p_ops[b*CNT_W +: CNT_W];  s_mac = s_mac + p_mac[b*CNT_W +: CNT_W];
                    s_stall = s_stall + p_stall[b*CNT_W +: CNT_W]; s_blk = s_blk + p_blk[b*CNT_W +: CNT_W];
                end
                s_crit = s_crit + mx;

                // ---- drain finished OFMaps ----
                for (b = 0; b < NB; b = b + 1) if (mask[b] && F(r,p,b,21) != 0) begin
                    $sformat(fname, "sim/data/r%0d_p%0d_b%0d_of.hex", r, p, b);
                    fd = $fopen(fname, "w");
                    drain(b, F(r,p,b,17), F(r,p,b,21), fd);
                    $fclose(fd);
                end
            end
        end
        $display("CLUSTER rounds=%0d passes=%0d max_busy=%0d overlap_cycles=%0d wall_nmc=%0d",
                 nrounds, npass, max_busy, overlap_cyc, wall_nmc);
        $display("PERF crit=%0d busy_sum=%0d mac=%0d ops=%0d stall=%0d blocked=%0d",
                 s_crit, s_busy_cyc, s_mac, s_ops, s_stall, s_blk);
        $finish;
    end
endmodule