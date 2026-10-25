`timescale 1ns/1ps
// ============================================================================
// Whole-layer testbench driven by scripts/layer_golden.py
//   reads  sim/data/ifmap.hex, weights.hex   (one 32-bit word per element)
//   runs   nmc_unit + sram_bank, optional core-port noise
//   writes sim/data/ofmap_rtl.hex            (OFMap words + 16 guard words)
// Layer shape comes from plusargs: +IFR +IFC +KR +KC +CIN +COUT +S +NOISE
// Add +VCD to dump a waveform (large for big layers).
// ============================================================================
module layer_golden_tb;

    localparam ADDR_WIDTH = 17;
    localparam DIM_WIDTH  = 16;
    localparam RF_AW      = 8;
    localparam NM         = 8;
    localparam DEPTH_PM   = 4096;
    localparam MAW        = $clog2(DEPTH_PM);
    localparam CAW        = 3 + MAW;
    localparam NUM_PE     = 16;

    localparam IFM_BASE = 0;
    localparam W_BASE   = 8192;
    localparam OFM_BASE = 12288;

    reg clk = 0, rst = 1;
    always #5 clk = ~clk;

    reg start = 0;
    wire done;
    reg [DIM_WIDTH-1:0]  d_ifmap_cols = 0, d_kr = 0, d_kc = 0, d_cin = 0, d_cout = 0, d_stride = 1;
    reg [DIM_WIDTH-1:0]  d_ofr = 0, d_ofc = 0, d_wpo = 0;
    reg [ADDR_WIDTH-1:0] d_if_ch = 0, d_of_ch = 0, d_if_rstep = 0;
    reg [RF_AW-1:0]      d_rf_rstep = 0, d_rf_ch = 0;

    wire [NM-1:0]       nmc_req, nmc_we, nmc_grant, nmc_blocked, nmc_rvalid;
    wire [NM*MAW-1:0]   nmc_addr;
    wire [NM*32-1:0]    nmc_wdata, nmc_rdata;
    wire                perf_valid;
    wire [47:0]         p_total, p_load, p_comp, p_store, p_mac, p_stall, p_blk, p_ops;

    nmc_unit #(.ADDR_WIDTH(ADDR_WIDTH), .DIM_WIDTH(DIM_WIDTH), .RF_AW(RF_AW),
               .NUM_MACROS(NM), .MACRO_AW(MAW)) dut (
        .clk(clk), .rst(rst), .start(start), .done(done),
        .ifmap_cols(d_ifmap_cols), .kernel_rows(d_kr), .kernel_cols(d_kc),
        .input_channels(d_cin), .output_channels(d_cout), .stride(d_stride),
        .ofmap_rows(d_ofr), .ofmap_cols(d_ofc),
        .ifmap_ch_size(d_if_ch), .ofmap_ch_size(d_of_ch), .ifmap_row_step(d_if_rstep),
        .rf_row_step(d_rf_rstep), .rf_ch_stride(d_rf_ch), .weights_per_oc(d_wpo),
        .ifmap_base_addr(IFM_BASE[ADDR_WIDTH-1:0]),
        .weight_base_addr(W_BASE[ADDR_WIDTH-1:0]),
        .ofmap_base_addr(OFM_BASE[ADDR_WIDTH-1:0]),
        .nmc_req(nmc_req), .nmc_we(nmc_we), .nmc_addr(nmc_addr), .nmc_wdata(nmc_wdata),
        .nmc_grant(nmc_grant), .nmc_rdata(nmc_rdata),
        .dbg_compute_start(), .dbg_compute_done(), .dbg_mac_en(), .dbg_stall(), .dbg_acc_out(),
        .perf_valid(perf_valid), .perf_cyc_total(p_total), .perf_cyc_load(p_load),
        .perf_cyc_compute(p_comp), .perf_cyc_store(p_store), .perf_cyc_mac(p_mac),
        .perf_cyc_stall(p_stall), .perf_cyc_mem_blocked(p_blk), .perf_mac_ops(p_ops));

    reg                core_req = 0, core_we = 0;
    reg  [CAW-1:0]     core_addr = 0;
    reg  [31:0]        core_wdata = 0;
    wire [31:0]        core_rdata;
    wire               core_rvalid;

    sram_bank #(.DATA_WIDTH(32), .NUM_MACROS(NM), .DEPTH_PER_MACRO(DEPTH_PM)) bank (
        .clk(clk), .rst(rst),
        .core_req(core_req), .core_we(core_we), .core_addr(core_addr),
        .core_wdata(core_wdata), .core_rdata(core_rdata), .core_rvalid(core_rvalid),
        .nmc_req(nmc_req), .nmc_we(nmc_we), .nmc_addr(nmc_addr), .nmc_wdata(nmc_wdata),
        .nmc_grant(nmc_grant), .nmc_blocked(nmc_blocked),
        .nmc_rdata(nmc_rdata), .nmc_rvalid(nmc_rvalid));

    // core hammering the bank while the NMC runs (loads / stores must retry cleanly)
    reg noise_en = 0;
    always @(negedge clk) if (noise_en) begin
        core_req  = $random & 1;
        core_we   = 1'b0;
        core_addr = $random;
    end

    task core_write(input [CAW-1:0] a, input [31:0] d);
        begin
            @(negedge clk); core_req = 1; core_we = 1; core_addr = a; core_wdata = d;
            @(negedge clk); core_req = 0; core_we = 0;
        end
    endtask

    task core_read(input [CAW-1:0] a, output [31:0] d);
        begin
            @(negedge clk); core_req = 1; core_we = 0; core_addr = a;
            @(negedge clk); core_req = 0;
            d = core_rdata;
        end
    endtask

    reg [31:0] ifm_mem [0:8191];
    reg [31:0] w_mem   [0:4095];

    integer ifr, ifc, kr, kc, cin, cout, s, noise;
    integer ofr, ofc, patch_r, patch_c, n_if, n_w, n_of;
    integer i, cyc, fd;
    reg [31:0] rd;
    reg        timed_out;

    initial begin
        if (!$value$plusargs("IFR=%d",   ifr))   ifr   = 18;
        if (!$value$plusargs("IFC=%d",   ifc))   ifc   = 18;
        if (!$value$plusargs("KR=%d",    kr))    kr    = 3;
        if (!$value$plusargs("KC=%d",    kc))    kc    = 3;
        if (!$value$plusargs("CIN=%d",   cin))   cin   = 3;
        if (!$value$plusargs("COUT=%d",  cout))  cout  = 8;
        if (!$value$plusargs("S=%d",     s))     s     = 1;
        if (!$value$plusargs("NOISE=%d", noise)) noise = 0;
        if ($test$plusargs("VCD")) begin
            $dumpfile("sim/waveform/layer_golden_wave.vcd");
            $dumpvars(0, layer_golden_tb);
        end

        ofr = (ifr - kr) / s + 1;   ofc = (ifc - kc) / s + 1;
        patch_r = 3*s + kr;         patch_c = 3*s + kc;
        n_if = cin*ifr*ifc;  n_w = cout*cin*kr*kc;  n_of = cout*ofr*ofc;

        // read exactly the words this layer needs (no "not enough words" warnings)
        $readmemh("sim/data/ifmap.hex",   ifm_mem, 0, n_if - 1);
        $readmemh("sim/data/weights.hex", w_mem,   0, n_w  - 1);

        d_ifmap_cols = ifc; d_kr = kr; d_kc = kc; d_cin = cin; d_cout = cout; d_stride = s;
        d_ofr = ofr; d_ofc = ofc;
        d_if_ch = ifr*ifc; d_of_ch = ofr*ofc; d_if_rstep = s*ifc;
        d_rf_rstep = s*patch_c; d_rf_ch = patch_r*patch_c;
        d_wpo = cin*kr*kc;

        #22 rst = 0;

        for (i = 0; i < n_if;      i = i + 1) core_write(IFM_BASE + i, ifm_mem[i]);
        for (i = 0; i < n_w;       i = i + 1) core_write(W_BASE   + i, w_mem[i]);
        for (i = 0; i < n_of + 16; i = i + 1) core_write(OFM_BASE + i, 32'hDEADBEEF);

        noise_en = noise;
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        cyc = 0;
        while (!done && cyc < 5000000) begin @(negedge clk); cyc = cyc + 1; end
        timed_out = !done;                 // sample the 1-cycle done pulse NOW
        noise_en = 0; @(negedge clk); core_req = 0;
        if (timed_out) begin $display("TIMEOUT"); $finish; end

        fd = $fopen("sim/data/ofmap_rtl.hex", "w");
        for (i = 0; i < n_of + 16; i = i + 1) begin
            core_read(OFM_BASE + i, rd);
            $fdisplay(fd, "%08x", rd);
        end
        $fclose(fd);

        $display("CYCLES=%0d", cyc);
        $display("PERF total=%0d mac=%0d ops=%0d stall=%0d blocked=%0d",
                 p_total, p_mac, p_ops, p_stall, p_blk);
        $finish;
    end
endmodule