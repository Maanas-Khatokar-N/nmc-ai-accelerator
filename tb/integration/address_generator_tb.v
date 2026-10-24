`timescale 1ns/1ps
// ============================================================================
// End-to-end, self-checking testbench: NMC UNIT (address generator + RFs + access controller + PE array) + sram_bank
//
//   core port (tb) --> sram_bank <--NMC port-- nmc_unit
//
// The OFMap written to SRAM is compared word by word with a software
// convolution. Every case runs quiet and with random core reads hammering the bank.
// ============================================================================
module address_generator_tb;

    localparam ADDR_WIDTH = 17;
    localparam DIM_WIDTH  = 16;
    localparam RF_AW      = 8;
    localparam NM         = 8;
    localparam DEPTH_PM   = 1024;
    localparam MAW        = $clog2(DEPTH_PM);
    localparam CAW        = 3 + MAW;
    localparam NUM_PE     = 16;
    localparam LANE_W     = 4;

    localparam IFM_BASE = 0;
    localparam W_BASE   = 3000;
    localparam OFM_BASE = 5000;

    reg clk = 0, rst = 1;
    always #5 clk = ~clk;

    reg start = 0;
    wire done;
    reg [DIM_WIDTH-1:0]  d_ifmap_cols = 0, d_kr = 0, d_kc = 0, d_cin = 0, d_cout = 0, d_stride = 1;
    reg [DIM_WIDTH-1:0]  d_ofr = 0, d_ofc = 0, d_wpo = 0;
    reg [ADDR_WIDTH-1:0] d_if_ch = 0, d_of_ch = 0, d_if_rstep = 0;
    reg [RF_AW-1:0]      d_rf_rstep = 0, d_rf_ch = 0;

    wire compute_start, compute_done, pe_mac_en, ac_stall;
    wire signed [NUM_PE*32-1:0] acc_out;
    wire [NM-1:0]           nmc_req, nmc_we, nmc_grant, nmc_blocked, nmc_rvalid;
    wire [NM*MAW-1:0]       nmc_addr;
    wire [NM*32-1:0]        nmc_wdata, nmc_rdata;

    nmc_unit #(
        .ADDR_WIDTH(ADDR_WIDTH), .DIM_WIDTH(DIM_WIDTH), .RF_AW(RF_AW),
        .NUM_MACROS(NM), .MACRO_AW(MAW)
    ) dut (
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
        .dbg_compute_start(compute_start), .dbg_compute_done(compute_done),
        .dbg_mac_en(pe_mac_en), .dbg_stall(ac_stall), .dbg_acc_out(acc_out)
    );

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

    // ---------------- core-port noise ----------------
    reg noise_en = 0;
    always @(negedge clk) if (noise_en) begin
        core_req   = ($random & 1);
        core_we    = 1'b0;
        core_addr  = $random;
    end

    integer errors = 0, total_errors = 0;

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

    reg signed [7:0] ifm [0:4095];
    reg signed [7:0] wgt [0:4095];

    integer ifr, ifc, kr, kc, cin, cout, s;
    integer ofr, ofc, patch_r, patch_c;
    integer c, r, col, oc, ic, ki, kj, idx, cyc;
    integer i;
    integer stall_cycles;
    reg signed [31:0] exp_v;
    reg [31:0] rd;

    always @(posedge clk) if (start || dut.u_ag.state != 0) begin
        if (ac_stall) stall_cycles = stall_cycles + 1;
    end

    task run_case(input integer p_ifr, input integer p_ifc, input integer p_kr, input integer p_kc,
                  input integer p_cin, input integer p_cout, input integer p_s, input integer p_noise);
        begin
            ifr = p_ifr; ifc = p_ifc; kr = p_kr; kc = p_kc;
            cin = p_cin; cout = p_cout; s = p_s;
            ofr = (ifr - kr) / s + 1;
            ofc = (ifc - kc) / s + 1;
            patch_r = 3*s + kr;
            patch_c = 3*s + kc;
            errors = 0;
            stall_cycles = 0;

            if (cin*patch_r*patch_c > 256 || cin*kr*kc > 256) begin
                $display("SKIP case (doesn't fit RF): %0dx%0d K%0dx%0d Cin%0d Cout%0d S%0d", ifr, ifc, kr, kc, cin, cout, s);
            end else begin
                noise_en = 0; core_req = 0; core_we = 0;

                d_ifmap_cols = ifc;  d_kr = kr; d_kc = kc; d_cin = cin; d_cout = cout; d_stride = s;
                d_ofr = ofr; d_ofc = ofc;
                d_if_ch = ifr*ifc;   d_of_ch = ofr*ofc;   d_if_rstep = s*ifc;
                d_rf_rstep = s*patch_c;  d_rf_ch = patch_r*patch_c;
                d_wpo = cin*kr*kc;

                for (i = 0; i < cin*ifr*ifc;  i = i + 1) ifm[i] = ($random & 15) - 8;
                for (i = 0; i < cout*cin*kr*kc; i = i + 1) wgt[i] = ($random & 7) - 4;

                for (i = 0; i < cin*ifr*ifc;  i = i + 1) core_write(IFM_BASE + i, {24'b0, ifm[i]});
                for (i = 0; i < cout*cin*kr*kc; i = i + 1) core_write(W_BASE + i, {24'b0, wgt[i]});
                for (i = 0; i < cout*ofr*ofc + 16; i = i + 1) core_write(OFM_BASE + i, 32'hDEADBEEF);

                noise_en = p_noise;

                @(negedge clk); start = 1;
                @(negedge clk); start = 0;
                cyc = 0;
                while (!done && cyc < 2000000) begin @(negedge clk); cyc = cyc + 1; end
                if (!done) begin $display("TIMEOUT"); errors = errors + 1; end
                noise_en = 0; @(negedge clk); core_req = 0;

                for (oc = 0; oc < cout; oc = oc + 1)
                  for (r = 0; r < ofr; r = r + 1)
                    for (col = 0; col < ofc; col = col + 1) begin
                        exp_v = 0;
                        for (ic = 0; ic < cin; ic = ic + 1)
                          for (ki = 0; ki < kr; ki = ki + 1)
                            for (kj = 0; kj < kc; kj = kj + 1)
                              exp_v = exp_v
                                + ifm[ic*ifr*ifc + (r*s + ki)*ifc + (col*s + kj)]
                                * wgt[((oc*cin + ic)*kr + ki)*kc + kj];
                        core_read(OFM_BASE + oc*ofr*ofc + r*ofc + col, rd);
                        if (rd !== exp_v) begin
                            errors = errors + 1;
                            if (errors < 10)
                                $display("  MISMATCH oc%0d (%0d,%0d): got %0d exp %0d", oc, r, col, $signed(rd), exp_v);
                        end
                    end
                for (i = 0; i < 16; i = i + 1) begin
                    core_read(OFM_BASE + cout*ofr*ofc + i, rd);
                    if (rd !== 32'hDEADBEEF) begin errors = errors + 1; $display("  GUARD overwritten at +%0d", i); end
                end

                $display("%s  ifm %0dx%0dx%0d  K%0dx%0d  Cout %0d  S%0d  noise=%0d  cycles=%0d  stalls=%0d  errors=%0d",
                         (errors == 0) ? "PASS" : "FAIL", ifr, ifc, cin, kr, kc, cout, s, p_noise, cyc, stall_cycles, errors);
                total_errors = total_errors + errors;
            end
        end
    endtask

    initial begin
        $dumpfile("sim/waveform/address_generator_wave.vcd");
        $dumpvars(0, address_generator_tb);

        #22 rst = 0;

        //       ifr ifc kr kc cin cout s noise
        run_case(  8,  8, 3, 3,  2,  2, 1, 0);
        run_case(  8,  8, 3, 3,  2,  2, 1, 1);
        run_case( 14, 14, 3, 3,  2,  3, 2, 0);
        run_case( 14, 14, 3, 3,  2,  3, 2, 1);
        run_case( 10, 10, 5, 5,  1,  2, 1, 0);
        run_case(  6,  6, 1, 1,  3,  2, 1, 1);
        run_case(  7, 11, 2, 3,  2,  2, 1, 0);
        run_case(  4,  4, 3, 3,  1,  1, 1, 1);
        run_case(  9,  9, 3, 3,  1,  1, 2, 0);
        run_case( 20, 20, 3, 3,  1,  1, 1, 1);

        if (total_errors == 0) $display("\nALL TESTS PASSED");
        else                   $display("\n%0d TOTAL ERRORS", total_errors);
        $finish;
    end

endmodule