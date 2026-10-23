`timescale 1ns/1ps
// ============================================================================
// MILESTONE 1: one 6x6 IFMap, 1 i/p and o/p channel, one 4x4 output tile, 3x3 kernel -> 16 outputs in 9 compute cycles
//
//   core port (tb) --> sram_bank <--NMC port-- address_generator
//        IFMap RF / weight RF --> access_controller --> PE array --> store --> OFMap
//
// The OFMap written to SRAM is compared word by word with a software
// convolution. Every case runs quiet and with random core reads hammering the bank.
// ============================================================================
module milestone1_tb;

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

    wire compute_start, seq_valid, seq_last, ifrf_flush, wrf_flush;
    wire [NUM_PE*RF_AW-1:0] ifmap_rf_addr;
    wire [RF_AW-1:0]        weight_rf_addr;
    wire [NUM_PE-1:0]       pe_mask;
    wire [NM-1:0]           ifrf_wr_en, wrf_wr_en;
    wire [NM*RF_AW-1:0]     rf_wr_addr;
    wire [NM-1:0]           nmc_req, nmc_we, nmc_grant, nmc_blocked, nmc_rvalid;
    wire [NM*MAW-1:0]       nmc_addr;
    wire [NM*LANE_W-1:0]    st_lane_sel;
    wire [NM*32-1:0]        nmc_wdata, nmc_rdata;

    wire compute_done, seq_enable;

    address_generator #(
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
        .compute_start(compute_start), .compute_done(compute_done),
        .seq_enable(seq_enable), .seq_valid(seq_valid), .seq_last(seq_last),
        .ifmap_rf_addr(ifmap_rf_addr), .weight_rf_addr(weight_rf_addr), .pe_mask(pe_mask),
        .ifrf_flush(ifrf_flush), .wrf_flush(wrf_flush),
        .ifrf_wr_en(ifrf_wr_en), .wrf_wr_en(wrf_wr_en), .rf_wr_addr(rf_wr_addr),
        .nmc_req(nmc_req), .nmc_we(nmc_we), .nmc_addr(nmc_addr), .nmc_grant(nmc_grant),
        .st_lane_sel(st_lane_sel)
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

    // ---------------- register files ----------------
    wire [NM*8-1:0] rf_wdata8;
    genvar gi;
    generate for (gi = 0; gi < NM; gi = gi + 1) begin : g_wd
        assign rf_wdata8[gi*8 +: 8] = nmc_rdata[gi*32 +: 8];
    end endgenerate

    wire [NUM_PE-1:0] ifrf_read_en;
    wire              wrf_read_en;
    wire [NUM_PE*8-1:0] i_rd;  wire [NUM_PE-1:0] i_rv;  wire [NM-1:0] i_blk;
    wire [7:0]          w_rd;  wire [0:0]        w_rv;  wire [NM-1:0] w_blk;

    register_file #(.DATA_WIDTH(8), .DEPTH(256), .NUM_READ_PORTS(NUM_PE), .NUM_WRITE_PORTS(NM)) ifrf (
        .clk(clk), .rst(rst), .flush(ifrf_flush),
        .write_en(ifrf_wr_en), .write_addr(rf_wr_addr), .write_data(rf_wdata8), .write_blocked(i_blk),
        .read_en(ifrf_read_en), .read_last({NUM_PE{1'b0}}),
        .read_addr(ifmap_rf_addr), .read_data(i_rd), .read_valid(i_rv));

    register_file #(.DATA_WIDTH(8), .DEPTH(256), .NUM_READ_PORTS(1), .NUM_WRITE_PORTS(NM)) wrf (
        .clk(clk), .rst(rst), .flush(wrf_flush),
        .write_en(wrf_wr_en), .write_addr(rf_wr_addr), .write_data(rf_wdata8), .write_blocked(w_blk),
        .read_en(wrf_read_en), .read_last(1'b0),
        .read_addr(weight_rf_addr), .read_data(w_rd), .read_valid(w_rv));

    // ---------------- access controller + PE array ----------------
    wire pe_clear, pe_mac_en, pe_psum_load, ac_stall;
    wire [NUM_PE-1:0] psrf_read_en, psrf_read_last;
    wire psum_ack;

    access_controller #(.NUM_PE(NUM_PE)) ac (
        .clk(clk), .rst(rst),
        .compute_start(compute_start), .seq_valid(seq_valid), .seq_last(seq_last), .pe_mask(pe_mask),
        .seq_enable(seq_enable), .compute_done(compute_done),
        .ifrf_read_valid(i_rv), .wrf_read_valid(w_rv[0]),
        .ifrf_read_en(ifrf_read_en), .wrf_read_en(wrf_read_en),
        .psum_req(1'b0), .psrf_read_valid({NUM_PE{1'b0}}),     // reload path not wired yet
        .psrf_read_en(psrf_read_en), .psrf_read_last(psrf_read_last), .psum_ack(psum_ack),
        .pe_clear(pe_clear), .pe_mac_en(pe_mac_en), .pe_psum_load(pe_psum_load), .stall(ac_stall));

    wire signed [NUM_PE*32-1:0] acc_out;
    pe_array #(.WEIGHT_BROADCAST(1)) pea (
        .clk(clk), .rst(rst), .clear(pe_clear), .pe_mask(pe_mask),
        .psum_load(pe_psum_load), .mac_en(pe_mac_en), .psum_in({NUM_PE*32{1'b0}}),
        .ifmap_in(i_rd), .weight_in(w_rd), .acc_out(acc_out));

    generate for (gi = 0; gi < NM; gi = gi + 1) begin : g_st
        wire [LANE_W-1:0] sel = st_lane_sel[gi*LANE_W +: LANE_W];
        assign nmc_wdata[gi*32 +: 32] = acc_out[sel*32 +: 32];
    end endgenerate

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

    // ---------------- milestone layer: 6x6 IFMap, 3x3 kernel, 1 in-ch, 1 out-ch -> 4x4 OFMap ----------------
    localparam IFM = 6, K = 3, OFM = 4;
    reg signed [7:0] ifm [0:IFM*IFM-1];
    reg signed [7:0] wgt [0:K*K-1];
    reg signed [31:0] exp_o [0:OFM*OFM-1];

    // ---------------- cycle accounting (sampled on posedge = what the edge latches) ----------------
    integer cyc_now = 0, t_start, t_cstart, t_first_mac, t_last_mac, t_cdone, t_done;
    integer mac_cycles, stall_cycles, seen_cdone;
    reg     measuring = 0;
    reg signed [NUM_PE*32-1:0] acc_snap;

    always @(posedge clk) begin
        cyc_now <= cyc_now + 1;
        if (measuring) begin
            if (start && t_start < 0)               t_start  = cyc_now;
            if (compute_start)                      t_cstart = cyc_now;
            if (pe_mac_en) begin
                if (mac_cycles == 0) t_first_mac = cyc_now;
                t_last_mac = cyc_now;
                mac_cycles = mac_cycles + 1;
            end
            if (ac_stall)                           stall_cycles = stall_cycles + 1;
            if (compute_done) begin t_cdone = cyc_now; acc_snap = acc_out; seen_cdone = seen_cdone + 1; end
            if (done)                               t_done = cyc_now;
        end
    end

    integer i, r, col, ki, kj, cyc;
    reg [31:0] rd;
    reg signed [31:0] got;

    task run(input integer noise);
        begin
            errors = 0; measuring = 0; noise_en = 0; core_req = 0; core_we = 0;
            t_start = -1; t_cstart = -1; t_first_mac = -1; t_last_mac = -1; t_cdone = -1; t_done = -1;
            mac_cycles = 0; stall_cycles = 0; seen_cdone = 0;

            d_ifmap_cols = IFM; d_kr = K; d_kc = K; d_cin = 1; d_cout = 1; d_stride = 1;
            d_ofr = OFM; d_ofc = OFM; d_if_ch = IFM*IFM; d_of_ch = OFM*OFM; d_if_rstep = IFM;
            d_rf_rstep = IFM; d_rf_ch = IFM*IFM; d_wpo = K*K;      // patch = 6x6 (3*1+3)

            for (i = 0; i < IFM*IFM; i = i + 1) ifm[i] = ($random & 15) - 8;
            for (i = 0; i < K*K;     i = i + 1) wgt[i] = ($random & 7) - 4;
            for (r = 0; r < OFM; r = r + 1) for (col = 0; col < OFM; col = col + 1) begin
                exp_o[r*OFM+col] = 0;
                for (ki = 0; ki < K; ki = ki + 1) for (kj = 0; kj < K; kj = kj + 1)
                    exp_o[r*OFM+col] = exp_o[r*OFM+col] + ifm[(r+ki)*IFM + (col+kj)] * wgt[ki*K+kj];
            end

            for (i = 0; i < IFM*IFM; i = i + 1) core_write(IFM_BASE + i, {24'b0, ifm[i]});
            for (i = 0; i < K*K;     i = i + 1) core_write(W_BASE + i,   {24'b0, wgt[i]});
            for (i = 0; i < OFM*OFM + 16; i = i + 1) core_write(OFM_BASE + i, 32'hDEADBEEF);

            noise_en = noise; measuring = 1;
            @(negedge clk); start = 1;
            @(negedge clk); start = 0;
            cyc = 0;
            while (!done && cyc < 100000) begin @(negedge clk); cyc = cyc + 1; end
            if (!done) begin $display("TIMEOUT"); errors = errors + 1; end
            @(posedge clk); #1; measuring = 0; noise_en = 0; @(negedge clk); core_req = 0;   // one more edge so the done pulse is timestamped

            // (a) accumulators at compute_done: the 16 outputs straight out of the PEs
            for (i = 0; i < 16; i = i + 1) begin
                got = acc_snap[i*32 +: 32];
                if (got !== exp_o[i]) begin errors = errors + 1; $display("  ACC MISMATCH PE%0d got %0d exp %0d", i, got, exp_o[i]); end
            end
            // (b) OFMap actually stored in SRAM
            for (i = 0; i < 16; i = i + 1) begin
                core_read(OFM_BASE + i, rd);
                if (rd !== exp_o[i]) begin errors = errors + 1; $display("  OFMAP MISMATCH %0d got %0d exp %0d", i, $signed(rd), exp_o[i]); end
            end
            for (i = 0; i < 16; i = i + 1) begin
                core_read(OFM_BASE + 16 + i, rd);
                if (rd !== 32'hDEADBEEF) begin errors = errors + 1; $display("  GUARD overwritten at +%0d", i); end
            end
            // (c) the milestone itself: 9 MACs, back-to-back, no stalls, one done pulse
            if (mac_cycles !== 9)                          begin errors = errors + 1; $display("  FAIL mac_cycles = %0d (want 9)", mac_cycles); end
            if (t_last_mac - t_first_mac + 1 !== 9)        begin errors = errors + 1; $display("  FAIL MACs not contiguous (span %0d)", t_last_mac - t_first_mac + 1); end
            if (stall_cycles !== 0)                        begin errors = errors + 1; $display("  FAIL stalls = %0d", stall_cycles); end
            if (seen_cdone !== 1)                          begin errors = errors + 1; $display("  FAIL compute_done pulses = %0d", seen_cdone); end
            if (t_cdone - t_last_mac !== 1)                begin errors = errors + 1; $display("  FAIL compute_done latency %0d (want 1)", t_cdone - t_last_mac); end

            $display("%s  noise=%0d | 16/16 outputs checked | MAC cycles=%0d (contiguous), stalls=%0d",
                     errors == 0 ? "PASS" : "FAIL", noise, mac_cycles, stall_cycles);
            $display("      timeline (cycles): start->compute_start=%0d  compute_start->first MAC=%0d  MACs=%0d  last MAC->compute_done=%0d  compute_done->layer done=%0d  total=%0d",
                     t_cstart - t_start, t_first_mac - t_cstart, t_last_mac - t_first_mac + 1,
                     t_cdone - t_last_mac, t_done - t_cdone, t_done - t_start);
            total_errors = total_errors + errors;
        end
    endtask

    initial begin
        $dumpfile("sim/waveform/milestone1_wave.vcd");
        $dumpvars(0, milestone1_tb);
        #22 rst = 0;
        run(0);
        run(1);      // core hammering the bank: loads/stores retry, compute must still be 9 clean cycles
        if (total_errors == 0) $display("\nMILESTONE 1 PASSED: 4x4 tile, 3x3 kernel -> 16 correct outputs in 9 compute cycles");
        else                   $display("\nMILESTONE 1 FAILED: %0d errors", total_errors);
        $finish;
    end
endmodule