`timescale 1ns/1ps
// Self-checking test of store_reload_generator against sram_bank:
// store a 4x4 accumulator tile (full and partial, unaligned address, with core
// noise), check the OFMap words, then RELOAD and check the psum RF writes.
module store_reload_generator_tb;
    localparam NM = 8, DPM = 256, MAW = 8, CAW = 3 + MAW, LW = 4;

    reg clk = 0, rst = 1;  always #5 clk = ~clk;

    reg start = 0, reload = 0;
    reg [15:0] ofmap_cols = 10;
    reg [16:0] tile_addr = 21;
    reg [2:0]  vrows = 4, vcols = 4;
    wire [NM-1:0] nmc_req, nmc_we, nmc_grant, nmc_blocked, nmc_rvalid, rl_en;
    wire [NM*MAW-1:0] nmc_addr;
    wire [NM*LW-1:0]  lane_sel, rl_lane;
    wire [NM*32-1:0]  nmc_wdata, nmc_rdata;
    wire busy, done;

    reg  core_req = 0, core_we = 0; reg [CAW-1:0] core_addr = 0; reg [31:0] core_wdata = 0;
    wire [31:0] core_rdata; wire core_rvalid;

    sram_bank #(.DATA_WIDTH(32), .NUM_MACROS(NM), .DEPTH_PER_MACRO(DPM)) bank (
        .clk(clk), .rst(rst), .core_req(core_req), .core_we(core_we), .core_addr(core_addr),
        .core_wdata(core_wdata), .core_rdata(core_rdata), .core_rvalid(core_rvalid),
        .nmc_req(nmc_req), .nmc_we(nmc_we), .nmc_addr(nmc_addr), .nmc_wdata(nmc_wdata),
        .nmc_grant(nmc_grant), .nmc_blocked(nmc_blocked), .nmc_rdata(nmc_rdata), .nmc_rvalid(nmc_rvalid));

    store_reload_generator #(.ADDR_WIDTH(17), .DIM_WIDTH(16), .NUM_MACROS(NM), .MACRO_AW(MAW)) dut (
        .clk(clk), .rst(rst), .start(start), .reload(reload),
        .ofmap_cols(ofmap_cols), .ofmap_tile_addr(tile_addr), .valid_rows(vrows[2:0]), .valid_cols(vcols[2:0]),
        .nmc_req(nmc_req), .nmc_we(nmc_we), .nmc_addr(nmc_addr), .nmc_grant(nmc_grant),
        .st_lane_sel(lane_sel), .rl_wr_en(rl_en), .rl_wr_lane(rl_lane), .busy(busy), .done(done));

    // accumulator model: lane i = 1000 + 7*i
    reg [16*32-1:0] acc;
    integer i;
    initial for (i = 0; i < 16; i = i + 1) acc[i*32 +: 32] = 1000 + 7*i;

    genvar g;
    generate for (g = 0; g < NM; g = g + 1) begin : g_w
        wire [LW-1:0] sel = lane_sel[g*LW +: LW];
        assign nmc_wdata[g*32 +: 32] = acc[sel*32 +: 32];
    end endgenerate

    // psum RF capture
    reg [31:0] psum [0:15];
    reg [15:0] psum_set;
    integer m;
    always @(posedge clk) for (m = 0; m < NM; m = m + 1)
        if (rl_en[m]) begin
            psum[rl_lane[m*LW +: LW]] <= nmc_rdata[m*32 +: 32];
            psum_set[rl_lane[m*LW +: LW]] <= 1'b1;
        end

    reg noise = 0;
    always @(negedge clk) if (noise) begin core_req = $random & 1; core_we = 0; core_addr = $random; end

    integer errors = 0;
    reg [31:0] rd;
    task core_write(input [CAW-1:0] a, input [31:0] d);
        begin @(negedge clk); core_req=1; core_we=1; core_addr=a; core_wdata=d;
              @(negedge clk); core_req=0; core_we=0; end
    endtask
    task core_read(input [CAW-1:0] a, output [31:0] d);
        begin @(negedge clk); core_req=1; core_we=0; core_addr=a;
              @(negedge clk); core_req=0; d=core_rdata; end
    endtask
    task run(input r);
        integer t;
        begin
            @(negedge clk); reload = r; start = 1;
            @(negedge clk); start = 0;
            t = 0; while (!done && t < 10000) begin @(negedge clk); t = t + 1; end
            if (!done) begin $display("TIMEOUT"); errors = errors + 1; end
            @(negedge clk);
        end
    endtask

    integer r, c, k, w;
    task one_case(input integer ta, input integer vr, input integer vc, input integer nz);
        begin
            tile_addr = ta; vrows = vr; vcols = vc;
            noise = 0; core_req = 0;
            for (i = 0; i < 80; i = i + 1) core_write(ta + i, 32'hDEADBEEF); // prefill window
            noise = nz;
            run(0);                       // STORE
            noise = 0; @(negedge clk); core_req = 0;
            for (r = 0; r < 4; r = r + 1) for (c = 0; c < 4; c = c + 1) begin
                core_read(ta + r*10 + c, rd);
                if (r < vr && c < vc) begin
                    if (rd !== 1000 + 7*(r*4 + c)) begin errors = errors + 1; $display("  store mismatch r%0d c%0d got %0d", r, c, rd); end
                end else if (rd !== 32'hDEADBEEF) begin errors = errors + 1; $display("  store wrote masked r%0d c%0d", r, c); end
            end
            psum_set = 0; noise = nz;
            run(1);                       // RELOAD
            noise = 0; @(negedge clk); core_req = 0;
            for (r = 0; r < 4; r = r + 1) for (c = 0; c < 4; c = c + 1) begin
                k = r*4 + c;
                if (r < vr && c < vc) begin
                    if (!psum_set[k] || psum[k] !== 1000 + 7*k) begin errors = errors + 1; $display("  reload mismatch lane %0d", k); end
                end else if (psum_set[k]) begin errors = errors + 1; $display("  reload touched masked lane %0d", k); end
            end
            $display("tile_addr=%0d vrows=%0d vcols=%0d noise=%0d done", ta, vr, vc, nz);
        end
    endtask

    initial begin
        $dumpfile("sim/waveform/store_reload_generator_wave.vcd");
        $dumpvars(0, store_reload_generator_tb);
        #22 rst = 0;
        one_case(21, 4, 4, 0);
        one_case(21, 4, 4, 1);
        one_case(30, 3, 2, 1);
        one_case(24, 1, 1, 0);
        one_case(27, 2, 4, 1);
        if (errors == 0) $display("ALL TESTS PASSED"); else $display("%0d ERRORS", errors);
        $finish;
    end
endmodule