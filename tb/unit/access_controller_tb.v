`timescale 1ns/1ps
// Self-checking unit test for access_controller:
//   compute handshake + load / store grants + arbiter + access window
//   + psum reload (psum load vs pe_clear, psum RF free bits on the load channel)
module access_controller_tb;
    localparam N = 16, NM = 8, MAW = 4;
    reg clk = 0, rst = 1;  always #5 clk = ~clk;

    // ---- compute side ----
    reg compute_start = 0, seq_valid = 0, seq_last = 0;
    reg [N-1:0] pe_mask = {N{1'b1}}, ifv = {N{1'b1}}, psv = {N{1'b1}};
    reg wv = 1, psum_req = 0;
    wire seq_enable, compute_done, wrf_en, psum_ack, pe_clear, mac_en, psum_load, stall;
    wire [N-1:0] if_en, ps_en, ps_last;

    // ---- load / store side ----
    reg  [NM-1:0]     ld_req = 0, st_req = 0, st_we = {NM{1'b1}};
    reg  [NM-1:0]     ifree = {NM{1'b1}}, wfree = {NM{1'b1}}, psfree = {NM{1'b1}};
    reg               ld_is_w = 0, ld_is_psum = 0;
    reg  [NM*MAW-1:0] ld_addr = 0, st_addr = 0;
    reg  [NM-1:0]     core_hit = 0;
    wire [NM-1:0]     nmc_req, nmc_we, ld_grant, st_grant;
    wire [NM*MAW-1:0] nmc_addr;
    wire [NM-1:0]     nmc_grant = nmc_req & ~core_hit;       // bank model: core wins its macro
    reg  [15:0]       allow = 16, block = 0;
    wire win_ok, blk_cfg, blk_core, ld_busy_blk;

    access_controller #(.NUM_PE(N), .NUM_MACROS(NM), .MACRO_AW(MAW)) dut (
        .clk(clk), .rst(rst), .compute_start(compute_start), .seq_valid(seq_valid), .seq_last(seq_last),
        .pe_mask(pe_mask), .seq_enable(seq_enable), .compute_done(compute_done),
        .ifrf_read_valid(ifv), .wrf_read_valid(wv), .ifrf_read_en(if_en), .wrf_read_en(wrf_en),
        .psum_req(psum_req), .psrf_read_valid(psv), .psrf_read_en(ps_en), .psrf_read_last(ps_last),
        .psum_ack(psum_ack), .pe_clear(pe_clear), .pe_mac_en(mac_en), .pe_psum_load(psum_load), .stall(stall),
        .ld_req(ld_req), .ld_addr(ld_addr), .ld_is_weight(ld_is_w), .ld_is_psum(ld_is_psum),
        .ifrf_dst_free(ifree), .wrf_dst_free(wfree), .psrf_dst_free(psfree), .ld_grant(ld_grant),
        .st_req(st_req), .st_we(st_we), .st_addr(st_addr), .st_grant(st_grant),
        .nmc_req(nmc_req), .nmc_we(nmc_we), .nmc_addr(nmc_addr), .nmc_grant(nmc_grant),
        .win_allow_cycles(allow), .win_block_cycles(block),
        .mem_access_allowed(win_ok), .mem_blocked_cfg(blk_cfg),
        .mem_blocked_core(blk_core), .ld_blocked_busy(ld_busy_blk));

    integer errors = 0, k;
    reg [0:9] pat = 10'b1110011100;      // allow=3, block=2 -> O O O C C O O O C C

    task check(input [255:0] name, input integer got, input integer exp);
        if (got !== exp) begin $display("FAIL %0s: got %0h expected %0h", name, got, exp); errors = errors + 1; end
    endtask

    initial begin
        $dumpfile("sim/waveform/access_controller_wave.vcd");
        $dumpvars(0, access_controller_tb);
        for (k = 0; k < NM; k = k + 1) begin
            ld_addr[k*MAW +: MAW] = k + 1;      // load  lane k -> macro-local addr k+1
            st_addr[k*MAW +: MAW] = k + 8;      // store lane k -> macro-local addr k+8
        end
        #12 rst = 0; @(negedge clk);

        // ================= COMPUTE (original tests) =================
        // 1. everything valid -> MAC fires, all lanes consumed
        seq_valid = 1; #1;
        check("fire: mac_en", mac_en, 1);  check("fire: seq_enable", seq_enable, 1);
        check("fire: ifrf_read_en", if_en, 16'hFFFF); check("fire: wrf_read_en", wrf_en, 1);
        check("fire: stall", stall, 0);

        // 2. one active IFMap lane invalid -> stall, nothing consumed
        ifv = 16'hFFFF & ~(1 << 5); #1;
        check("stall lane: mac_en", mac_en, 0); check("stall lane: seq_enable", seq_enable, 0);
        check("stall lane: ifrf_read_en", if_en, 0); check("stall lane: wrf_read_en", wrf_en, 0);
        check("stall lane: stall", stall, 1);

        // 3. same lane but masked -> fires, masked lane not read
        pe_mask = 16'hFFFF & ~(1 << 5); #1;
        check("masked: mac_en", mac_en, 1);
        check("masked: ifrf_read_en", if_en, 16'hFFDF);

        // 4. weight invalid -> stall
        ifv = {N{1'b1}}; pe_mask = {N{1'b1}}; wv = 0; #1;
        check("weight stall: mac_en", mac_en, 0); check("weight stall: stall", stall, 1);
        wv = 1;

        // 5. no op pending
        seq_valid = 0; #1;
        check("idle: mac_en", mac_en, 0); check("idle: stall", stall, 0); check("idle: read_en", if_en, 0);

        // 6. compute_done: only after the LAST op actually fires, one cycle later
        seq_valid = 1; seq_last = 0; @(negedge clk);
        check("done not after non-last", compute_done, 0);
        seq_last = 1; ifv = 0; @(negedge clk);
        check("done not while stalled", compute_done, 0);
        ifv = {N{1'b1}}; #1; check("last fires", mac_en, 1);
        @(negedge clk);
        check("done 1 cycle after last MAC", compute_done, 1);
        seq_valid = 0; seq_last = 0; @(negedge clk);
        check("done is a 1-cycle pulse", compute_done, 0);

        // 7. pe_clear is compute_start delayed 1 cycle
        compute_start = 1; @(negedge clk); compute_start = 0;
        check("clear delayed", pe_clear, 1); @(negedge clk);
        check("clear 1-cycle", pe_clear, 0);

        // 8. psum reload
        seq_valid = 1; psum_req = 1; #1;
        check("psum: load", psum_load, 1); check("psum: ack", psum_ack, 1);
        check("psum: read_en", ps_en, 16'hFFFF); check("psum: read_last", ps_last, 16'hFFFF);
        check("psum: mac blocked", mac_en, 0);
        // 9. psum lane invalid & active -> wait
        psv = 16'hFFFF & ~(1 << 3); #1;
        check("psum wait: load", psum_load, 0); check("psum wait: read_en", ps_en, 0);
        // 10. that lane masked -> load proceeds
        pe_mask = 16'hFFFF & ~(1 << 3); #1;
        check("psum masked: load", psum_load, 1); check("psum masked: read_en", ps_en, 16'hFFF7);
        psum_req = 0; pe_mask = {N{1'b1}}; psv = {N{1'b1}}; #1;
        check("psum off: mac resumes", mac_en, 1); check("psum off: no load", psum_load, 0);
        seq_valid = 0;

        // ================= LOAD grant (V=0 per word) =================
        // 11. all destination words free -> every lane granted
        ld_req = 8'hFF; #1;
        check("ld free: nmc_req",  nmc_req, 8'hFF);
        check("ld free: ld_grant", ld_grant, 8'hFF);
        check("ld free: st_grant", st_grant, 0);
        check("ld free: nmc_we",   nmc_we, 0);
        check("ld free: busy flag", ld_busy_blk, 0);

        // 12. weight job looks at the WEIGHT RF free bits, not the IFMap ones
        ld_is_w = 1; wfree = 8'hF0; #1;
        check("w job: nmc_req",  nmc_req, 8'hF0);
        check("w job: ld_grant", ld_grant, 8'hF0);
        check("w job: busy flag", ld_busy_blk, 1);
        ld_is_w = 0; wfree = {NM{1'b1}};

        // 13. one IFMap word still live (V=1) -> only that lane is held back
        ifree = 8'hFB; #1;
        check("live word: nmc_req",  nmc_req, 8'hFB);
        check("live word: ld_grant", ld_grant, 8'hFB);
        check("live word: busy flag", ld_busy_blk, 1);
        ifree = {NM{1'b1}};

        // ================= STORE grant + arbiter =================
        // 14. store beats load on shared macros (lanes 2,3); addresses follow the winner
        ld_req = 8'h0F; st_req = 8'h3C; #1;
        check("arb: nmc_req",  nmc_req, 8'h3F);
        check("arb: st_grant", st_grant, 8'h3C);
        check("arb: ld_grant", ld_grant, 8'h03);
        check("arb: nmc_we",   nmc_we, 8'h3C);
        check("arb: lane0 = load addr",  nmc_addr[0*MAW +: MAW], 1);
        check("arb: lane1 = load addr",  nmc_addr[1*MAW +: MAW], 2);
        check("arb: lane2 = store addr", nmc_addr[2*MAW +: MAW], 10);
        check("arb: lane5 = store addr", nmc_addr[5*MAW +: MAW], 13);

        // 15. core takes macro 3 -> store lane 3 not granted, retried later; others unaffected
        core_hit = 8'h08; #1;
        check("core: st_grant", st_grant, 8'h34);
        check("core: ld_grant", ld_grant, 8'h03);
        check("core: blocked_core", blk_core, 1);
        core_hit = 0; ld_req = 0; st_req = 0;

        // ================= ACCESS WINDOW =================
        // 16. allow=3, block=2: requests only pass while the window is open
        @(negedge clk);
        ld_req = 8'hFF; allow = 3; block = 2; #1;
        for (k = 0; k < 10; k = k + 1) begin
            if (k != 0) begin @(negedge clk); #1; end
            check("win: open pattern", win_ok,  pat[k]);
            check("win: nmc_req",      nmc_req, pat[k] ? 8'hFF : 8'h00);
            check("win: blocked_cfg",  blk_cfg, !pat[k]);
        end
        // 17. block=0 -> always open again
        block = 0; @(negedge clk); #1;
        check("window off: open", win_ok, 1);
        check("window off: nmc_req", nmc_req, 8'hFF);

        // ================= PSUM RELOAD =================
        ld_req = 0; st_req = 0; seq_valid = 0; psum_req = 0; allow = 16; block = 0;
        @(negedge clk);

        // 18. psum load must not land in the pe_clear cycle (clear would win in the MAC)
        psum_req = 1; compute_start = 1; @(negedge clk); compute_start = 0; #1;
        check("clear cycle: pe_clear",  pe_clear,  1);
        check("clear cycle: no load",   psum_load, 0);
        check("clear cycle: no ack",    psum_ack,  0);
        check("clear cycle: no read_en", ps_en,    0);
        @(negedge clk); #1;
        check("after clear: pe_clear low", pe_clear, 0);
        check("after clear: psum_load", psum_load, 1);
        check("after clear: psum_ack",  psum_ack,  1);
        psum_req = 0;

        // 19. a psum reload on the load channel looks at the PSUM RF free bits only
        ld_req = 8'hFF; ld_is_psum = 1; ifree = 0; wfree = 0; psfree = 8'hF0; #1;
        check("ps job: nmc_req",   nmc_req,  8'hF0);
        check("ps job: ld_grant",  ld_grant, 8'hF0);
        check("ps job: nmc_we",    nmc_we,   0);
        check("ps job: busy flag", ld_busy_blk, 1);
        // all psum words free -> every lane granted even though IFMap/weight words are "live"
        psfree = {NM{1'b1}}; #1;
        check("ps job all free: nmc_req",  nmc_req,  8'hFF);
        check("ps job all free: ld_grant", ld_grant, 8'hFF);
        check("ps job all free: busy flag", ld_busy_blk, 0);
        // 20. a store still beats a psum reload on the same macro
        st_req = 8'h0C; #1;
        check("ps vs store: st_grant", st_grant, 8'h0C);
        check("ps vs store: ld_grant", ld_grant, 8'hF3);
        st_req = 0; ld_is_psum = 0; ld_req = 0;
        ifree = {NM{1'b1}}; wfree = {NM{1'b1}};

        if (errors == 0) $display("ALL TESTS PASSED"); else $display("%0d ERRORS", errors);
        $finish;
    end
endmodule