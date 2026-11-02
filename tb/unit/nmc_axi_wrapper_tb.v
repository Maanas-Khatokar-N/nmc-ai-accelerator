`timescale 1ns/1ps
// ============================================================================
// Self-checking test of nmc_axi_wrapper (real nmc_top behind it), AXI master model in tasks.
// Checks: register read/write over AXI4-Lite, SLVERR on RO write, wrapper ID, soft reset,
// irq + W1C, SRAM single access, INCR burst write/read (rlast), partial-strobe RMW,
// FIXED burst, WRAP -> SLVERR, access to a different bank, cfg access while a burst runs.
//
//   iverilog -g2012 -o sim/build/nmc_axi_wrapper.vvp tb/unit/nmc_axi_wrapper_tb.v \
//            $(find rtl -name '*.v') && vvp sim/build/nmc_axi_wrapper.vvp
// ============================================================================
module nmc_axi_wrapper_tb;
    localparam IDW = 16, MAW = 19;

    reg clk = 0, aresetn = 0;
    always #5 clk = ~clk;

    // cfg
    reg  [11:0] c_awaddr = 0, c_araddr = 0;  reg c_awvalid = 0, c_wvalid = 0, c_arvalid = 0;
    reg  [31:0] c_wdata = 0;                 reg c_bready = 1, c_rready = 1;
    wire        c_awready, c_wready, c_arready, c_bvalid, c_rvalid;
    wire [1:0]  c_bresp, c_rresp;  wire [31:0] c_rdata;
    // mem
    reg  [IDW-1:0] m_awid = 0, m_arid = 0;
    reg  [MAW-1:0] m_awaddr = 0, m_araddr = 0;
    reg  [7:0]  m_awlen = 0, m_arlen = 0;
    reg  [2:0]  m_awsize = 3'd2, m_arsize = 3'd2;
    reg  [1:0]  m_awburst = 2'b01, m_arburst = 2'b01;
    reg         m_awvalid = 0, m_wvalid = 0, m_wlast = 0, m_arvalid = 0, m_bready = 1, m_rready = 1;
    reg  [31:0] m_wdata = 0;  reg [3:0] m_wstrb = 4'hF;
    wire        m_awready, m_wready, m_arready, m_bvalid, m_rvalid, m_rlast;
    wire [IDW-1:0] m_bid, m_rid;
    wire [1:0]  m_bresp, m_rresp;  wire [31:0] m_rdata;
    wire        irq;

    nmc_axi_wrapper #(.ID_W(IDW), .MEM_AW(MAW)) dut (
        .aclk(clk), .aresetn(aresetn), .irq(irq),
        .s_axi_cfg_awaddr(c_awaddr), .s_axi_cfg_awvalid(c_awvalid), .s_axi_cfg_awready(c_awready),
        .s_axi_cfg_wdata(c_wdata), .s_axi_cfg_wstrb(4'hF), .s_axi_cfg_wvalid(c_wvalid), .s_axi_cfg_wready(c_wready),
        .s_axi_cfg_bresp(c_bresp), .s_axi_cfg_bvalid(c_bvalid), .s_axi_cfg_bready(c_bready),
        .s_axi_cfg_araddr(c_araddr), .s_axi_cfg_arvalid(c_arvalid), .s_axi_cfg_arready(c_arready),
        .s_axi_cfg_rdata(c_rdata), .s_axi_cfg_rresp(c_rresp), .s_axi_cfg_rvalid(c_rvalid), .s_axi_cfg_rready(c_rready),
        .s_axi_mem_awid(m_awid), .s_axi_mem_awaddr(m_awaddr), .s_axi_mem_awlen(m_awlen),
        .s_axi_mem_awsize(m_awsize), .s_axi_mem_awburst(m_awburst),
        .s_axi_mem_awvalid(m_awvalid), .s_axi_mem_awready(m_awready),
        .s_axi_mem_wdata(m_wdata), .s_axi_mem_wstrb(m_wstrb), .s_axi_mem_wlast(m_wlast),
        .s_axi_mem_wvalid(m_wvalid), .s_axi_mem_wready(m_wready),
        .s_axi_mem_bid(m_bid), .s_axi_mem_bresp(m_bresp), .s_axi_mem_bvalid(m_bvalid), .s_axi_mem_bready(m_bready),
        .s_axi_mem_arid(m_arid), .s_axi_mem_araddr(m_araddr), .s_axi_mem_arlen(m_arlen),
        .s_axi_mem_arsize(m_arsize), .s_axi_mem_arburst(m_arburst),
        .s_axi_mem_arvalid(m_arvalid), .s_axi_mem_arready(m_arready),
        .s_axi_mem_rid(m_rid), .s_axi_mem_rdata(m_rdata), .s_axi_mem_rresp(m_rresp),
        .s_axi_mem_rlast(m_rlast), .s_axi_mem_rvalid(m_rvalid), .s_axi_mem_rready(m_rready));

    integer errors = 0;
    task check(input [255:0] name, input [63:0] got, input [63:0] exp);
        if (got !== exp) begin $display("FAIL %0s: got %0h expected %0h", name, got, exp); errors = errors + 1; end
    endtask

    // ---- AXI4-Lite master. Inputs change at negedge, handshakes are sampled 1 ns before posedge ----
    task cfg_write(input [11:0] a, input [31:0] d, output [1:0] resp);
        reg aw_d, w_d, done;
        begin
            @(negedge clk);
            c_awaddr = a; c_awvalid = 1; c_wdata = d; c_wvalid = 1;
            aw_d = 0; w_d = 0; done = 0;
            while (!(aw_d && w_d)) begin
                #4;
                if (c_awvalid && c_awready) aw_d = 1;
                if (c_wvalid  && c_wready)  w_d  = 1;
                @(negedge clk);
                if (aw_d) c_awvalid = 0;
                if (w_d)  c_wvalid  = 0;
            end
            while (!done) begin
                #4;
                if (c_bvalid && c_bready) begin resp = c_bresp; done = 1; end
                @(negedge clk);
            end
        end
    endtask

    task cfg_read(input [11:0] a, output [31:0] d, output [1:0] resp);
        reg done;
        begin
            @(negedge clk);
            c_araddr = a; c_arvalid = 1; done = 0;
            while (!done) begin
                #4;
                if (c_arvalid && c_arready) done = 1;
                @(negedge clk);
            end
            c_arvalid = 0; done = 0;
            while (!done) begin
                #4;
                if (c_rvalid && c_rready) begin d = c_rdata; resp = c_rresp; done = 1; end
                @(negedge clk);
            end
        end
    endtask

    // ---- AXI4 master ----
    reg [31:0] wbuf [0:255];
    reg [31:0] rbuf [0:255];
    integer    rbeats;
    reg        rlast_seen_early;

    task mem_write(input [MAW-1:0] a, input integer beats, input [3:0] strb, input [1:0] burst, output [1:0] resp);
        reg done;
        integer i;
        begin
            @(negedge clk);
            m_awid = 16'h1A5; m_awaddr = a; m_awlen = beats - 1; m_awburst = burst; m_awvalid = 1;
            done = 0;
            while (!done) begin #4; if (m_awready) done = 1; @(negedge clk); end
            m_awvalid = 0;
            for (i = 0; i < beats; i = i + 1) begin
                m_wdata = wbuf[i]; m_wstrb = strb; m_wlast = (i == beats - 1); m_wvalid = 1;
                done = 0;
                while (!done) begin #4; if (m_wready) done = 1; @(negedge clk); end
            end
            m_wvalid = 0; m_wlast = 0; done = 0;
            while (!done) begin
                #4;
                if (m_bvalid && m_bready) begin resp = m_bresp; check("bid", m_bid, 16'h1A5); done = 1; end
                @(negedge clk);
            end
        end
    endtask

    task mem_read(input [MAW-1:0] a, input integer beats, input [1:0] burst, output [1:0] resp);
        reg done;
        begin
            @(negedge clk);
            m_arid = 16'h2B6; m_araddr = a; m_arlen = beats - 1; m_arburst = burst; m_arvalid = 1;
            done = 0;
            while (!done) begin #4; if (m_arready) done = 1; @(negedge clk); end
            m_arvalid = 0;
            rbeats = 0; done = 0; rlast_seen_early = 0;
            while (!done) begin
                #4;
                if (m_rvalid && m_rready) begin
                    rbuf[rbeats] = m_rdata; resp = m_rresp;
                    check("rid", m_rid, 16'h2B6);
                    if (m_rlast !== (rbeats == beats - 1)) begin
                        $display("FAIL rlast wrong at beat %0d", rbeats); errors = errors + 1;
                    end
                    rbeats = rbeats + 1;
                    if (m_rlast) done = 1;
                end
                @(negedge clk);
            end
            check("read beats", rbeats, beats);
        end
    endtask

    reg [31:0] rd;
    reg [1:0]  rsp, rsp2;
    reg [31:0] rd2;
    integer i;

    initial begin
        $dumpfile("sim/waveform/nmc_axi_wrapper_wave.vcd");
        $dumpvars(0, nmc_axi_wrapper_tb);
        repeat (5) @(negedge clk);
        aresetn = 1;
        repeat (5) @(negedge clk);

        // ---------------- registers over AXI4-Lite ----------------
        cfg_read(12'h000, rd, rsp);  check("nmc ID", rd, 32'h4E4D0001);  check("ID resp", rsp, 0);
        cfg_read(12'hF00, rd, rsp);  check("wrapper ID", rd, 32'h41584901);
        cfg_write(12'h008, 32'h5, rsp);   check("UNIT_MASK write resp", rsp, 0);
        cfg_read (12'h008, rd, rsp);      check("UNIT_MASK readback", rd, 5);
        cfg_write(12'h000, 32'h1, rsp);   check("write RO ID -> SLVERR", rsp, 2);
        cfg_read (12'h500, rd, rsp);      check("unmapped read -> SLVERR", rsp, 2);
        cfg_read (12'hF08, rd, rsp);      check("wrapper reserved -> SLVERR", rsp, 2);

        // ---------------- interrupt: start with empty PP mask is rejected -> IRQ_STATUS[2] ----------------
        cfg_write(12'h024, 32'h4, rsp);   // IRQ_ENABLE: START_REJECTED
        check("irq low", irq, 0);
        cfg_write(12'h00C, 32'h0, rsp);   // PP_MASK = 0
        cfg_write(12'h004, 32'h2, rsp);   // PP_START with empty mask
        repeat (3) @(negedge clk);
        check("irq high", irq, 1);
        cfg_read (12'h020, rd, rsp);      check("IRQ_STATUS", rd, 4);
        cfg_write(12'h020, 32'h4, rsp);   // W1C
        repeat (2) @(negedge clk);
        check("irq cleared", irq, 0);

        // ---------------- soft reset ----------------
        cfg_write(12'hF04, 32'h1, rsp);   check("soft reset write", rsp, 0);
        repeat (30) @(negedge clk);
        cfg_read (12'hF04, rd, rsp);      check("soft reset done", rd, 0);
        cfg_read (12'h008, rd, rsp);      check("UNIT_MASK cleared by soft reset", rd, 0);
        cfg_read (12'h000, rd, rsp);      check("ID after soft reset", rd, 32'h4E4D0001);

        // ---------------- SRAM: single word ----------------
        wbuf[0] = 32'hCAFEF00D;
        mem_write(19'h00000, 1, 4'hF, 2'b01, rsp);  check("wr resp", rsp, 0);
        mem_read (19'h00000, 1, 2'b01, rsp);        check("rd resp", rsp, 0);
        check("word 0", rbuf[0], 32'hCAFEF00D);

        // ---------------- SRAM: INCR burst of 16 words crossing macros ----------------
        for (i = 0; i < 16; i = i + 1) wbuf[i] = 32'h1000_0000 + i * 32'h0101;
        mem_write(19'h00100, 16, 4'hF, 2'b01, rsp);
        mem_read (19'h00100, 16, 2'b01, rsp);
        for (i = 0; i < 16; i = i + 1) check("burst word", rbuf[i], 32'h1000_0000 + i * 32'h0101);
        // partial overlapping read (starts mid burst)
        mem_read (19'h00108, 4, 2'b01, rsp);
        for (i = 0; i < 4; i = i + 1) check("burst sub-read", rbuf[i], 32'h1000_0000 + (i + 2) * 32'h0101);

        // ---------------- SRAM: bank 1 (bit 17) and bank 3, 256-beat burst ----------------
        wbuf[0] = 32'hB1B1B1B1;  mem_write(19'h20000, 1, 4'hF, 2'b01, rsp);
        wbuf[0] = 32'hB3B3B3B3;  mem_write(19'h60040, 1, 4'hF, 2'b01, rsp);
        mem_read (19'h20000, 1, 2'b01, rsp);  check("bank1 word", rbuf[0], 32'hB1B1B1B1);
        mem_read (19'h60040, 1, 2'b01, rsp);  check("bank3 word", rbuf[0], 32'hB3B3B3B3);
        mem_read (19'h00000, 1, 2'b01, rsp);  check("bank0 untouched", rbuf[0], 32'hCAFEF00D);
        for (i = 0; i < 256; i = i + 1) wbuf[i] = i * 32'h01010101;
        mem_write(19'h40000, 256, 4'hF, 2'b01, rsp);
        mem_read (19'h40000, 256, 2'b01, rsp);
        for (i = 0; i < 256; i = i + 1) if (rbuf[i] !== i * 32'h01010101) begin
            $display("FAIL 256-beat word %0d got %0h", i, rbuf[i]); errors = errors + 1; end

        // ---------------- partial strobes: read-modify-write ----------------
        wbuf[0] = 32'hAABBCCDD;  mem_write(19'h00200, 1, 4'hF, 2'b01, rsp);
        wbuf[0] = 32'h11223344;  mem_write(19'h00200, 1, 4'b0101, 2'b01, rsp);
        mem_read (19'h00200, 1, 2'b01, rsp);  check("RMW lanes 0,2", rbuf[0], 32'hAA22CC44);
        wbuf[0] = 32'h55667788;  mem_write(19'h00204, 1, 4'hF, 2'b01, rsp);      // neighbour word
        wbuf[0] = 32'h000000EE;  mem_write(19'h00204, 1, 4'b0001, 2'b01, rsp);   // int8 into the low byte only
        wbuf[0] = 32'h0000EE00;  mem_write(19'h00201, 1, 4'b0010, 2'b01, rsp);   // byte store at 0x201 -> lane 1 of word 0x200
        mem_read (19'h00200, 2, 2'b01, rsp);
        check("RMW lane1 byte store", rbuf[0], 32'hAA22EE44);
        check("RMW low byte of neighbour", rbuf[1], 32'h556677EE);
        wbuf[0] = 32'hAA22CC44;  mem_write(19'h00200, 1, 4'hF, 2'b01, rsp);      // restore
        wbuf[0] = 32'hFFFFFFFF;  mem_write(19'h00200, 1, 4'b0000, 2'b01, rsp);   // strobe 0 -> no write
        mem_read (19'h00200, 1, 2'b01, rsp);  check("strobe 0 no-op", rbuf[0], 32'hAA22CC44);

        // ---------------- FIXED burst: all beats hit the same word, last wins ----------------
        wbuf[0] = 32'h0000_0001; wbuf[1] = 32'h0000_0002; wbuf[2] = 32'h0000_0003;
        mem_write(19'h00300, 3, 4'hF, 2'b00, rsp);
        mem_read (19'h00300, 1, 2'b01, rsp);  check("FIXED burst last wins", rbuf[0], 3);

        // ---------------- WRAP burst: completes but SLVERR ----------------
        wbuf[0] = 1; wbuf[1] = 2;
        mem_write(19'h00400, 2, 4'hF, 2'b10, rsp);  check("WRAP write -> SLVERR", rsp, 2);

        // ---------------- cfg access in the middle of a long read burst ----------------
        fork
            mem_read(19'h40000, 64, 2'b01, rsp);
            begin
                repeat (40) @(negedge clk);
                cfg_read(12'h000, rd2, rsp2);   check("cfg read during burst", rd2, 32'h4E4D0001);
                cfg_write(12'h008, 32'h3, rsp2); check("cfg write during burst", rsp2, 0);
                cfg_read(12'h008, rd2, rsp2);   check("cfg readback during burst", rd2, 3);
            end
        join
        for (i = 0; i < 64; i = i + 1) check("burst data intact", rbuf[i], i * 32'h01010101);

        if (errors == 0) $display("ALL TESTS PASSED"); else $display("%0d ERRORS", errors);
        $finish;
    end

    initial begin
        #5000000;
        $display("GLOBAL TIMEOUT"); $finish;
    end
endmodule