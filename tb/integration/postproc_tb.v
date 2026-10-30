`timescale 1ns/1ps
// Whole-layer post-processing testbench driven by scripts/postproc_golden.py
//   reads  sim/data/pp_src.hex  (int32 OFMap words), sim/data/pp_qp.hex (3 lines per channel: bias, mult, shift)
//   writes sim/data/pp_rtl.hex  (dst words)
// Plusargs: +ROWS +COLS +CH +POOL +RELU +ZP +NOISE [+VCD]
module postproc_tb;
    localparam ADDR_WIDTH = 17, DIM_WIDTH = 16, NM = 8, DEPTH_PM = 4096;
    localparam MAW = $clog2(DEPTH_PM), CAW = 3 + MAW;
    localparam SRC_BASE = 0, DST_BASE = 20000;

    reg clk = 0, rst = 1;  always #5 clk = ~clk;

    reg start = 0;  wire done, busy;
    reg [DIM_WIDTH-1:0]  d_cols = 0, d_orows = 0, d_ocols = 0, d_ch = 0, d_pk = 1;
    reg [ADDR_WIDTH-1:0] d_chsz = 0, d_rstep = 0, d_total = 0;
    reg signed [7:0]     d_zp = 0;  reg d_relu = 0;
    reg                  qp_we = 0;
    reg [8:0]            qp_wa = 0;
    reg signed [31:0]    qp_b = 0;  reg [15:0] qp_m = 0;  reg [4:0] qp_s = 0;

    wire [NM-1:0]      nmc_req, nmc_we, nmc_grant, nmc_blocked, nmc_rvalid;
    wire [NM*MAW-1:0]  nmc_addr;
    wire [NM*32-1:0]   nmc_wdata, nmc_rdata;

    postproc_unit #(.ADDR_WIDTH(ADDR_WIDTH), .DIM_WIDTH(DIM_WIDTH), .NUM_MACROS(NM), .MACRO_AW(MAW)) dut (
        .clk(clk), .rst(rst), .start(start), .done(done), .busy(busy), .mem_allow(1'b1),
        .src_base(SRC_BASE[ADDR_WIDTH-1:0]), .dst_base(DST_BASE[ADDR_WIDTH-1:0]),
        .src_cols(d_cols), .out_rows(d_orows), .out_cols(d_ocols), .channels(d_ch), .pool_k(d_pk),
        .src_ch_size(d_chsz), .row_step(d_rstep), .total_out(d_total), .out_zp(d_zp), .relu_en(d_relu),
        .qp_we(qp_we), .qp_waddr(qp_wa), .qp_bias(qp_b), .qp_mult(qp_m), .qp_shift(qp_s),
        .nmc_req(nmc_req), .nmc_we(nmc_we), .nmc_addr(nmc_addr), .nmc_wdata(nmc_wdata),
        .nmc_grant(nmc_grant), .nmc_rdata(nmc_rdata));

    reg                core_req = 0, core_we = 0;
    reg  [CAW-1:0]     core_addr = 0;
    reg  [31:0]        core_wdata = 0;
    wire [31:0]        core_rdata;
    wire               core_rvalid;

    sram_bank #(.DATA_WIDTH(32), .NUM_MACROS(NM), .DEPTH_PER_MACRO(DEPTH_PM)) bank (
        .clk(clk), .rst(rst), .core_req(core_req), .core_we(core_we), .core_addr(core_addr),
        .core_wdata(core_wdata), .core_rdata(core_rdata), .core_rvalid(core_rvalid),
        .nmc_req(nmc_req), .nmc_we(nmc_we), .nmc_addr(nmc_addr), .nmc_wdata(nmc_wdata),
        .nmc_grant(nmc_grant), .nmc_blocked(nmc_blocked), .nmc_rdata(nmc_rdata), .nmc_rvalid(nmc_rvalid));

    reg noise_en = 0;
    always @(negedge clk) if (noise_en) begin
        core_req = $random & 1;  core_we = 1'b0;  core_addr = $random;
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

    reg [31:0] src_mem [0:16383];
    reg [31:0] qpm [0:3*512-1];
    integer rows, cols, ch, pool, relu, zp, noise, orows, ocols, n_src, n_dst, i, cyc, fd;
    reg [31:0] rd;
    reg timed_out;

    initial begin
        if (!$value$plusargs("ROWS=%d",  rows))  rows  = 8;
        if (!$value$plusargs("COLS=%d",  cols))  cols  = 8;
        if (!$value$plusargs("CH=%d",    ch))    ch    = 2;
        if (!$value$plusargs("POOL=%d",  pool))  pool  = 2;
        if (!$value$plusargs("RELU=%d",  relu))  relu  = 1;
        if (!$value$plusargs("ZP=%d",    zp))    zp    = 0;
        if (!$value$plusargs("NOISE=%d", noise)) noise = 0;
        if ($test$plusargs("VCD")) begin $dumpfile("sim/waveform/postproc_wave.vcd"); $dumpvars(0, postproc_tb); end

        orows = rows / pool;  ocols = cols / pool;
        n_src = ch*rows*cols; n_dst = ch*orows*ocols;
        $readmemh("sim/data/pp_src.hex", src_mem, 0, n_src-1);
        $readmemh("sim/data/pp_qp.hex",  qpm,     0, 3*ch-1);

        d_cols = cols; d_orows = orows; d_ocols = ocols; d_ch = ch; d_pk = pool;
        d_chsz = rows*cols; d_rstep = pool*cols; d_total = n_dst; d_zp = zp; d_relu = relu;

        #22 rst = 0;
        for (i = 0; i < n_src;  i = i + 1) core_write(SRC_BASE + i, src_mem[i]);
        for (i = 0; i < n_dst + 16; i = i + 1) core_write(DST_BASE + i, 32'hDEADBEEF);

        for (i = 0; i < ch; i = i + 1) begin
            @(negedge clk); qp_we = 1; qp_wa = i; qp_b = qpm[3*i]; qp_m = qpm[3*i+1]; qp_s = qpm[3*i+2];
        end
        @(negedge clk); qp_we = 0;

        noise_en = noise;
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        cyc = 0;
        while (!done && cyc < 2000000) begin @(negedge clk); cyc = cyc + 1; end
        timed_out = !done;
        noise_en = 0; @(negedge clk); core_req = 0; core_we = 0;
        if (timed_out) begin $display("TIMEOUT"); $finish; end

        fd = $fopen("sim/data/pp_rtl.hex", "w");
        for (i = 0; i < n_dst + 16; i = i + 1) begin
            core_read(DST_BASE + i, rd);
            $fdisplay(fd, "%08x", rd);
        end
        $fclose(fd);
        $display("CYCLES=%0d OUTPUTS=%0d", cyc, n_dst);
        $finish;
    end
endmodule