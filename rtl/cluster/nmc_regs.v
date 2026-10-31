// ============================================================================
// NMC register interface (bus slave for nmc_cluster)
//
//   host / RISC-V --reg bus--> nmc_regs --> nmc_cluster (descriptors, masks, start,
//                                                        requant params)
//                              nmc_regs <-- nmc_cluster (irq, busy, perf counters)
//
// Output port names match the nmc_cluster input ports (and input port names
// match its outputs), so hook-up is a plain name-for-name connection.
// The cluster's host DATA port (core_*, SRAM access) is NOT part of this module.
//
// ----------------------------------------------------------------------------
// Bus (single cycle, no wait states, 32-bit word access only)
//   reg_req     request valid (read or write), one transfer per cycle
//   reg_we      1 = write, 0 = read
//   reg_addr    BYTE address; bits [1:0] ignored
//   reg_wdata   write data
//   reg_rvalid  1 cycle after every request
//   reg_rdata   read data (valid with reg_rvalid; 0 for writes)
//   reg_err     valid with reg_rvalid: unmapped address, write to a read-only
//               or reserved register
//
// Address map  (BUS_AW = 12 -> 4 KB window)
//   0x000 - 0x0FF  global block
//   0x100 + 0x100*b  bank b block (b = 0 .. NUM_BANKS-1, max 15 banks)
//
// Global block
//   0x00 ID           RO   0x4E4D0001
//   0x04 CTRL         WO   [0] CONV_START  [1] PP_START   (self-clearing, read 0)
//   0x08 UNIT_MASK    RW   conv units taking part in the round
//   0x0C PP_MASK      RW   post-proc units taking part in the round
//   0x10 STATUS       RO   [0] CONV_ACTIVE (round in flight)  [1] PP_ACTIVE
//   0x14 UNIT_BUSY    RO   per bank, from cluster unit_busy
//   0x18 PP_BUSY      RO   per bank, from cluster pp_busy
//   0x1C PERF_VALID   RO   per bank, counters hold a complete layer
//   0x20 IRQ_STATUS   RW1C [0] CONV_DONE  [1] PP_DONE
//                          [2] START_REJECTED  [3] CFG_WRITE_BLOCKED
//   0x24 IRQ_ENABLE   RW   same bit layout; irq_o = |(IRQ_STATUS & IRQ_ENABLE)
//   0x28 CONFIG       RO   [7:0] NUM_BANKS  [15:8] CNT_W  [31:16] MAX_OC
//   0x40 QP_BANK_MASK RW   banks that receive the next requant commit
//   0x44 QP_ADDR      RW   channel index (auto-increments after each commit)
//   0x48 QP_BIAS      RW   signed 32
//   0x4C QP_MULT      RW   unsigned 16
//   0x50 QP_SHIFT     RW   [4:0]. WRITING THIS REGISTER COMMITS {bias,mult,shift}
//                          to channel QP_ADDR of every bank in QP_BANK_MASK
//
// Bank block (offsets relative to 0x100 + 0x100*b)
//   conv descriptor (RW)             post-proc descriptor (RW)
//   0x00 IFMAP_COLS                  0x50 PP_SRC_BASE
//   0x04 KERNEL_ROWS                 0x54 PP_DST_BASE
//   0x08 KERNEL_COLS                 0x58 PP_SRC_COLS
//   0x0C INPUT_CHANNELS              0x5C PP_OUT_ROWS
//   0x10 OUTPUT_CHANNELS             0x60 PP_OUT_COLS
//   0x14 STRIDE                      0x64 PP_CHANNELS
//   0x18 OFMAP_ROWS                  0x68 PP_POOL_K
//   0x1C OFMAP_COLS                  0x6C PP_SRC_CH_SIZE
//   0x20 WEIGHTS_PER_OC              0x70 PP_ROW_STEP
//   0x24 IFMAP_CH_SIZE               0x74 PP_TOTAL_OUT
//   0x28 OFMAP_CH_SIZE               0x78 PP_OUT_ZP   [7:0]
//   0x2C IFMAP_ROW_STEP              0x7C PP_RELU_EN  [0]
//   0x30 IFMAP_BASE
//   0x34 WEIGHT_BASE
//   0x38 OFMAP_BASE
//   0x3C RF_ROW_STEP
//   0x40 RF_CH_STRIDE
//   0x44 ACCUMULATE  [0]
//   perf counters (RO, 48-bit as LO/HI word pairs, hold after done)
//   0x80 TOTAL_LO   0x84 TOTAL_HI
//   0x88 MAC_LO     0x8C MAC_HI
//   0x90 STALL_LO   0x94 STALL_HI
//   0x98 BLOCKED_LO 0x9C BLOCKED_HI
//   0xA0 MAC_OPS_LO 0xA4 MAC_OPS_HI
//
// Field widths follow the cluster ports (DIM_WIDTH / ADDR_WIDTH / RF_AW): unused
// upper bits are not stored and read back as 0.
//
// Behaviour
//  - START is accepted only if no round is in flight and the mask is non-zero.
//    Otherwise it is dropped and IRQ_STATUS[2] is set. CONV_START and PP_START in
//    the same write: conv wins, PP is rejected. ("In flight" = from START until
//    the cluster irq / pp_irq pulse; the cluster's own unit_busy drops earlier.)
//  - The cluster needs descriptors / masks static while a round runs, so writes
//    are blocked while that round is in flight (sticky IRQ_STATUS[3], no reg_err):
//      conv words, ACCUMULATE, UNIT_MASK   blocked while CONV_ACTIVE
//      PP words, PP_MASK, QP_*             blocked while PP_ACTIVE
//    Everything else may be written at any time, e.g. requant parameters for the
//    next layer can be loaded while a conv round runs.
//  - Perf counters are cleared by the cluster at the next start and hold after
//    done. Read them after the irq (or when PERF_VALID is set) to avoid a LO/HI
//    tear.
// ============================================================================
module nmc_regs #(
    parameter NUM_BANKS  = 4,
    parameter ADDR_WIDTH = 17,
    parameter DIM_WIDTH  = 16,
    parameter RF_AW      = 8,
    parameter CNT_W      = 48,                 // <= 64
    parameter MAX_OC     = 512,
    parameter QA_W       = $clog2(MAX_OC),
    parameter BUS_AW     = 12                  // 4 KB window, banks 0..14
)(
    input  wire                         clk,
    input  wire                         rst,

    // ---------------- register bus ----------------
    input  wire                         reg_req,
    input  wire                         reg_we,
    input  wire [BUS_AW-1:0]            reg_addr,
    input  wire [31:0]                  reg_wdata,
    output reg                          reg_rvalid,
    output reg  [31:0]                  reg_rdata,
    output reg                          reg_err,
    output wire                         irq_o,

    // ---------------- conv control (-> nmc_cluster) ----------------
    output wire [NUM_BANKS-1:0]         unit_mask,
    output wire                         start,
    output wire [NUM_BANKS-1:0]         accumulate,
    input  wire                         irq,
    input  wire [NUM_BANKS-1:0]         unit_busy,

    // ---------------- conv descriptors (packed, bank b at [b*W +: W]) ----------------
    output wire [NUM_BANKS*DIM_WIDTH-1:0]  ifmap_cols,
    output wire [NUM_BANKS*DIM_WIDTH-1:0]  kernel_rows,
    output wire [NUM_BANKS*DIM_WIDTH-1:0]  kernel_cols,
    output wire [NUM_BANKS*DIM_WIDTH-1:0]  input_channels,
    output wire [NUM_BANKS*DIM_WIDTH-1:0]  output_channels,
    output wire [NUM_BANKS*DIM_WIDTH-1:0]  stride,
    output wire [NUM_BANKS*DIM_WIDTH-1:0]  ofmap_rows,
    output wire [NUM_BANKS*DIM_WIDTH-1:0]  ofmap_cols,
    output wire [NUM_BANKS*DIM_WIDTH-1:0]  weights_per_oc,
    output wire [NUM_BANKS*ADDR_WIDTH-1:0] ifmap_ch_size,
    output wire [NUM_BANKS*ADDR_WIDTH-1:0] ofmap_ch_size,
    output wire [NUM_BANKS*ADDR_WIDTH-1:0] ifmap_row_step,
    output wire [NUM_BANKS*ADDR_WIDTH-1:0] ifmap_base_addr,
    output wire [NUM_BANKS*ADDR_WIDTH-1:0] weight_base_addr,
    output wire [NUM_BANKS*ADDR_WIDTH-1:0] ofmap_base_addr,
    output wire [NUM_BANKS*RF_AW-1:0]      rf_row_step,
    output wire [NUM_BANKS*RF_AW-1:0]      rf_ch_stride,

    // ---------------- post-proc control + descriptors ----------------
    output wire [NUM_BANKS-1:0]            pp_mask,
    output wire                            pp_start,
    input  wire                            pp_irq,
    input  wire [NUM_BANKS-1:0]            pp_busy,
    output wire [NUM_BANKS*ADDR_WIDTH-1:0] pp_src_base,
    output wire [NUM_BANKS*ADDR_WIDTH-1:0] pp_dst_base,
    output wire [NUM_BANKS*DIM_WIDTH-1:0]  pp_src_cols,
    output wire [NUM_BANKS*DIM_WIDTH-1:0]  pp_out_rows,
    output wire [NUM_BANKS*DIM_WIDTH-1:0]  pp_out_cols,
    output wire [NUM_BANKS*DIM_WIDTH-1:0]  pp_channels,
    output wire [NUM_BANKS*DIM_WIDTH-1:0]  pp_pool_k,
    output wire [NUM_BANKS*ADDR_WIDTH-1:0] pp_src_ch_size,
    output wire [NUM_BANKS*ADDR_WIDTH-1:0] pp_row_step,
    output wire [NUM_BANKS*ADDR_WIDTH-1:0] pp_total_out,
    output wire [NUM_BANKS*8-1:0]          pp_out_zp,
    output wire [NUM_BANKS-1:0]            pp_relu_en,
    output wire [NUM_BANKS-1:0]            pp_qp_we,
    output wire [QA_W-1:0]                 pp_qp_waddr,
    output wire signed [31:0]              pp_qp_bias,
    output wire [15:0]                     pp_qp_mult,
    output wire [4:0]                      pp_qp_shift,

    // ---------------- performance (packed per bank, from the cluster) ----------------
    input  wire [NUM_BANKS-1:0]            perf_valid,
    input  wire [NUM_BANKS*CNT_W-1:0]      perf_cyc_total,
    input  wire [NUM_BANKS*CNT_W-1:0]      perf_cyc_mac,
    input  wire [NUM_BANKS*CNT_W-1:0]      perf_cyc_stall,
    input  wire [NUM_BANKS*CNT_W-1:0]      perf_cyc_mem_blocked,
    input  wire [NUM_BANKS*CNT_W-1:0]      perf_mac_ops
);

    // ------------------------------------------------------------------
    // Address map constants
    // ------------------------------------------------------------------
    localparam NF     = 32;                        // config words per bank (0x00..0x7C)
    localparam BLK_W  = BUS_AW - 8;

    localparam [31:0] ID_VAL     = 32'h4E4D0001;
    localparam [31:0] CONFIG_VAL = (MAX_OC << 16) | (CNT_W << 8) | NUM_BANKS;

    // global word indices (byte address / 4)
    localparam [5:0] G_ID = 6'd0,  G_CTRL = 6'd1,  G_UMASK = 6'd2, G_PMASK = 6'd3,
                     G_STATUS = 6'd4, G_UBUSY = 6'd5, G_PBUSY = 6'd6, G_PVALID = 6'd7,
                     G_IRQST = 6'd8,  G_IRQEN = 6'd9, G_CONFIG = 6'd10,
                     G_QPMASK = 6'd16, G_QPADDR = 6'd17, G_QPBIAS = 6'd18,
                     G_QPMULT = 6'd19, G_QPSHIFT = 6'd20;

    // bank config word indices
    localparam [4:0] F_IFMAP_COLS = 5'd0,  F_KERNEL_ROWS = 5'd1,  F_KERNEL_COLS = 5'd2,
                     F_IN_CH = 5'd3,       F_OUT_CH = 5'd4,       F_STRIDE = 5'd5,
                     F_OFMAP_ROWS = 5'd6,  F_OFMAP_COLS = 5'd7,   F_WPO = 5'd8,
                     F_IFMAP_CH_SIZE = 5'd9, F_OFMAP_CH_SIZE = 5'd10, F_IFMAP_ROW_STEP = 5'd11,
                     F_IFMAP_BASE = 5'd12, F_WEIGHT_BASE = 5'd13, F_OFMAP_BASE = 5'd14,
                     F_RF_ROW_STEP = 5'd15, F_RF_CH_STRIDE = 5'd16, F_ACCUM = 5'd17,
                     F_PP_SRC_BASE = 5'd20, F_PP_DST_BASE = 5'd21, F_PP_SRC_COLS = 5'd22,
                     F_PP_OUT_ROWS = 5'd23, F_PP_OUT_COLS = 5'd24, F_PP_CHANNELS = 5'd25,
                     F_PP_POOL_K = 5'd26,   F_PP_SRC_CH_SIZE = 5'd27, F_PP_ROW_STEP = 5'd28,
                     F_PP_TOTAL_OUT = 5'd29, F_PP_OUT_ZP = 5'd30, F_PP_RELU = 5'd31;
    // words 0..19 belong to the conv descriptor, 20..31 to the post-proc descriptor

    localparam [31:0] M_DIM = (32'h1 << DIM_WIDTH)  - 32'h1;
    localparam [31:0] M_ADR = (32'h1 << ADDR_WIDTH) - 32'h1;
    localparam [31:0] M_RF  = (32'h1 << RF_AW)      - 32'h1;
    localparam [31:0] M_QA  = (32'h1 << QA_W)       - 32'h1;
    localparam [31:0] M_BNK = (32'h1 << NUM_BANKS)  - 32'h1;

    // which bits of a config word exist (0 = undefined / reserved word)
    function [31:0] fmask(input [4:0] f);
        case (f)
            F_IFMAP_COLS, F_KERNEL_ROWS, F_KERNEL_COLS, F_IN_CH, F_OUT_CH, F_STRIDE,
            F_OFMAP_ROWS, F_OFMAP_COLS, F_WPO,
            F_PP_SRC_COLS, F_PP_OUT_ROWS, F_PP_OUT_COLS, F_PP_CHANNELS, F_PP_POOL_K: fmask = M_DIM;
            F_IFMAP_CH_SIZE, F_OFMAP_CH_SIZE, F_IFMAP_ROW_STEP,
            F_IFMAP_BASE, F_WEIGHT_BASE, F_OFMAP_BASE,
            F_PP_SRC_BASE, F_PP_DST_BASE, F_PP_SRC_CH_SIZE, F_PP_ROW_STEP, F_PP_TOTAL_OUT: fmask = M_ADR;
            F_RF_ROW_STEP, F_RF_CH_STRIDE: fmask = M_RF;
            F_ACCUM, F_PP_RELU:            fmask = 32'h1;
            F_PP_OUT_ZP:                   fmask = 32'hFF;
            default:                       fmask = 32'h0;
        endcase
    endfunction

    // ------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------
    reg [31:0]            cfg [0:NUM_BANKS*NF-1];      // [bank*NF + word]

    reg [NUM_BANKS-1:0]   umask_r, pmask_r;
    reg                   conv_start_q, pp_start_q;
    reg                   conv_active, pp_active;
    reg [3:0]             irq_st, irq_en;

    reg [NUM_BANKS-1:0]   qp_mask, qp_we_q;
    reg [QA_W-1:0]        qp_addr;
    reg signed [31:0]     qp_bias;
    reg [15:0]            qp_mult;
    reg [4:0]             qp_shift;

    assign unit_mask   = umask_r;
    assign pp_mask     = pmask_r;
    assign start       = conv_start_q;
    assign pp_start    = pp_start_q;
    assign pp_qp_we    = qp_we_q;
    assign pp_qp_waddr = qp_addr;
    assign pp_qp_bias  = qp_bias;
    assign pp_qp_mult  = qp_mult;
    assign pp_qp_shift = qp_shift;
    assign irq_o       = |(irq_st & irq_en);

    // ------------------------------------------------------------------
    // Address decode
    // ------------------------------------------------------------------
    wire [5:0]       word_lo = reg_addr[7:2];
    wire [BLK_W-1:0] blk     = reg_addr[BUS_AW-1:8];
    wire             g_blk   = (blk == 0);
    wire             b_blk   = (blk != 0) && (blk <= NUM_BANKS);
    wire [31:0]      bidx    = {{(32-BLK_W){1'b0}}, blk} - 32'd1;

    wire wr        = reg_req &  reg_we;
    wire busy_any  = conv_active | pp_active;

    // ------------------------------------------------------------------
    // Read mux (combinational, registered on the bus below)
    // ------------------------------------------------------------------
    reg [31:0] rdata_c;
    reg        rerr_c;
    reg [63:0] pv;

    always @(*) begin
        rdata_c = 32'h0;
        rerr_c  = 1'b0;
        pv      = 64'h0;
        if (g_blk) begin
            case (word_lo)
                G_ID:      rdata_c = ID_VAL;
                G_CTRL:    rdata_c = 32'h0;
                G_UMASK:   rdata_c = umask_r;
                G_PMASK:   rdata_c = pmask_r;
                G_STATUS:  rdata_c = {30'h0, pp_active, conv_active};
                G_UBUSY:   rdata_c = unit_busy;
                G_PBUSY:   rdata_c = pp_busy;
                G_PVALID:  rdata_c = perf_valid;
                G_IRQST:   rdata_c = irq_st;
                G_IRQEN:   rdata_c = irq_en;
                G_CONFIG:  rdata_c = CONFIG_VAL;
                G_QPMASK:  rdata_c = qp_mask;
                G_QPADDR:  rdata_c = qp_addr;
                G_QPBIAS:  rdata_c = qp_bias;
                G_QPMULT:  rdata_c = qp_mult;
                G_QPSHIFT: rdata_c = qp_shift;
                default:   rerr_c  = 1'b1;
            endcase
        end else if (b_blk) begin
            if (!word_lo[5]) begin
                rdata_c = cfg[bidx*NF + word_lo[4:0]];
                if (fmask(word_lo[4:0]) == 32'h0) rerr_c = 1'b1;
            end else begin
                case (word_lo[4:1])
                    4'd0:    pv[CNT_W-1:0] = perf_cyc_total      [bidx*CNT_W +: CNT_W];
                    4'd1:    pv[CNT_W-1:0] = perf_cyc_mac        [bidx*CNT_W +: CNT_W];
                    4'd2:    pv[CNT_W-1:0] = perf_cyc_stall      [bidx*CNT_W +: CNT_W];
                    4'd3:    pv[CNT_W-1:0] = perf_cyc_mem_blocked[bidx*CNT_W +: CNT_W];
                    4'd4:    pv[CNT_W-1:0] = perf_mac_ops        [bidx*CNT_W +: CNT_W];
                    default: rerr_c = 1'b1;
                endcase
                rdata_c = word_lo[0] ? pv[63:32] : pv[31:0];
            end
        end else begin
            rerr_c = 1'b1;
        end
    end

    // ------------------------------------------------------------------
    // Writes, start handling, status, bus response
    // ------------------------------------------------------------------
    reg        werr;
    reg [4:0]  f;
    reg        do_conv, do_pp, conv_ok, pp_ok;
    integer    i;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (i = 0; i < NUM_BANKS*NF; i = i + 1) cfg[i] <= 32'h0;
            umask_r <= 0;  pmask_r <= 0;
            conv_start_q <= 1'b0;  pp_start_q <= 1'b0;
            conv_active  <= 1'b0;  pp_active  <= 1'b0;
            irq_st <= 4'h0;  irq_en <= 4'h0;
            qp_mask <= 0;  qp_we_q <= 0;  qp_addr <= 0;
            qp_bias <= 0;  qp_mult <= 0;  qp_shift <= 0;
            reg_rvalid <= 1'b0;  reg_rdata <= 32'h0;  reg_err <= 1'b0;
        end else begin
            conv_start_q <= 1'b0;
            pp_start_q   <= 1'b0;
            qp_we_q      <= 0;
            // channel pointer moves AFTER the commit cycle (qp_we_q high), so the
            // cluster sees the address it was committed with; a bus write wins
            if (|qp_we_q) qp_addr <= qp_addr + 1'b1;

            werr = 1'b0;

            if (wr) begin
                if (g_blk) begin
                    case (word_lo)
                        G_CTRL: begin
                            do_conv = reg_wdata[0];
                            do_pp   = reg_wdata[1];
                            conv_ok = do_conv && !busy_any && (|umask_r);
                            pp_ok   = do_pp && !do_conv && !busy_any && (|pmask_r);
                            if (conv_ok) begin conv_start_q <= 1'b1; conv_active <= 1'b1; end
                            if (pp_ok)   begin pp_start_q   <= 1'b1; pp_active   <= 1'b1; end
                            if ((do_conv && !conv_ok) || (do_pp && !pp_ok)) irq_st[2] <= 1'b1;
                        end
                        G_UMASK:  if (conv_active) irq_st[3] <= 1'b1;
                                  else             umask_r   <= reg_wdata & M_BNK;
                        G_PMASK:  if (pp_active)   irq_st[3] <= 1'b1;
                                  else             pmask_r   <= reg_wdata & M_BNK;
                        G_IRQST:  irq_st <= irq_st & ~reg_wdata[3:0];
                        G_IRQEN:  irq_en <= reg_wdata[3:0];
                        G_QPMASK, G_QPADDR, G_QPBIAS, G_QPMULT, G_QPSHIFT:
                            if (pp_active) irq_st[3] <= 1'b1;
                            else case (word_lo)
                                G_QPMASK:  qp_mask  <= reg_wdata & M_BNK;
                                G_QPADDR:  qp_addr  <= reg_wdata & M_QA;
                                G_QPBIAS:  qp_bias  <= reg_wdata;
                                G_QPMULT:  qp_mult  <= reg_wdata[15:0];
                                default: begin                    // G_QPSHIFT: commit
                                    qp_shift <= reg_wdata[4:0];
                                    qp_we_q  <= qp_mask;
                                end
                            endcase
                        default: werr = 1'b1;                      // RO / reserved
                    endcase
                end else if (b_blk && !word_lo[5]) begin
                    f = word_lo[4:0];
                    if (fmask(f) == 32'h0)                   werr = 1'b1;
                    else if ((f < 5'd20) ? conv_active : pp_active)
                                                             irq_st[3] <= 1'b1;
                    else cfg[bidx*NF + f] <= reg_wdata & fmask(f);
                end else begin
                    werr = 1'b1;                                   // perf (RO) or unmapped
                end
            end

            // events (after the write block so a set wins over a same-cycle clear)
            if (irq)    begin irq_st[0] <= 1'b1; conv_active <= 1'b0; end
            if (pp_irq) begin irq_st[1] <= 1'b1; pp_active   <= 1'b0; end

            reg_rvalid <= reg_req;
            if (reg_req) reg_rdata <= reg_we ? 32'h0 : rdata_c;
            reg_err    <= reg_req & (reg_we ? werr : rerr_c);
        end
    end

    // ------------------------------------------------------------------
    // Descriptor fan-out to the cluster
    // ------------------------------------------------------------------
    genvar b, k;
    generate for (b = 0; b < NUM_BANKS; b = b + 1) begin : g_bank
        wire [NF*32-1:0] bc;
        for (k = 0; k < NF; k = k + 1) begin : g_w
            assign bc[k*32 +: 32] = cfg[b*NF + k];
        end

        assign ifmap_cols      [b*DIM_WIDTH  +: DIM_WIDTH ] = bc[F_IFMAP_COLS*32     +: DIM_WIDTH ];
        assign kernel_rows     [b*DIM_WIDTH  +: DIM_WIDTH ] = bc[F_KERNEL_ROWS*32    +: DIM_WIDTH ];
        assign kernel_cols     [b*DIM_WIDTH  +: DIM_WIDTH ] = bc[F_KERNEL_COLS*32    +: DIM_WIDTH ];
        assign input_channels  [b*DIM_WIDTH  +: DIM_WIDTH ] = bc[F_IN_CH*32          +: DIM_WIDTH ];
        assign output_channels [b*DIM_WIDTH  +: DIM_WIDTH ] = bc[F_OUT_CH*32         +: DIM_WIDTH ];
        assign stride          [b*DIM_WIDTH  +: DIM_WIDTH ] = bc[F_STRIDE*32         +: DIM_WIDTH ];
        assign ofmap_rows      [b*DIM_WIDTH  +: DIM_WIDTH ] = bc[F_OFMAP_ROWS*32     +: DIM_WIDTH ];
        assign ofmap_cols      [b*DIM_WIDTH  +: DIM_WIDTH ] = bc[F_OFMAP_COLS*32     +: DIM_WIDTH ];
        assign weights_per_oc  [b*DIM_WIDTH  +: DIM_WIDTH ] = bc[F_WPO*32            +: DIM_WIDTH ];
        assign ifmap_ch_size   [b*ADDR_WIDTH +: ADDR_WIDTH] = bc[F_IFMAP_CH_SIZE*32  +: ADDR_WIDTH];
        assign ofmap_ch_size   [b*ADDR_WIDTH +: ADDR_WIDTH] = bc[F_OFMAP_CH_SIZE*32  +: ADDR_WIDTH];
        assign ifmap_row_step  [b*ADDR_WIDTH +: ADDR_WIDTH] = bc[F_IFMAP_ROW_STEP*32 +: ADDR_WIDTH];
        assign ifmap_base_addr [b*ADDR_WIDTH +: ADDR_WIDTH] = bc[F_IFMAP_BASE*32     +: ADDR_WIDTH];
        assign weight_base_addr[b*ADDR_WIDTH +: ADDR_WIDTH] = bc[F_WEIGHT_BASE*32    +: ADDR_WIDTH];
        assign ofmap_base_addr [b*ADDR_WIDTH +: ADDR_WIDTH] = bc[F_OFMAP_BASE*32     +: ADDR_WIDTH];
        assign rf_row_step     [b*RF_AW      +: RF_AW     ] = bc[F_RF_ROW_STEP*32    +: RF_AW     ];
        assign rf_ch_stride    [b*RF_AW      +: RF_AW     ] = bc[F_RF_CH_STRIDE*32   +: RF_AW     ];
        assign accumulate[b]                                = bc[F_ACCUM*32];

        assign pp_src_base     [b*ADDR_WIDTH +: ADDR_WIDTH] = bc[F_PP_SRC_BASE*32    +: ADDR_WIDTH];
        assign pp_dst_base     [b*ADDR_WIDTH +: ADDR_WIDTH] = bc[F_PP_DST_BASE*32    +: ADDR_WIDTH];
        assign pp_src_cols     [b*DIM_WIDTH  +: DIM_WIDTH ] = bc[F_PP_SRC_COLS*32    +: DIM_WIDTH ];
        assign pp_out_rows     [b*DIM_WIDTH  +: DIM_WIDTH ] = bc[F_PP_OUT_ROWS*32    +: DIM_WIDTH ];
        assign pp_out_cols     [b*DIM_WIDTH  +: DIM_WIDTH ] = bc[F_PP_OUT_COLS*32    +: DIM_WIDTH ];
        assign pp_channels     [b*DIM_WIDTH  +: DIM_WIDTH ] = bc[F_PP_CHANNELS*32    +: DIM_WIDTH ];
        assign pp_pool_k       [b*DIM_WIDTH  +: DIM_WIDTH ] = bc[F_PP_POOL_K*32      +: DIM_WIDTH ];
        assign pp_src_ch_size  [b*ADDR_WIDTH +: ADDR_WIDTH] = bc[F_PP_SRC_CH_SIZE*32 +: ADDR_WIDTH];
        assign pp_row_step     [b*ADDR_WIDTH +: ADDR_WIDTH] = bc[F_PP_ROW_STEP*32    +: ADDR_WIDTH];
        assign pp_total_out    [b*ADDR_WIDTH +: ADDR_WIDTH] = bc[F_PP_TOTAL_OUT*32   +: ADDR_WIDTH];
        assign pp_out_zp       [b*8          +: 8         ] = bc[F_PP_OUT_ZP*32      +: 8         ];
        assign pp_relu_en[b]                                = bc[F_PP_RELU*32];
    end endgenerate

endmodule