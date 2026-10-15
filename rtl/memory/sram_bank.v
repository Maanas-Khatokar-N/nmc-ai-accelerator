// ============================================================================
// SRAM bank
//
//   RISC-V core : ONE word per cycle per bank (single I/O). Word address is
//                 low-order interleaved: macro = addr[SEL_W-1:0],
//                 local word = addr[CORE_AW-1:SEL_W]. The core is agnostic of
//                 the internal organisation (paper Sec. III-A).
//   NMC unit    : ONE independent request PER MACRO each cycle, with its own
//                 macro-local address, write enable and write data. All
//                 NUM_MACROS macros can be accessed in the same cycle, so
//
//                   BW_NMC  = #Banks x #Macros  [access/cycle]   (Eq. 1)
//                   BW_Core = #Banks            [access/cycle]   (Eq. 2)
//
//                 i.e. NUM_MACROS x the core bandwidth inside one bank
//                 (8x here, 32x over 4 banks).
//
// Conflicts (paper Sec. III-B, fine-grained resolution): if the core and the
// NMC hit the SAME macro in the same cycle, the NMC request to that macro is
// suppressed (nmc_blocked = 1, nmc_grant = 0) and the core wins. The NMC
// retries; the other macros are unaffected. The core is therefore never
// stalled and has no grant output.
//
// Timing: everything is a single-cycle access.
//   - nmc_grant / nmc_blocked are COMBINATIONAL from nmc_req and core_req
//   - read data arrives one cycle after a granted read:
//        core_rvalid / nmc_rvalid[m] are high in that cycle
//   - writes produce no rvalid
//
// NMC address is macro-local (the caller decides which macro holds which
// element, e.g. IFMap lane i -> macro i). Round-robin / access_allowed
// windows from the paper belong to the NMC controller, which simply gates
// nmc_req; this wrapper only does the per-macro conflict resolution.
// ============================================================================
module sram_bank #(
    parameter DATA_WIDTH      = 32,
    parameter NUM_MACROS      = 8,
    parameter DEPTH_PER_MACRO = 4096,
    parameter MACRO_AW        = $clog2(DEPTH_PER_MACRO),   // macro-local addr bits
    parameter SEL_W           = $clog2(NUM_MACROS),        // macro-select bits
    parameter CORE_AW         = SEL_W + MACRO_AW           // core word-addr bits
) (
    input  wire                           clk,
    input  wire                           rst,

    // ---------------- RISC-V core port (1 word / cycle) ----------------
    input  wire                           core_req,
    input  wire                           core_we,
    input  wire [CORE_AW-1:0]             core_addr,       // word address
    input  wire [DATA_WIDTH-1:0]          core_wdata,
    output wire [DATA_WIDTH-1:0]          core_rdata,
    output wire                           core_rvalid,

    // ---------------- NMC port (1 request per macro) ----------------
    input  wire [NUM_MACROS-1:0]                  nmc_req,
    input  wire [NUM_MACROS-1:0]                  nmc_we,
    input  wire [NUM_MACROS*MACRO_AW-1:0]         nmc_addr,   // macro m at [m*MACRO_AW +: MACRO_AW]
    input  wire [NUM_MACROS*DATA_WIDTH-1:0]       nmc_wdata,
    output wire [NUM_MACROS-1:0]                  nmc_grant,   // request accepted this cycle
    output wire [NUM_MACROS-1:0]                  nmc_blocked, // request lost to the core
    output wire [NUM_MACROS*DATA_WIDTH-1:0]       nmc_rdata,
    output wire [NUM_MACROS-1:0]                  nmc_rvalid
);

    // ------------------------------------------------------------------
    // Core address decode (low-order interleave)
    // ------------------------------------------------------------------
    wire [SEL_W-1:0]    core_sel   = core_addr[SEL_W-1:0];
    wire [MACRO_AW-1:0] core_local = core_addr[CORE_AW-1:SEL_W];

    // One-hot "core hits macro m"
    wire [NUM_MACROS-1:0] core_hit;

    // ------------------------------------------------------------------
    // Per-macro arbitration and macro instances
    // ------------------------------------------------------------------
    wire [NUM_MACROS*DATA_WIDTH-1:0] macro_rdata;

    genvar m;
    generate
        for (m = 0; m < NUM_MACROS; m = m + 1) begin : gen_macro

            assign core_hit[m]    = core_req && (core_sel == m);
            assign nmc_grant[m]   = nmc_req[m] && !core_hit[m];
            assign nmc_blocked[m] = nmc_req[m] &&  core_hit[m];

            wire                  m_en    = core_hit[m] | nmc_grant[m];
            wire                  m_we    = core_hit[m] ? core_we : nmc_we[m];
            wire [MACRO_AW-1:0]   m_addr  = core_hit[m] ? core_local
                                                        : nmc_addr[m*MACRO_AW +: MACRO_AW];
            wire [DATA_WIDTH-1:0] m_wdata = core_hit[m] ? core_wdata
                                                        : nmc_wdata[m*DATA_WIDTH +: DATA_WIDTH];

            sram_macro #(
                .DATA_WIDTH(DATA_WIDTH),
                .DEPTH     (DEPTH_PER_MACRO)
            ) u_macro (
                .clk  (clk),
                .en   (m_en),
                .we   (m_we),
                .addr (m_addr),
                .wdata(m_wdata),
                .rdata(macro_rdata[m*DATA_WIDTH +: DATA_WIDTH])
            );
        end
    endgenerate

    // ------------------------------------------------------------------
    // Read return path: remember who read what, one cycle later
    // ------------------------------------------------------------------
    reg                    core_rd_q;
    reg [SEL_W-1:0]        core_sel_q;
    reg [NUM_MACROS-1:0]   nmc_rd_q;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            core_rd_q  <= 1'b0;
            core_sel_q <= {SEL_W{1'b0}};
            nmc_rd_q   <= {NUM_MACROS{1'b0}};
        end else begin
            core_rd_q  <= core_req && !core_we;
            core_sel_q <= core_sel;
            nmc_rd_q   <= nmc_grant & ~nmc_we;
        end
    end

    assign core_rvalid = core_rd_q;
    assign core_rdata  = macro_rdata[core_sel_q*DATA_WIDTH +: DATA_WIDTH];

    // NMC sees every macro's output directly (that is the wide read);
    // nmc_rvalid says which lanes carry a read result this cycle.
    assign nmc_rdata  = macro_rdata;
    assign nmc_rvalid = nmc_rd_q;

endmodule