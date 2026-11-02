// ============================================================================
// NMC AXI wrapper  (connects nmc_top to the Zynq / FPGA ARM CPU)
//
//   ARM (PS) --AXI4-Lite--> s_axi_cfg  --+
//                                        +--> arbiter --> nmc_top (1 bus port)
//   ARM / DMA --AXI4------> s_axi_mem  --+                 |-> nmc_regs
//                                                          |-> nmc_cluster (SRAM)
//   irq  <---------------------------------------------- irq_o
//
// Address map (assign both in the Vivado Address Editor):
//   s_axi_cfg : 4 KB    register block, byte offsets exactly as in nmc_regs.v
//                       + wrapper registers at 0xF00 (bank blocks only use 0x100..0x4FF
//                         with NUM_BANKS = 4, so 0xF00 is free)
//                         0xF00  WRAP_ID    RO  0x41584901
//                         0xF04  WRAP_CTRL  RW  [0] SOFT_RESET: write 1 -> resets nmc_top
//                                               for SOFT_RST_CYCLES cycles (self-clearing,
//                                               reads 1 while the reset is active).
//                                               Clears descriptors, masks, irq state.
//   s_axi_mem : 512 KB  SRAM window, byte offset = {bank, word-in-bank, 2'b00}
//                       (4 banks x 128 KB, one 32-bit word per element, like nmc_top)
//
// Bus behaviour
//   - 32-bit data, one outstanding transaction per port (no interleaving, IDs echoed).
//   - s_axi_cfg  : AXI4-Lite, wstrb ignored (word access only). Errors from nmc_regs
//                  (unmapped / read-only / reserved) come back as SLVERR.
//   - s_axi_mem  : AXI4, INCR and FIXED bursts up to 256 beats, awsize/arsize 4 bytes
//                  for full speed. Partial wstrb is handled with a read-modify-write
//                  (SRAM has no byte enables), so byte/halfword stores from the ARM work.
//                  WRAP / reserved burst types complete but return SLVERR.
//   - The ARM never stalls the accelerator and vice versa: contention is resolved
//     inside sram_bank (host wins per macro, NMC retries). Config accesses have
//     priority over SRAM accesses at the single nmc_top bus port.
//
// Reset / clock
//   - one clock (aclk), nmc_top runs on it directly (it is synchronous to aclk).
//   - aresetn (active low, from Processor System Reset) is converted to the
//     active-high async-assert / sync-deassert reset the NMC RTL uses.
//   - SOFT_RESET resets only nmc_top, the AXI side keeps running.
//
// Interrupt
//   - irq = nmc_top irq_o (level, active high). Connect to the PS IRQ_F2P
//     (level-sensitive). Clear through IRQ_STATUS (W1C) in nmc_regs.
//
// Throughput (bring-up, not optimised): ~3 cycles per read beat, ~2-3 per write beat.
// Use 4-byte aligned accesses / bursts for data. Do not use the SRAM window with
// a cacheable mapping from Linux (use /dev/mem O_SYNC or UIO, or device memory attributes).
// ============================================================================
module nmc_axi_wrapper #(
    parameter ID_W            = 16,      // AXI ID width of s_axi_mem (>= master's ID width)
    parameter MEM_AW          = 19,      // s_axi_mem address bits (512 KB)
    parameter SOFT_RST_CYCLES = 8,       // length of SOFT_RESET (1..255)
    // must match nmc_top
    parameter NUM_BANKS       = 4,
    parameter NUM_MACROS      = 8,
    parameter DEPTH_PER_MACRO = 4096
)(
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axi_cfg:s_axi_mem, ASSOCIATED_RESET aresetn" *)
    input  wire                 aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire                 aresetn,

    (* X_INTERFACE_INFO = "xilinx.com:signal:interrupt:1.0 irq INTERRUPT" *)
    (* X_INTERFACE_PARAMETER = "SENSITIVITY LEVEL_HIGH" *)
    output wire                 irq,

    // ---------------- AXI4-Lite slave: registers (4 KB) ----------------
    input  wire [11:0]          s_axi_cfg_awaddr,
    input  wire                 s_axi_cfg_awvalid,
    output wire                 s_axi_cfg_awready,
    input  wire [31:0]          s_axi_cfg_wdata,
    input  wire [3:0]           s_axi_cfg_wstrb,
    input  wire                 s_axi_cfg_wvalid,
    output wire                 s_axi_cfg_wready,
    output wire [1:0]           s_axi_cfg_bresp,
    output wire                 s_axi_cfg_bvalid,
    input  wire                 s_axi_cfg_bready,
    input  wire [11:0]          s_axi_cfg_araddr,
    input  wire                 s_axi_cfg_arvalid,
    output wire                 s_axi_cfg_arready,
    output wire [31:0]          s_axi_cfg_rdata,
    output wire [1:0]           s_axi_cfg_rresp,
    output wire                 s_axi_cfg_rvalid,
    input  wire                 s_axi_cfg_rready,

    // ---------------- AXI4 slave: SRAM window (512 KB) ----------------
    input  wire [ID_W-1:0]      s_axi_mem_awid,
    input  wire [MEM_AW-1:0]    s_axi_mem_awaddr,
    input  wire [7:0]           s_axi_mem_awlen,
    input  wire [2:0]           s_axi_mem_awsize,
    input  wire [1:0]           s_axi_mem_awburst,
    input  wire                 s_axi_mem_awvalid,
    output wire                 s_axi_mem_awready,
    input  wire [31:0]          s_axi_mem_wdata,
    input  wire [3:0]           s_axi_mem_wstrb,
    input  wire                 s_axi_mem_wlast,
    input  wire                 s_axi_mem_wvalid,
    output wire                 s_axi_mem_wready,
    output wire [ID_W-1:0]      s_axi_mem_bid,
    output wire [1:0]           s_axi_mem_bresp,
    output wire                 s_axi_mem_bvalid,
    input  wire                 s_axi_mem_bready,
    input  wire [ID_W-1:0]      s_axi_mem_arid,
    input  wire [MEM_AW-1:0]    s_axi_mem_araddr,
    input  wire [7:0]           s_axi_mem_arlen,
    input  wire [2:0]           s_axi_mem_arsize,
    input  wire [1:0]           s_axi_mem_arburst,
    input  wire                 s_axi_mem_arvalid,
    output wire                 s_axi_mem_arready,
    output wire [ID_W-1:0]      s_axi_mem_rid,
    output wire [31:0]          s_axi_mem_rdata,
    output wire [1:0]           s_axi_mem_rresp,
    output wire                 s_axi_mem_rlast,
    output wire                 s_axi_mem_rvalid,
    input  wire                 s_axi_mem_rready
);

    localparam MEM_BYTE_AW = $clog2(NUM_BANKS) + $clog2(NUM_MACROS) + $clog2(DEPTH_PER_MACRO) + 2;
    localparam TOP_AW      = MEM_BYTE_AW + 1;
    localparam [31:0] WRAP_ID = 32'h41584901;

    localparam [1:0] OKAY = 2'b00, SLVERR = 2'b10;

    // ------------------------------------------------------------------
    // Reset: async assert, synchronous release; plus soft reset for nmc_top
    // ------------------------------------------------------------------
    reg [1:0] rst_pipe;
    always @(posedge aclk or negedge aresetn)
        if (!aresetn) rst_pipe <= 2'b11;
        else          rst_pipe <= {rst_pipe[0], 1'b0};
    wire rst = rst_pipe[1];

    reg        sr_set;
    reg [7:0]  sr_cnt;
    reg        core_rst_q;
    wire       soft_rst = (sr_cnt != 8'd0);

    always @(posedge aclk or posedge rst)
        if (rst) begin
            sr_cnt     <= 8'd0;
            core_rst_q <= 1'b1;
        end else begin
            if (sr_set)            sr_cnt <= SOFT_RST_CYCLES[7:0];
            else if (sr_cnt != 0)  sr_cnt <= sr_cnt - 8'd1;
            core_rst_q <= soft_rst;
        end

    wire core_rst = rst | core_rst_q;

    // ------------------------------------------------------------------
    // nmc_top bus (single port, shared by the two AXI slaves)
    // ------------------------------------------------------------------
    wire                bus_req, bus_we, bus_rvalid, bus_err;
    wire [TOP_AW-1:0]   bus_addr;
    wire [31:0]         bus_wdata, bus_rdata;

    nmc_top #(
        .NUM_BANKS(NUM_BANKS), .NUM_MACROS(NUM_MACROS), .DEPTH_PER_MACRO(DEPTH_PER_MACRO)
    ) u_nmc (
        .clk(aclk), .rst(core_rst),
        .bus_req(bus_req), .bus_we(bus_we), .bus_addr(bus_addr), .bus_wdata(bus_wdata),
        .bus_rvalid(bus_rvalid), .bus_rdata(bus_rdata), .bus_err(bus_err),
        .irq_o(irq)
    );

    // response timing of nmc_top: data / err are valid in the cycle AFTER the request
    // (bus_rvalid). Each FSM below enters a *_RESP state exactly then.

    // ------------------------------------------------------------------
    // CFG port: AXI4-Lite slave FSM
    // ------------------------------------------------------------------
    localparam [2:0] C_IDLE = 3'd0, C_REQ = 3'd1, C_RESP = 3'd2, C_B = 3'd3, C_R = 3'd4;
    reg [2:0]  cst;
    reg        have_aw, have_w, have_ar;
    reg [11:0] c_addr;
    reg [31:0] c_wdata, c_rdata;
    reg        c_we, c_err;

    wire c_idle  = (cst == C_IDLE) & ~rst;
    wire c_local = (c_addr[11:8] == 4'hF);              // wrapper registers
    wire c_req   = (cst == C_REQ) & ~c_local;
    wire c_gnt   = c_req & ~core_rst;

    assign s_axi_cfg_awready = c_idle & ~have_aw & ~have_ar;
    assign s_axi_cfg_wready  = c_idle & ~have_w  & ~have_ar;
    assign s_axi_cfg_arready = c_idle & ~have_ar & ~have_aw & ~have_w
                                      & ~s_axi_cfg_awvalid & ~s_axi_cfg_wvalid;

    assign s_axi_cfg_bvalid  = (cst == C_B);
    assign s_axi_cfg_bresp   = c_err ? SLVERR : OKAY;
    assign s_axi_cfg_rvalid  = (cst == C_R);
    assign s_axi_cfg_rdata   = c_rdata;
    assign s_axi_cfg_rresp   = c_err ? SLVERR : OKAY;

    always @(posedge aclk or posedge rst) begin
        if (rst) begin
            cst <= C_IDLE;
            have_aw <= 1'b0; have_w <= 1'b0; have_ar <= 1'b0;
            c_addr <= 12'd0; c_wdata <= 32'd0; c_rdata <= 32'd0;
            c_we <= 1'b0; c_err <= 1'b0; sr_set <= 1'b0;
        end else begin
            sr_set <= 1'b0;
            case (cst)
                C_IDLE: begin
                    if (s_axi_cfg_awvalid & s_axi_cfg_awready) begin
                        have_aw <= 1'b1; c_addr <= s_axi_cfg_awaddr;
                    end
                    if (s_axi_cfg_wvalid & s_axi_cfg_wready) begin
                        have_w <= 1'b1; c_wdata <= s_axi_cfg_wdata;
                    end
                    if (s_axi_cfg_arvalid & s_axi_cfg_arready) begin
                        have_ar <= 1'b1; c_addr <= s_axi_cfg_araddr;
                    end
                    if (have_aw & have_w)  begin c_we <= 1'b1; cst <= C_REQ; end
                    else if (have_ar)      begin c_we <= 1'b0; cst <= C_REQ; end
                end

                C_REQ: begin
                    if (c_local) begin                       // wrapper registers, no bus access
                        c_err <= 1'b0;
                        if (c_we) begin
                            if (c_addr[7:2] == 6'd1) begin
                                if (c_wdata[0]) sr_set <= 1'b1;
                            end else c_err <= 1'b1;          // WRAP_ID is RO, rest reserved
                            c_rdata <= 32'd0;
                        end else begin
                            case (c_addr[7:2])
                                6'd0:    c_rdata <= WRAP_ID;
                                6'd1:    c_rdata <= {31'd0, soft_rst};
                                default: begin c_rdata <= 32'd0; c_err <= 1'b1; end
                            endcase
                        end
                        cst <= c_we ? C_B : C_R;
                    end else if (c_gnt) begin
                        cst <= C_RESP;
                    end
                end

                C_RESP: begin
                    c_rdata <= bus_rdata;
                    c_err   <= bus_err;
                    cst     <= c_we ? C_B : C_R;
                end

                C_B: if (s_axi_cfg_bready) begin
                    have_aw <= 1'b0; have_w <= 1'b0; cst <= C_IDLE;
                end

                C_R: if (s_axi_cfg_rready) begin
                    have_ar <= 1'b0; cst <= C_IDLE;
                end

                default: cst <= C_IDLE;
            endcase
        end
    end

    // ------------------------------------------------------------------
    // MEM port: AXI4 slave FSM (one burst at a time)
    // ------------------------------------------------------------------
    localparam [3:0] M_IDLE = 4'd0,  M_WDATA = 4'd1, M_WRD  = 4'd2, M_WRDRESP = 4'd3,
                     M_WREQ = 4'd4,  M_WRESP = 4'd5, M_WNEXT = 4'd6, M_B = 4'd7,
                     M_RREQ = 4'd8,  M_RRESP = 4'd9, M_ROUT  = 4'd10;
    reg [3:0]        mst;
    reg [ID_W-1:0]   m_id;
    reg [MEM_AW-1:0] m_addr;
    reg [7:0]        m_left;          // beats remaining after the current one
    reg [1:0]        m_burst;
    reg              m_err;
    reg [31:0]       m_wdata, m_rdata;
    reg [3:0]        m_wstrb;

    wire m_idle = (mst == M_IDLE) & ~rst;
    assign s_axi_mem_awready = m_idle;
    assign s_axi_mem_arready = m_idle & ~s_axi_mem_awvalid;      // writes first
    assign s_axi_mem_wready  = (mst == M_WDATA);
    assign s_axi_mem_bvalid  = (mst == M_B);
    assign s_axi_mem_bid     = m_id;
    assign s_axi_mem_bresp   = m_err ? SLVERR : OKAY;
    assign s_axi_mem_rvalid  = (mst == M_ROUT);
    assign s_axi_mem_rid     = m_id;
    assign s_axi_mem_rdata   = m_rdata;
    assign s_axi_mem_rresp   = m_err ? SLVERR : OKAY;
    assign s_axi_mem_rlast   = (m_left == 8'd0);

    wire m_req = (mst == M_WRD) | (mst == M_WREQ) | (mst == M_RREQ);
    wire m_we  = (mst == M_WREQ);
    wire m_gnt = m_req & ~c_req & ~core_rst;

    // merge for partial-strobe writes: strobed bytes from the write data, rest from the SRAM word
    wire [31:0] merged = { m_wstrb[3] ? m_wdata[31:24] : bus_rdata[31:24],
                           m_wstrb[2] ? m_wdata[23:16] : bus_rdata[23:16],
                           m_wstrb[1] ? m_wdata[15:8]  : bus_rdata[15:8],
                           m_wstrb[0] ? m_wdata[7:0]   : bus_rdata[7:0] };

    wire [MEM_AW-1:0] m_addr_next = (m_burst == 2'b00) ? m_addr
                                  : {m_addr[MEM_AW-1:2] + 1'b1, 2'b00};

    always @(posedge aclk or posedge rst) begin
        if (rst) begin
            mst <= M_IDLE; m_id <= {ID_W{1'b0}}; m_addr <= {MEM_AW{1'b0}};
            m_left <= 8'd0; m_burst <= 2'b01; m_err <= 1'b0;
            m_wdata <= 32'd0; m_rdata <= 32'd0; m_wstrb <= 4'd0;
        end else begin
            case (mst)
                M_IDLE: begin
                    if (s_axi_mem_awvalid) begin
                        m_id <= s_axi_mem_awid;  m_addr <= s_axi_mem_awaddr;
                        m_left <= s_axi_mem_awlen; m_burst <= s_axi_mem_awburst;
                        m_err <= s_axi_mem_awburst[1];                // WRAP / reserved
                        mst <= M_WDATA;
                    end else if (s_axi_mem_arvalid) begin
                        m_id <= s_axi_mem_arid;  m_addr <= s_axi_mem_araddr;
                        m_left <= s_axi_mem_arlen; m_burst <= s_axi_mem_arburst;
                        m_err <= s_axi_mem_arburst[1];
                        mst <= M_RREQ;
                    end
                end

                // ---------------- write burst ----------------
                M_WDATA: if (s_axi_mem_wvalid) begin
                    m_wdata <= s_axi_mem_wdata;
                    m_wstrb <= s_axi_mem_wstrb;
                    if (s_axi_mem_wstrb == 4'hF)      mst <= M_WREQ;
                    else if (s_axi_mem_wstrb == 4'h0) mst <= M_WNEXT;
                    else                              mst <= M_WRD;   // read-modify-write
                end
                M_WRD:     if (m_gnt) mst <= M_WRDRESP;
                M_WRDRESP: begin m_wdata <= merged; mst <= M_WREQ; end
                M_WREQ:    if (m_gnt) mst <= M_WRESP;
                M_WRESP:   mst <= M_WNEXT;
                M_WNEXT: begin
                    if (m_left == 8'd0) mst <= M_B;
                    else begin
                        m_left <= m_left - 8'd1;
                        m_addr <= m_addr_next;
                        mst    <= M_WDATA;
                    end
                end
                M_B: if (s_axi_mem_bready) mst <= M_IDLE;

                // ---------------- read burst ----------------
                M_RREQ:  if (m_gnt) mst <= M_RRESP;
                M_RRESP: begin m_rdata <= bus_rdata; mst <= M_ROUT; end
                M_ROUT:  if (s_axi_mem_rready) begin
                    if (m_left == 8'd0) mst <= M_IDLE;
                    else begin
                        m_left <= m_left - 8'd1;
                        m_addr <= m_addr_next;
                        mst    <= M_RREQ;
                    end
                end

                default: mst <= M_IDLE;
            endcase
        end
    end

    // ------------------------------------------------------------------
    // Arbiter: config port wins, SRAM port waits (config accesses are rare)
    // ------------------------------------------------------------------
    wire [TOP_AW-1:0] cfg_baddr = {1'b1, {(MEM_BYTE_AW-12){1'b0}}, c_addr};
    wire [TOP_AW-1:0] mem_baddr = {1'b0, m_addr[MEM_BYTE_AW-1:0]};

    assign bus_req   = (c_req | m_req) & ~core_rst;
    assign bus_we    = c_req ? c_we    : m_we;
    assign bus_addr  = c_req ? cfg_baddr : mem_baddr;
    assign bus_wdata = c_req ? c_wdata : m_wdata;

endmodule