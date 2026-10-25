// ============================================================================
// Access controller  -- owner of all three grants
//
//   COMPUTE : MAC fires when all ACTIVE operands are valid
//   LOAD    : a lane's request reaches the SRAM only if the RF word it will
//             write is FREE (V=0) and the access window is open.
//             The check is made at REQUEST time (read data returns one cycle
//             after the grant, so a late write_blocked would lose the word).
//   STORE   : a store request reaches the SRAM when the window is open; it is
//             GRANTED when the macro port is free (not taken by the core).
//   Per macro, STORE wins over LOAD (same rule the old arbiter in
//   address_generator used).
//
// Access window (paper Sec. III-D: access_allowed / access_not_allowed):
//   win_block_cycles == 0 : always open (default)
//   otherwise             : open for win_allow_cycles, closed for win_block_cycles, repeat.
//   Only memory requests are gated; compute never touches the SRAM port.
//
// Status (paper's memory control states):
//   mem_access_allowed  = NMC_memory_access_allowed
//   mem_blocked_cfg     = NMC_blocked_by_configuration
//   mem_blocked_core    = NMC_blocked_by_core_access
//   ld_blocked_busy     = a load lane held back because its RF word is still live
// ============================================================================
module access_controller #(
    parameter NUM_PE     = 16,
    parameter NUM_MACROS = 8,
    parameter MACRO_AW   = 12,
    parameter WIN_W      = 16
)(
    input  wire              clk,
    input  wire              rst,

    // ================= COMPUTE channel (unchanged) =================
    input  wire              compute_start,
    input  wire              seq_valid,
    input  wire              seq_last,
    input  wire [NUM_PE-1:0] pe_mask,
    output wire              seq_enable,
    output reg               compute_done,

    input  wire [NUM_PE-1:0] ifrf_read_valid,
    input  wire              wrf_read_valid,
    output wire [NUM_PE-1:0] ifrf_read_en,
    output wire              wrf_read_en,

    input  wire              psum_req,
    input  wire [NUM_PE-1:0] psrf_read_valid,
    output wire [NUM_PE-1:0] psrf_read_en,
    output wire [NUM_PE-1:0] psrf_read_last,
    output wire              psum_ack,

    output reg               pe_clear,
    output wire              pe_mac_en,
    output wire              pe_psum_load,
    output wire              stall,

    // ================= LOAD channel (from load_generator) =================
    input  wire [NUM_MACROS-1:0]          ld_req,         // raw per-macro requests
    input  wire [NUM_MACROS*MACRO_AW-1:0] ld_addr,
    input  wire                           ld_is_weight,   // 1: writes weight RF, 0: IFMap RF
    input  wire [NUM_MACROS-1:0]          ifrf_dst_free,  // V=0 at the lane's destination word
    input  wire [NUM_MACROS-1:0]          wrf_dst_free,
    output wire [NUM_MACROS-1:0]          ld_grant,

    // ================= STORE channel (from store_reload_generator) =================
    input  wire [NUM_MACROS-1:0]          st_req,
    input  wire [NUM_MACROS-1:0]          st_we,
    input  wire [NUM_MACROS*MACRO_AW-1:0] st_addr,
    output wire [NUM_MACROS-1:0]          st_grant,

    // ================= SRAM bank NMC port =================
    output wire [NUM_MACROS-1:0]          nmc_req,
    output wire [NUM_MACROS-1:0]          nmc_we,
    output wire [NUM_MACROS*MACRO_AW-1:0] nmc_addr,
    input  wire [NUM_MACROS-1:0]          nmc_grant,

    // ================= access window =================
    input  wire [WIN_W-1:0]  win_allow_cycles,
    input  wire [WIN_W-1:0]  win_block_cycles,

    output wire              mem_access_allowed,
    output wire              mem_blocked_cfg,
    output wire              mem_blocked_core,
    output wire              ld_blocked_busy
);

    // ------------------------------------------------------------------
    // COMPUTE / PSUM  (identical to the version that passes milestone 1)
    // ------------------------------------------------------------------
    wire if_ok   = &(ifrf_read_valid  | ~pe_mask);
    wire ps_ok   = &(psrf_read_valid  | ~pe_mask);
    wire w_ok    = wrf_read_valid;

    wire mac_fire  = seq_valid & if_ok & w_ok & ~psum_req;
    wire psum_fire = psum_req & ps_ok;

    assign pe_mac_en      = mac_fire;
    assign seq_enable     = mac_fire;
    assign ifrf_read_en   = pe_mask & {NUM_PE{mac_fire}};
    assign wrf_read_en    = mac_fire;

    assign pe_psum_load   = psum_fire;
    assign psum_ack       = psum_fire;
    assign psrf_read_en   = pe_mask & {NUM_PE{psum_fire}};
    assign psrf_read_last = psrf_read_en;

    assign stall          = seq_valid & ~mac_fire;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            compute_done <= 1'b0;
            pe_clear     <= 1'b0;
        end else begin
            compute_done <= mac_fire & seq_last;
            pe_clear     <= compute_start;
        end
    end

    // ------------------------------------------------------------------
    // Access window
    // ------------------------------------------------------------------
    reg  [WIN_W-1:0] win_cnt;
    reg              win_open;
    wire             win_free_run = (win_block_cycles == {WIN_W{1'b0}});
    wire [WIN_W:0]   cnt_n        = {1'b0, win_cnt} + 1'b1;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            win_cnt  <= {WIN_W{1'b0}};
            win_open <= 1'b1;
        end else if (win_free_run) begin
            win_cnt  <= {WIN_W{1'b0}};
            win_open <= 1'b1;
        end else if (win_open) begin
            if (cnt_n >= {1'b0, win_allow_cycles}) begin
                win_cnt  <= {WIN_W{1'b0}};
                win_open <= 1'b0;
            end else win_cnt <= cnt_n[WIN_W-1:0];
        end else begin
            if (cnt_n >= {1'b0, win_block_cycles}) begin
                win_cnt  <= {WIN_W{1'b0}};
                win_open <= 1'b1;
            end else win_cnt <= cnt_n[WIN_W-1:0];
        end
    end

    assign mem_access_allowed = win_open;

    // ------------------------------------------------------------------
    // LOAD / STORE grants and the per-macro arbiter
    // ------------------------------------------------------------------
    wire [NUM_MACROS-1:0] win_vec   = {NUM_MACROS{win_open}};
    wire [NUM_MACROS-1:0] ld_free   = ld_is_weight ? wrf_dst_free : ifrf_dst_free;

    wire [NUM_MACROS-1:0] ld_req_ok = ld_req & ld_free & win_vec;   // word free + window open
    wire [NUM_MACROS-1:0] st_req_ok = st_req & win_vec;             // window open

    assign nmc_req  = ld_req_ok | st_req_ok;
    assign nmc_we   = st_req_ok & st_we;

    genvar m;
    generate for (m = 0; m < NUM_MACROS; m = m + 1) begin : g_arb
        assign nmc_addr[m*MACRO_AW +: MACRO_AW] = st_req_ok[m] ? st_addr[m*MACRO_AW +: MACRO_AW]
                                                               : ld_addr[m*MACRO_AW +: MACRO_AW];
    end endgenerate

    // nmc_grant already carries the core-conflict result from sram_bank
    assign st_grant = nmc_grant & st_req_ok;
    assign ld_grant = nmc_grant & ld_req_ok & ~st_req_ok;

    // status
    assign mem_blocked_cfg  = ~win_open & (|(ld_req | st_req));
    assign mem_blocked_core = |(nmc_req & ~nmc_grant);
    assign ld_blocked_busy  = win_open & (|(ld_req & ~ld_free));

endmodule