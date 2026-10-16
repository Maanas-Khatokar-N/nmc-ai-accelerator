// ============================================================================
// Store / reload generator  (address generator, part 4 of 4)
//
//   reload = 0 : STORE  PE accumulators -> OFMap macros
//   reload = 1 : RELOAD OFMap macros    -> psum RF (feeds psum_in / psum_load)
//
// One PE row (up to PE_COLS consecutive OFMap words) is handled per step, so the
// words of a step always fall in distinct macros (PE_COLS <= NUM_MACROS) and a
// row can never conflict with itself. Rows beyond valid_rows and columns beyond
// valid_cols (partial edge tiles) are skipped.
//
// For macro m, the word it serves in the current row is  w0 + c  with
// c = (m - w0[2:0]) mod 8. The PE lane involved is  row*PE_COLS + c:
//   STORE : st_lane_sel[m] tells the datapath which accumulator drives macro m's
//           write data (same cycle as the grant).
//   RELOAD: one cycle after the grant, rl_wr_en[m] / rl_wr_lane[m] say "write
//           nmc_rdata[m] into psum RF address rl_wr_lane[m]" (psum RF word i
//           holds PE i's psum).
//
// Requests lost to the core stay pending and are retried (a lane is never
// served twice). done pulses when the last write has landed (store) or the last
// RF write has been issued and applied (reload).
// ============================================================================
module store_reload_generator #(
    parameter ADDR_WIDTH = 17,
    parameter DIM_WIDTH  = 16,
    parameter NUM_MACROS = 8,
    parameter MACRO_AW   = 12,
    parameter PE_ROWS    = 4,
    parameter PE_COLS    = 4,
    parameter NUM_PE     = PE_ROWS * PE_COLS,
    parameter LANE_W     = $clog2(NUM_PE),
    parameter VR_W       = $clog2(PE_ROWS + 1),
    parameter VC_W       = $clog2(PE_COLS + 1)
)(
    input  wire                              clk,
    input  wire                              rst,
    input  wire                              start,
    input  wire                              reload,

    input  wire [DIM_WIDTH-1:0]              ofmap_cols,
    input  wire [ADDR_WIDTH-1:0]             ofmap_tile_addr, // word addr of tile top-left, THIS output channel
    input  wire [VR_W-1:0]                   valid_rows,
    input  wire [VC_W-1:0]                   valid_cols,

    // NMC memory port
    output wire [NUM_MACROS-1:0]             nmc_req,
    output wire [NUM_MACROS-1:0]             nmc_we,
    output wire [NUM_MACROS*MACRO_AW-1:0]    nmc_addr,
    input  wire [NUM_MACROS-1:0]             nmc_grant,

    // store data routing
    output wire [NUM_MACROS*LANE_W-1:0]      st_lane_sel,

    // reload -> psum RF
    output wire [NUM_MACROS-1:0]             rl_wr_en,
    output wire [NUM_MACROS*LANE_W-1:0]      rl_wr_lane,

    output reg                               busy,
    output reg                               done
);

    reg                       mode_reload;
    reg                       fresh;
    reg [NUM_MACROS-1:0]      pend;
    reg [ADDR_WIDTH-1:0]      row_addr;        // word address of column 0 of the current PE row
    reg [DIM_WIDTH-1:0]       row_cnt;
    reg [LANE_W-1:0]          lane_base;       // row * PE_COLS
    reg [VR_W-1:0]            vrows_q;
    reg [VC_W-1:0]            vcols_q;
    reg                       fin1;

    reg [NUM_MACROS-1:0]      rl_en_q;
    reg [NUM_MACROS*LANE_W-1:0] rl_lane_q;

    // per-macro decode of the current row
    reg [NUM_MACROS-1:0]        m_valid;
    reg [NUM_MACROS*MACRO_AW-1:0] m_addr;
    reg [NUM_MACROS*LANE_W-1:0] m_lane;
    reg [2:0]  c;
    reg [3:0]  s;
    integer m;
    wire [2:0] w0lo = row_addr[2:0];
    always @(*) begin
        for (m = 0; m < NUM_MACROS; m = m + 1) begin
            c = m[2:0] - w0lo;
            s = {1'b0, w0lo} + {1'b0, c};
            m_valid[m] = (c < vcols_q);
            m_addr[m*MACRO_AW +: MACRO_AW] = row_addr[ADDR_WIDTH-1:3] + s[3];
            m_lane[m*LANE_W +: LANE_W]     = lane_base + c;
        end
    end

    wire [NUM_MACROS-1:0] cur_pend = fresh ? m_valid : pend;
    assign nmc_req     = busy ? cur_pend : {NUM_MACROS{1'b0}};
    assign nmc_we      = {NUM_MACROS{~mode_reload}};
    assign nmc_addr    = m_addr;
    assign st_lane_sel = m_lane;
    assign rl_wr_en    = rl_en_q;
    assign rl_wr_lane  = rl_lane_q;

    wire [NUM_MACROS-1:0] granted = nmc_req & nmc_grant;
    wire [NUM_MACROS-1:0] rem     = cur_pend & ~nmc_grant;
    wire more_rows = (row_cnt + 1 < vrows_q);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            busy <= 1'b0; done <= 1'b0; fin1 <= 1'b0;
            mode_reload <= 1'b0; fresh <= 1'b0; pend <= 0;
            row_addr <= 0; row_cnt <= 0; lane_base <= 0; vrows_q <= 0; vcols_q <= 0;
            rl_en_q <= 0; rl_lane_q <= 0;
        end else begin
            done    <= fin1;
            fin1    <= 1'b0;
            rl_en_q <= 0;

            if (start && !busy) begin
                busy        <= 1'b1;
                mode_reload <= reload;
                fresh       <= 1'b1;
                row_addr    <= ofmap_tile_addr;
                row_cnt     <= 0;
                lane_base   <= 0;
                vrows_q     <= valid_rows;
                vcols_q     <= valid_cols;
            end else if (busy) begin
                if (mode_reload) begin
                    rl_en_q   <= granted;
                    rl_lane_q <= m_lane;
                end
                if (rem == 0) begin
                    fresh <= 1'b1;
                    if (more_rows) begin
                        row_cnt   <= row_cnt + 1'b1;
                        row_addr  <= row_addr + ofmap_cols;
                        lane_base <= lane_base + PE_COLS;
                    end else begin
                        busy <= 1'b0;
                        fin1 <= 1'b1;
                    end
                end else begin
                    pend  <= rem;
                    fresh <= 1'b0;
                end
            end
        end
    end

endmodule