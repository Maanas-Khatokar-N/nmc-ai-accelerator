// ============================================================================
// Psum register file (ping-pong), purpose-built replacement for
//   register_file_pp #(.DATA_WIDTH(32), .DEPTH(16), .NUM_READ_PORTS(16), .NUM_WRITE_PORTS(8))
// in nmc_unit.  Same cycle behaviour for every word the datapath actually uses, about 1/6 of the LUTs.
//
//   word i of a half holds PE i's partial sum; read port i always reads word i.
//   half selects / flush follow the weight RF (flush_sel, wr_sel, rd_sel), like before.
//
//   write  : NUM_MACROS lanes (reload generator, one cycle after the grant).
//            lane m writes word write_addr[m] with write_data[m]  -> V = 1
//   chk    : chk_free[m] = V of word chk_addr[m] in the WRITE half is 0 (request-time check)
//   read   : read_valid[i] = V of word i in the READ half;
//            read_en[i] & read_last[i] & V consumes the word (V -> 0)
//   flush  : clears V of the flush_sel half (data is not touched)
//
// What was removed (nothing here is ever observed by nmc_unit):
//   - write_blocked / overwrite protection: a word is written at most once after its flush,
//     and chk_free already keeps the reload generator off live words.
//   - D bits.
//   - zero-masking of read_data when V = 0. The PEs only take psum_in under psum_load & pe_mask,
//     and psum_load waits for V = 1 on every active lane, so stale data never reaches an accumulator.
//   - the per-word 8:1 write mux. Lane address = row*PE_COLS + col and the lanes written in one
//     cycle have distinct col, so ONE mux per column (PE_COLS x 8:1) feeds the words of all rows.
//
// CONTRACT (holds for store_reload_generator in reload mode):
//   - lanes written in the same cycle have distinct (write_addr % PE_COLS)
//   - a word is written only while its V = 0
// PE_COLS must be a power of two.
// ============================================================================
module psum_rf_pp #(
    parameter NUM_PE     = 16,
    parameter DATA_WIDTH = 32,
    parameter NUM_MACROS = 8,
    parameter PE_COLS    = 4,
    parameter LANE_W     = $clog2(NUM_PE),
    parameter COL_W      = $clog2(PE_COLS)
) (
    input  wire                             clk,
    input  wire                             rst,
    input  wire                             flush,
    input  wire                             flush_sel,
    input  wire                             wr_sel,
    input  wire                             rd_sel,

    input  wire [NUM_MACROS-1:0]            write_en,
    input  wire [NUM_MACROS*LANE_W-1:0]     write_addr,
    input  wire [NUM_MACROS*DATA_WIDTH-1:0] write_data,

    input  wire [NUM_MACROS*LANE_W-1:0]     chk_addr,
    output wire [NUM_MACROS-1:0]            chk_free,

    input  wire [NUM_PE-1:0]                read_en,
    input  wire [NUM_PE-1:0]                read_last,
    output wire [NUM_PE*DATA_WIDTH-1:0]     read_data,
    output wire [NUM_PE-1:0]                read_valid
);

    // ---------------- write path: one mux per column, shared by rows and halves ----------------
    reg [PE_COLS*DATA_WIDTH-1:0] col_data;
    reg [NUM_PE-1:0]             word_wr;          // some lane writes word i this cycle
    integer m, c, i;
    always @(*) begin
        col_data = {(PE_COLS*DATA_WIDTH){1'b0}};
        word_wr  = {NUM_PE{1'b0}};
        for (m = 0; m < NUM_MACROS; m = m + 1)
            if (write_en[m]) begin
                word_wr[write_addr[m*LANE_W +: LANE_W]] = 1'b1;
                for (c = 0; c < PE_COLS; c = c + 1)
                    if (write_addr[m*LANE_W +: COL_W] == c)
                        col_data[c*DATA_WIDTH +: DATA_WIDTH] =
                            col_data[c*DATA_WIDTH +: DATA_WIDTH] | write_data[m*DATA_WIDTH +: DATA_WIDTH];
            end
    end

    // ---------------- the two halves ----------------
    reg  [NUM_PE-1:0]           v_a, v_b;
    reg  [DATA_WIDTH-1:0]       d_a [0:NUM_PE-1];
    reg  [DATA_WIDTH-1:0]       d_b [0:NUM_PE-1];

    wire fl_a = flush & ~flush_sel;
    wire fl_b = flush &  flush_sel;
    wire wr_a = ~wr_sel, wr_b = wr_sel;
    wire rd_a = ~rd_sel, rd_b = rd_sel;

    // valid bits: read-consume, then write (write wins), flush clears
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            v_a <= {NUM_PE{1'b0}};
            v_b <= {NUM_PE{1'b0}};
        end else begin
            if (fl_a) v_a <= {NUM_PE{1'b0}};
            else      v_a <= (v_a & ~({NUM_PE{rd_a}} & read_en & read_last)) | ({NUM_PE{wr_a}} & word_wr);
            if (fl_b) v_b <= {NUM_PE{1'b0}};
            else      v_b <= (v_b & ~({NUM_PE{rd_b}} & read_en & read_last)) | ({NUM_PE{wr_b}} & word_wr);
        end
    end

    // data: plain flip-flops with clock enable, no reset
    genvar g;
    generate for (g = 0; g < NUM_PE; g = g + 1) begin : g_w
        always @(posedge clk) begin
            if (wr_a && !fl_a && word_wr[g]) d_a[g] <= col_data[(g % PE_COLS)*DATA_WIDTH +: DATA_WIDTH];
            if (wr_b && !fl_b && word_wr[g]) d_b[g] <= col_data[(g % PE_COLS)*DATA_WIDTH +: DATA_WIDTH];
        end
        assign read_data[g*DATA_WIDTH +: DATA_WIDTH] = rd_sel ? d_b[g] : d_a[g];
    end endgenerate

    assign read_valid = rd_sel ? v_b : v_a;

    // ---------------- request-time free check on the write half ----------------
    wire [NUM_PE-1:0] v_wr = wr_sel ? v_b : v_a;
    genvar k;
    generate for (k = 0; k < NUM_MACROS; k = k + 1) begin : g_chk
        assign chk_free[k] = ~v_wr[chk_addr[k*LANE_W +: LANE_W]];
    end endgenerate

endmodule
