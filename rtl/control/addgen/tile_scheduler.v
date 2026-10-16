// ============================================================================
// Tile scheduler  (address generator, part 1 of 4)
//
// Walks the OFMap in PE_ROWS x PE_COLS output tiles (row-major) and reports,
// for the current tile:
//   - tile origin in the OFMap (tile_row / tile_col)
//   - word address of the tile's top-left IFMap input pixel (ifmap_origin)
//   - word address of the tile's top-left OFMap output (ofmap_origin), channel 0
//   - valid_rows / valid_cols and the PE valid mask (partial tiles at the edges)
//
// No multipliers: every address is an incremental pointer. The only
// "multiplies" are by the constants PE_ROWS / PE_COLS (shifts and adds).
//
//   start   : load tile 0, tile_valid = 1
//   advance : move to the next tile; tile_valid drops after the last one
// ============================================================================
module tile_scheduler #(
    parameter ADDR_WIDTH = 17,
    parameter DIM_WIDTH  = 16,
    parameter PE_ROWS    = 4,
    parameter PE_COLS    = 4,
    parameter VR_W       = $clog2(PE_ROWS + 1),
    parameter VC_W       = $clog2(PE_COLS + 1)
)(
    input  wire                          clk,
    input  wire                          rst,
    input  wire                          start,
    input  wire                          advance,

    // layer descriptor (static while a layer runs)
    input  wire [DIM_WIDTH-1:0]          ofmap_rows,
    input  wire [DIM_WIDTH-1:0]          ofmap_cols,
    input  wire [DIM_WIDTH-1:0]          stride,
    input  wire [ADDR_WIDTH-1:0]         ifmap_row_step,   // stride * ifmap_cols
    input  wire [ADDR_WIDTH-1:0]         ifmap_base_addr,
    input  wire [ADDR_WIDTH-1:0]         ofmap_base_addr,

    output reg                           tile_valid,
    output reg  [DIM_WIDTH-1:0]          tile_row,
    output reg  [DIM_WIDTH-1:0]          tile_col,
    output reg  [ADDR_WIDTH-1:0]         ifmap_origin,
    output reg  [ADDR_WIDTH-1:0]         ofmap_origin,
    output wire [VR_W-1:0]               valid_rows,
    output wire [VC_W-1:0]               valid_cols,
    output reg  [PE_ROWS*PE_COLS-1:0]    pe_mask
);

    // origin of column 0 of the current tile row (so a row change needs no multiply)
    reg [ADDR_WIDTH-1:0] ifrow_origin;
    reg [ADDR_WIDTH-1:0] ofrow_origin;

    // constant-coefficient steps
    wire [ADDR_WIDTH-1:0] if_col_step  = PE_COLS * stride;
    wire [ADDR_WIDTH-1:0] if_trow_step = PE_ROWS * ifmap_row_step;
    wire [ADDR_WIDTH-1:0] of_trow_step = PE_ROWS * ofmap_cols;

    wire last_col = (tile_col + PE_COLS >= ofmap_cols);
    wire last_row = (tile_row + PE_ROWS >= ofmap_rows);

    // partial-tile sizes
    wire [DIM_WIDTH-1:0] rem_rows = ofmap_rows - tile_row;
    wire [DIM_WIDTH-1:0] rem_cols = ofmap_cols - tile_col;
    assign valid_rows = (rem_rows >= PE_ROWS) ? PE_ROWS[VR_W-1:0] : rem_rows[VR_W-1:0];
    assign valid_cols = (rem_cols >= PE_COLS) ? PE_COLS[VC_W-1:0] : rem_cols[VC_W-1:0];

    integer i;
    always @(*) begin
        for (i = 0; i < PE_ROWS*PE_COLS; i = i + 1)
            pe_mask[i] = ((i / PE_COLS) < valid_rows) && ((i % PE_COLS) < valid_cols);
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            tile_valid   <= 1'b0;
            tile_row     <= 0;
            tile_col     <= 0;
            ifmap_origin <= 0;
            ofmap_origin <= 0;
            ifrow_origin <= 0;
            ofrow_origin <= 0;
        end else if (start) begin
            tile_valid   <= 1'b1;
            tile_row     <= 0;
            tile_col     <= 0;
            ifmap_origin <= ifmap_base_addr;
            ofmap_origin <= ofmap_base_addr;
            ifrow_origin <= ifmap_base_addr;
            ofrow_origin <= ofmap_base_addr;
        end else if (advance && tile_valid) begin
            if (!last_col) begin
                tile_col     <= tile_col + PE_COLS;
                ifmap_origin <= ifmap_origin + if_col_step;
                ofmap_origin <= ofmap_origin + PE_COLS;
            end else if (!last_row) begin
                tile_col     <= 0;
                tile_row     <= tile_row + PE_ROWS;
                ifrow_origin <= ifrow_origin + if_trow_step;
                ifmap_origin <= ifrow_origin + if_trow_step;
                ofrow_origin <= ofrow_origin + of_trow_step;
                ofmap_origin <= ofrow_origin + of_trow_step;
            end else begin
                tile_valid   <= 1'b0;
            end
        end
    end

endmodule