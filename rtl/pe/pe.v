// ============================================================================
// Processing Element
//
//   mask = 1 : PE is active (normal operation)
//   mask = 0 : PE is disabled. It ignores mac_en and load_en and simply HOLDS
//              its accumulator. Used for edge tiles where the 4x4 PE grid
//              hangs past the edge of the OFMap (e.g. 14x14 / 7x7 layers).
//              'clear' and 'rst' are NOT masked, so a masked PE can still be
//              reset between tiles and never leaks stale data.
//
// ============================================================================
module pe #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire                         clk,
    input  wire                         rst,
    input  wire                         clear,
    input  wire                         mask,       // 1 = active, 0 = hold
    input  wire                         psum_load,  // load psum_in into accumulator
    input  wire                         mac_en,     // accumulate ifmap*weight
    input  wire signed [ACC_WIDTH-1:0]  psum_in,
    input  wire signed [DATA_WIDTH-1:0] ifmap_in,
    input  wire signed [DATA_WIDTH-1:0] weight_in,
    output wire signed [ACC_WIDTH-1:0]  acc_out
);

    mac #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH)
    ) m0 (
        .clk      (clk),
        .rst      (rst),
        .clear    (clear),
        .load_en  (psum_load & mask),   // masked PEs never load
        .mac_en   (mac_en    & mask),   // masked PEs never accumulate
        .load_data(psum_in),
        .a        (ifmap_in),
        .b        (weight_in),
        .acc_out  (acc_out)
    );

endmodule