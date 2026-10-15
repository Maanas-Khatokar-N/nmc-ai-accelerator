// ============================================================================
// MAC unit
//
// Operation priority (highest -> lowest):
//   1. rst       : asynchronous reset, acc = 0
//   2. clear     : synchronous clear,  acc = 0   (start of a new output tile)
//   3. load_en   : acc = load_data               (reload a stored psum from the OFMap macros)
//   4. mac_en    : acc = acc + a*b               (next kernel multiplication)
//   otherwise    : acc holds its value
//
// "mask" is applied by the PE (it gates load_en and mac_en), so this module
// stays a plain MAC.
// ============================================================================
module mac #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire                         clk,
    input  wire                         rst,        // async reset
    input  wire                         clear,      // sync clear accumulator
    input  wire                         load_en,    // load partial sum
    input  wire                         mac_en,     // multiply-accumulate
    input  wire signed [ACC_WIDTH-1:0]  load_data,  // psum coming from OFMap macro
    input  wire signed [DATA_WIDTH-1:0] a,          // IFMap pixel
    input  wire signed [DATA_WIDTH-1:0] b,          // kernel weight
    output reg  signed [ACC_WIDTH-1:0]  acc_out
);

    // Signed product, explicitly sized so sign extension is unambiguous
    wire signed [2*DATA_WIDTH-1:0] product = a * b;

    always @(posedge clk or posedge rst) begin
        if (rst)
            acc_out <= {ACC_WIDTH{1'b0}};
        else if (clear)
            acc_out <= {ACC_WIDTH{1'b0}};
        else if (load_en)                       // load wins over MAC: a reload
            acc_out <= load_data;               // cycle never also accumulates
        else if (mac_en)
            acc_out <= acc_out + product;       // product is sign-extended
    end

endmodule