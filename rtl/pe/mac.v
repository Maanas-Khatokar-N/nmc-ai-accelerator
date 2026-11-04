// ============================================================================
// MAC unit  (FPGA version: synchronous reset so the accumulator can live in a DSP48)
//
// Operation priority (highest -> lowest):
//   1. rst / clear : acc = 0   (rst is now SYNCHRONOUS: hold it for at least one clock edge;
//                               nmc_axi_wrapper holds it for >= 2 cycles)
//   2. load_en     : acc = load_data               (reload a stored psum from the OFMap macros)
//   3. mac_en      : acc = acc + a*b               (next kernel multiplication)
//   otherwise      : acc holds its value
//
// "mask" is applied by the PE (it gates load_en and mac_en), so this module stays a plain MAC.
// Interface and cycle behaviour are unchanged. An asynchronous reset on the accumulator is the
// one thing the DSP48E1 cannot absorb, which is why Vivado used to build this from LUTs.
// ============================================================================
(* use_dsp = "yes" *)
module mac #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire                         clk,
    input  wire                         rst,        // sync reset
    input  wire                         clear,      // sync clear accumulator
    input  wire                         load_en,    // load partial sum
    input  wire                         mac_en,     // multiply-accumulate
    input  wire signed [ACC_WIDTH-1:0]  load_data,  // psum coming from OFMap macro
    input  wire signed [DATA_WIDTH-1:0] a,          // IFMap pixel
    input  wire signed [DATA_WIDTH-1:0] b,          // kernel weight
    output reg  signed [ACC_WIDTH-1:0]  acc_out
);

    wire signed [2*DATA_WIDTH-1:0] product = a * b;

    always @(posedge clk) begin
        if (rst | clear)
            acc_out <= {ACC_WIDTH{1'b0}};
        else if (load_en)
            acc_out <= load_data;
        else if (mac_en)
            acc_out <= acc_out + product;       // product is sign-extended
    end

endmodule
