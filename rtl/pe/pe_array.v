// ============================================================================
// PE array -- 16 PEs, 4x4 grid, PE i = row i/4, col i%4
//
// psum_load / psum_in : per-PE partial-sum reload (OFMap -> PE)
// weight broadcast    : WEIGHT_BROADCAST=1 -> ONE weight feeds all PEs
//                       (weight-unicast, output-stationary dataflow of the
//                       paper: each cycle 1 kernel weight, 16 IFMap pixels)
//                       WEIGHT_BROADCAST=0 -> each PE gets its own weight
//                       (useful for fully-connected / weight-parallel layers)
// pe_mask             : per-PE enable bit for edge tiles
//
// Packing convention: PE i occupies bits [i*W +: W] of every packed bus.
// ============================================================================
module pe_array #(
    parameter NUM_PE            = 16,
    parameter DATA_WIDTH        = 8,
    parameter ACC_WIDTH         = 32,
    parameter WEIGHT_BROADCAST  = 1,
    parameter TOTAL_DATA_WIDTH  = NUM_PE*DATA_WIDTH,
    parameter TOTAL_ACC_WIDTH   = NUM_PE*ACC_WIDTH,
    // 1 weight when broadcasting, NUM_PE weights otherwise
    parameter WEIGHT_WIDTH      = WEIGHT_BROADCAST ? DATA_WIDTH : NUM_PE*DATA_WIDTH
)(
    input  wire                                clk,
    input  wire                                rst,
    input  wire                                clear,      // clear all accumulators
    input  wire [NUM_PE-1:0]                   pe_mask,    // per-PE active bit
    input  wire                                psum_load,  // reload psums (global)
    input  wire                                mac_en,     // accumulate (global)
    input  wire signed [TOTAL_ACC_WIDTH-1:0]   psum_in,    // 16 psums from OFMap
    input  wire signed [TOTAL_DATA_WIDTH-1:0]  ifmap_in,   // 16 IFMap pixels
    input  wire signed [WEIGHT_WIDTH-1:0]      weight_in,  // 1 (broadcast) or 16 weights
    output wire signed [TOTAL_ACC_WIDTH-1:0]   acc_out     // 16 accumulators
);

    genvar i;
    generate
        for (i = 0; i < NUM_PE; i = i + 1) begin : gen_pe

            // Pick this PE's weight: shared one, or its own slice
            wire signed [DATA_WIDTH-1:0] w_i;
            if (WEIGHT_BROADCAST) begin : g_bcast
                assign w_i = weight_in;                         // same for all PEs
            end else begin : g_unicast
                assign w_i = weight_in[i*DATA_WIDTH +: DATA_WIDTH];
            end

            pe #(
                .DATA_WIDTH(DATA_WIDTH),
                .ACC_WIDTH (ACC_WIDTH)
            ) pe_inst (
                .clk      (clk),
                .rst      (rst),
                .clear    (clear),
                .mask     (pe_mask[i]),
                .psum_load(psum_load),
                .mac_en   (mac_en),
                .psum_in  (psum_in [i*ACC_WIDTH  +: ACC_WIDTH]),
                .ifmap_in (ifmap_in[i*DATA_WIDTH +: DATA_WIDTH]),
                .weight_in(w_i),
                .acc_out  (acc_out [i*ACC_WIDTH  +: ACC_WIDTH])
            );
        end
    endgenerate

endmodule