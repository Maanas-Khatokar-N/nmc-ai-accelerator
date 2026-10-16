// ============================================================================
// Compute sequencer  (address generator, part 2 of 4)
//
// Loop order (innermost first):  kernel col -> kernel row -> input channel.
// All input channels of one output channel are reduced inside the PE
// One op per cycle:
//
//   weight_rf_addr : index into the weight RF (counts 0 .. Cin*Kr*Kc-1, which
//                    matches the [ic][kr][kc] weight layout)
//   ifmap_rf_addr  : 16 read addresses into the IFMap RF (local patch), one per PE
//
// Patch layout in the IFMap RF:  [channel][row][col], row pitch = patch_cols.
// PE (pr,pc) at kernel element (kr,kc) of channel ic reads
//     ic*rf_ch_stride + (pr*S + kr)*patch_cols + (pc*S + kc)
//   = bp + pr*rf_row_step + pc*S          with  bp = ic*rf_ch_stride + kr*patch_cols + kc
//
// bp is an incremental pointer, the per-lane offsets are constant multiples of
// rf_row_step / stride (shift-add), so there are no runtime multipliers and no divider.
//
// Output handshake (valid / enable):
//   Outputs are REGISTERED. valid=1 means ifmap_rf_addr / weight_rf_addr / last
//   describe the op for THIS cycle. The consumer asserts `enable` in the cycle it
//   consumes the op. While enable=0 the outputs hold (stall).
//   Because the RF read is combinational, the PEs see the operands in the same
//   cycle as valid, so no extra delay is needed on mac_en as long as the
//   access controller does not add a pipeline stage between RF and PE. If it
//   does, delay mac_en / mask / clear / psum_load by the same number of cycles.
//
//   done pulses (1 cycle) in the cycle after the last op was consumed.
// ============================================================================
module compute_sequencer #(
    parameter DIM_WIDTH = 16,
    parameter RF_AW     = 8,
    parameter NUM_PE    = 16,
    parameter PE_ROWS   = 4,
    parameter PE_COLS   = 4
)(
    input  wire                      clk,
    input  wire                      rst,
    input  wire                      start,
    input  wire                      enable,     // consumer accepts the current op

    // layer descriptor
    input  wire [DIM_WIDTH-1:0]      kernel_rows,
    input  wire [DIM_WIDTH-1:0]      kernel_cols,
    input  wire [DIM_WIDTH-1:0]      input_channels,
    input  wire [DIM_WIDTH-1:0]      stride,
    input  wire [RF_AW-1:0]          patch_cols,     // RF row pitch
    input  wire [RF_AW-1:0]          rf_row_step,    // stride * patch_cols
    input  wire [RF_AW-1:0]          rf_ch_stride,   // patch_rows * patch_cols

    output reg  [NUM_PE*RF_AW-1:0]   ifmap_rf_addr,
    output reg  [RF_AW-1:0]          weight_rf_addr,
    output reg                       valid,
    output reg                       last,           // last op of this output channel / tile
    output reg                       done
);

    reg [DIM_WIDTH-1:0] kc, kr, ic;
    reg [RF_AW-1:0]     bp;          // patch index of PE(0,0)'s operand
    reg [RF_AW-1:0]     ch_base;     // patch index of channel ic, row 0, col 0
    reg [RF_AW-1:0]     widx;
    reg                 active;      // counters point at an op that has not been emitted yet

    wire kc_last = (kc == kernel_cols     - 1);
    wire kr_last = (kr == kernel_rows     - 1);
    wire ic_last = (ic == input_channels  - 1);
    wire op_last = kc_last & kr_last & ic_last;

    // bp increment when kc wraps and kr advances: patch_cols - (Kc-1)
    wire [RF_AW-1:0] row_wrap_inc = patch_cols - kernel_cols[RF_AW-1:0] + 1'b1;

    // 16 lane addresses = bp + constant-multiple offsets
    wire [NUM_PE*RF_AW-1:0] lane_addr_next;
    genvar g;
    generate for (g = 0; g < NUM_PE; g = g + 1) begin : gen_lane
        wire [RF_AW-1:0] off = (g / PE_COLS) * rf_row_step + (g % PE_COLS) * stride[RF_AW-1:0];
        assign lane_addr_next[g*RF_AW +: RF_AW] = bp + off;
    end endgenerate

    wire load_out = active && (!valid || enable);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            kc <= 0; kr <= 0; ic <= 0;
            bp <= 0; ch_base <= 0; widx <= 0;
            active <= 1'b0;
            ifmap_rf_addr  <= 0;
            weight_rf_addr <= 0;
            valid <= 1'b0; last <= 1'b0; done <= 1'b0;
        end else begin
            done <= 1'b0;

            if (start) begin
                kc <= 0; kr <= 0; ic <= 0;
                bp <= 0; ch_base <= 0; widx <= 0;
                active <= 1'b1;
                valid  <= 1'b0;
                last   <= 1'b0;
            end else begin
                // current op consumed
                if (valid && enable) begin
                    valid <= 1'b0;
                    if (last) done <= 1'b1;
                end

                // emit next op (may overwrite valid <= 0 above)
                if (load_out) begin
                    ifmap_rf_addr  <= lane_addr_next;
                    weight_rf_addr <= widx;
                    valid          <= 1'b1;
                    last           <= op_last;
                    widx           <= widx + 1'b1;

                    if (!kc_last) begin
                        kc <= kc + 1'b1;
                        bp <= bp + 1'b1;
                    end else begin
                        kc <= 0;
                        if (!kr_last) begin
                            kr <= kr + 1'b1;
                            bp <= bp + row_wrap_inc;
                        end else begin
                            kr <= 0;
                            if (!ic_last) begin
                                ic      <= ic + 1'b1;
                                bp      <= ch_base + rf_ch_stride;
                                ch_base <= ch_base + rf_ch_stride;
                            end else begin
                                ic     <= 0;
                                active <= 1'b0;
                            end
                        end
                    end
                end
            end
        end
    end

endmodule