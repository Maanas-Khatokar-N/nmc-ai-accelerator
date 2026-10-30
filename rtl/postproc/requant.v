// ============================================================================
// Requantizer + ReLU   (3-stage pipeline, one result per cycle)
//
//   y = clamp( round_shift( (x + bias) * mult , shift ) + zp , lo , 127 )
//
//   round_shift(v, s) = (v + (1 << (s-1))) >>> s     (round half up, s = 0: no rounding)
//   lo = relu_en ? zp : -128        ReLU in the quantized domain: real 0 == zp
//
//   mult is UNSIGNED (per-channel scale >= 0). That makes the whole chain monotone
//   non-decreasing in x, which is what allows max-pool to run BEFORE requant
//   (max(f(a),f(b)) == f(max(a,b))): one requant per pooled output instead of four.
//
//   stage 1: xb   = x + bias                  (ACC_W+1 bits, cannot overflow)
//   stage 2: prod = xb * mult                 (ACC_W+MUL_W+2 bits, cannot overflow)
//   stage 3: round, shift, + zp, clamp -> y
//   zp / relu_en are static while a layer runs (not pipelined).
// ============================================================================
module requant #(
    parameter ACC_W = 32,
    parameter MUL_W = 16,
    parameter SH_W  = 5
)(
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     in_valid,
    input  wire signed [ACC_W-1:0]  x,
    input  wire signed [ACC_W-1:0]  bias,
    input  wire        [MUL_W-1:0]  mult,
    input  wire        [SH_W-1:0]   shift,
    input  wire signed [7:0]        zp,
    input  wire                     relu_en,
    output reg                      out_valid,
    output reg  signed [7:0]        y
);
    localparam PW = ACC_W + MUL_W + 2;

    reg                     v1, v2;
    reg signed [ACC_W:0]    xb;
    reg        [MUL_W-1:0]  m1;
    reg        [SH_W-1:0]   s1, s2;
    reg signed [PW-1:0]     prod;

    wire signed [ACC_W:0]   sum = x + bias;

    wire signed [PW-1:0] rnd  = (s2 == 0) ? {PW{1'b0}} : ({{(PW-1){1'b0}}, 1'b1} << (s2 - 1));
    wire signed [PW-1:0] rsum = prod + rnd;
    wire signed [PW-1:0] q    = rsum >>> s2;
    wire signed [PW-1:0] z    = q + zp;
    wire signed [PW-1:0] lo   = relu_en ? zp : -128;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            v1 <= 1'b0; v2 <= 1'b0; out_valid <= 1'b0;
        end else begin
            v1        <= in_valid;
            v2        <= v1;
            out_valid <= v2;
        end
    end

    always @(posedge clk) begin
        xb   <= sum;   m1 <= mult;  s1 <= shift;
        prod <= xb * $signed({1'b0, m1});   s2 <= s1;
        y    <= (z < lo) ? lo[7:0] : (z > 127) ? 8'sd127 : z[7:0];
    end
endmodule