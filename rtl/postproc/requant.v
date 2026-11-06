// ============================================================================
// Requantizer + ReLU   (6-stage pipeline, one result per cycle)
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
//   stage 3: rsum = prod + rnd          (round)
//   stage 4: q    = rsum >>> shift
//   stage 5: qs   = clamp(q, -256, 255)  (exact: with zp in [-128,127] anything outside this range
//                                         clamps to the int8 limits anyway)
//   stage 6: y    = clamp(qs + zp, lo, 127)
//   (stages 3-6 used to be ONE combinational stage: 25 logic levels, failed 100 MHz on the 7z020.
//    The postproc unit only needs out_valid, its credit counter makes it latency-tolerant.)
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

    reg                     v1, v2, v3, v4, v5;
    reg signed [ACC_W:0]    xb;
    reg        [MUL_W-1:0]  m1;
    reg        [SH_W-1:0]   s1, s2, s3;
    reg signed [PW-1:0]     prod, rsum, q;
    reg signed [8:0]        qs;

    wire signed [ACC_W:0]   sum = x + bias;

    // stage 3: round
    wire signed [PW-1:0] rnd = (s2 == 0) ? {PW{1'b0}} : ({{(PW-1){1'b0}}, 1'b1} << (s2 - 1));

    // stage 5: saturate to 9 bits (exact, see header)
    wire signed [8:0] qsat = (q >  255) ? 9'sd255 :
                             (q < -256) ? -9'sd256 : q[8:0];

    // stage 6: add zero point, ReLU lower bound, int8 upper bound
    wire signed [9:0] z  = qs + zp;
    wire signed [9:0] lo = relu_en ? zp : -10'sd128;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; v4 <= 1'b0; v5 <= 1'b0; out_valid <= 1'b0;
        end else begin
            v1        <= in_valid;
            v2        <= v1;
            v3        <= v2;
            v4        <= v3;
            v5        <= v4;
            out_valid <= v5;
        end
    end

    always @(posedge clk) begin
        xb   <= sum;                         m1 <= mult;  s1 <= shift;      // stage 1
        prod <= xb * $signed({1'b0, m1});    s2 <= s1;                      // stage 2
        rsum <= prod + rnd;                  s3 <= s2;                      // stage 3
        q    <= rsum >>> s3;                                                // stage 4
        qs   <= qsat;                                                       // stage 5
        y    <= (z < lo) ? lo[7:0] : (z > 127) ? 8'sd127 : z[7:0];          // stage 6
    end
endmodule