// ============================================================================
// Access controller
//
// Sits between the address generator (compute sequencer) and the datapath
// (IFMap RF, weight RF, psum RF, PE array). It is purely a handshake block:
//
//   MAC fire  = seq_valid & all ACTIVE IFMap lanes valid & weight valid & !psum_req
//
// When MAC fires, in that same cycle it
//   - asserts pe_mac_en              (PE accumulates ifmap*weight at the edge)
//   - asserts seq_enable             (sequencer advances to the next op)
//   - asserts ifrf_read_en[lane]     (active lanes only: D bit 01->11)
//   - asserts wrf_read_en
// When it does not fire (operand not yet valid) nothing is consumed and the
// sequencer holds: a stall, never a wrong result.
//
// There is NO pipeline stage between RF and PE (RF reads are combinational), so
// a stall-free kernel runs one MAC per cycle: 3x3 kernel = 9 compute cycles.
//
// Psum reload (OFMap -> psum RF -> PE): psum_req from the store/reload side.
//   psum_load fires when all active psum RF lanes are valid. The psum words are
//   consumed (read_last = read_en -> DV 10). MAC is blocked while psum_req is
//   high, so load and MAC never collide.
//
// Timing:
//   pe_clear     = compute_start delayed 1 cycle (first op appears >= 2 cycles
//                  after compute_start, so the clear still lands first)
//   compute_done = (last MAC fired) delayed 1 cycle = the accumulator register.
//                  In the cycle compute_done is high, acc_out is final.
// ============================================================================
module access_controller #(
    parameter NUM_PE = 16
)(
    input  wire              clk,
    input  wire              rst,

    // ---- address generator side ----
    input  wire              compute_start,
    input  wire              seq_valid,
    input  wire              seq_last,
    input  wire [NUM_PE-1:0] pe_mask,
    output wire              seq_enable,
    output reg               compute_done,

    // ---- IFMap / weight RF ----
    input  wire [NUM_PE-1:0] ifrf_read_valid,
    input  wire              wrf_read_valid,
    output wire [NUM_PE-1:0] ifrf_read_en,
    output wire              wrf_read_en,

    // ---- psum reload ----
    input  wire              psum_req,
    input  wire [NUM_PE-1:0] psrf_read_valid,
    output wire [NUM_PE-1:0] psrf_read_en,
    output wire [NUM_PE-1:0] psrf_read_last,
    output wire              psum_ack,

    // ---- PE array ----
    output reg               pe_clear,
    output wire              pe_mac_en,
    output wire              pe_psum_load,

    output wire              stall          // op pending but not issued
);

    wire if_ok   = &(ifrf_read_valid  | ~pe_mask);   // masked lanes ignored
    wire ps_ok   = &(psrf_read_valid  | ~pe_mask);
    wire w_ok    = wrf_read_valid;

    wire mac_fire  = seq_valid & if_ok & w_ok & ~psum_req;
    wire psum_fire = psum_req & ps_ok;

    assign pe_mac_en      = mac_fire;
    assign seq_enable     = mac_fire;
    assign ifrf_read_en   = pe_mask & {NUM_PE{mac_fire}};
    assign wrf_read_en    = mac_fire;

    assign pe_psum_load   = psum_fire;
    assign psum_ack       = psum_fire;
    assign psrf_read_en   = pe_mask & {NUM_PE{psum_fire}};
    assign psrf_read_last = psrf_read_en;

    assign stall          = seq_valid & ~mac_fire;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            compute_done <= 1'b0;
            pe_clear     <= 1'b0;
        end else begin
            compute_done <= mac_fire & seq_last;
            pe_clear     <= compute_start;
        end
    end

endmodule