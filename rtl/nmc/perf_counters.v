// ============================================================================
// Performance counters for one NMC unit
//
// Measures how well the 16 PEs are kept busy
//
// Window: counting starts the cycle after `start` is accepted and stops after
// the cycle in which `done` is high, so   cyc_total == (cycle of done) - (cycle
// of start)  as measured by a testbench. Counters are cleared by the next
// `start`, and hold their values afterwards (read them when perf_valid = 1).
//
// Counters (all CNT_W bits, no saturation):
//   cyc_total        layer busy cycles
//   cyc_load         cycles in the load phases    (IFMap patch + weights)
//   cyc_compute      cycles in the compute phase  (compute_start .. compute_done)
//   cyc_store        cycles in the store phase
//                    (remaining cycles = FSM overhead: flushes, next-oc, ...)
//   cyc_mac          cycles with mac_en = 1  (one kernel element per cycle)
//   cyc_stall        cycles a MAC op was pending but not issued (operand not valid)
//   cyc_mem_blocked  cycles with >= 1 NMC memory request lost to the core
//   mac_ops          sum over cycles of (#active PEs doing a MAC)
//                    = popcount(pe_mask) on every mac_en cycle
//
// Derived (compute outside the RTL, no divider needed in hardware):
//   PE efficiency (layer)   = mac_ops / (NUM_PE * cyc_total)
//   PE efficiency (compute) = mac_ops / (NUM_PE * cyc_compute)
//   PE activity (paper)     = mac_ops / (NUM_PE * cyc_mac)
//                             -> 100 % for full tiles, < 100 % only from edge-tile masking
//   GOPS                    = 2 * mac_ops / (cyc_total * T_clk)
// ============================================================================
module perf_counters #(
    parameter NUM_PE = 16,
    parameter CNT_W  = 48
)(
    input  wire              clk,
    input  wire              rst,

    input  wire              start,        // layer start (same signal the address generator sees)
    input  wire              done,         // layer done pulse

    input  wire              mac_en,       // PE array accumulates this cycle
    input  wire [NUM_PE-1:0] pe_mask,      // active PEs of the current tile
    input  wire              stall,        // op pending but not issued
    input  wire              mem_blocked,  // >= 1 NMC memory request lost to the core
    input  wire              ph_load,      // address generator is in a load phase
    input  wire              ph_compute,   // ... compute phase
    input  wire              ph_store,     // ... store phase

    output reg               running,
    output reg               perf_valid,   // counters hold a complete layer

    output reg [CNT_W-1:0]   cyc_total,
    output reg [CNT_W-1:0]   cyc_load,
    output reg [CNT_W-1:0]   cyc_compute,
    output reg [CNT_W-1:0]   cyc_store,
    output reg [CNT_W-1:0]   cyc_mac,
    output reg [CNT_W-1:0]   cyc_stall,
    output reg [CNT_W-1:0]   cyc_mem_blocked,
    output reg [CNT_W-1:0]   mac_ops
);

    // number of set bits in pe_mask
    function [$clog2(NUM_PE+1)-1:0] popcount(input [NUM_PE-1:0] v);
        integer i;
        begin
            popcount = 0;
            for (i = 0; i < NUM_PE; i = i + 1)
                popcount = popcount + v[i];
        end
    endfunction

    wire [$clog2(NUM_PE+1)-1:0] active_pes = popcount(pe_mask);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            running <= 1'b0;  perf_valid <= 1'b0;
            cyc_total <= 0;   cyc_load <= 0;  cyc_compute <= 0;  cyc_store <= 0;
            cyc_mac <= 0;     cyc_stall <= 0; cyc_mem_blocked <= 0;  mac_ops <= 0;
        end else if (start && !running) begin
            running <= 1'b1;  perf_valid <= 1'b0;
            cyc_total <= 0;   cyc_load <= 0;  cyc_compute <= 0;  cyc_store <= 0;
            cyc_mac <= 0;     cyc_stall <= 0; cyc_mem_blocked <= 0;  mac_ops <= 0;
        end else if (running) begin
            cyc_total <= cyc_total + 1'b1;
            if (ph_load)     cyc_load        <= cyc_load        + 1'b1;
            if (ph_compute)  cyc_compute     <= cyc_compute     + 1'b1;
            if (ph_store)    cyc_store       <= cyc_store       + 1'b1;
            if (stall)       cyc_stall       <= cyc_stall       + 1'b1;
            if (mem_blocked) cyc_mem_blocked <= cyc_mem_blocked + 1'b1;
            if (mac_en) begin
                cyc_mac <= cyc_mac + 1'b1;
                mac_ops <= mac_ops + active_pes;
            end
            if (done) begin
                running    <= 1'b0;
                perf_valid <= 1'b1;
            end
        end
    end

endmodule