// ============================================================================
// NMC-controlled register file with Dirty/Valid (DV) lifecycle tracking
//
// Paper (Sec. III-E.2): every data word carries two control bits, D and V,
// that track the lifecycle of the element (like a hardware-controlled cache):
//
//   DV  meaning
//   00  empty      - nothing loaded
//   01  loaded     - valid data, not used yet
//   11  in use     - data was used, but will be used again
//   10  consumed   - data was used, lifecycle over, may be replaced
//
// Transitions (in occurrence order):  00 -> 01 -> 11 (..repeat..) -> 10 -> 01
//
//   write  (accepted)            : any replaceable word (00/10) -> 01
//   read   (read_en)             : 01/11 -> 11
//   read + read_last             : 01/11 -> 10
//   flush                        : all words -> 00   (nmc_clear / new layer)
//
// So V = "data is live and readable", D = "data has been used at least once".
//
// Design choices:
//  - Read is asynchronous (combinational) so PEs get operands in the same
//    cycle; only the DV state updates on the clock edge.
//  - read_data is forced to 0 and read_valid to 0 when V=0, so the access
//    controller can use read_valid as the "operand available" handshake.
//  - A write to a live word (V=1) is rejected and flagged on write_blocked
//    (ENFORCE_REPLACE=1). This is what protects un-consumed IFMap data from
//    being overwritten. Set ENFORCE_REPLACE=0 to allow unconditional writes.
//  - Several writes to the same address in one cycle: highest port wins.
//  - Several reads of the same address in one cycle: if ANY of them asserts
//    read_last the word becomes consumed (10).
//  - The data array has no reset (cheaper in silicon); DV bits are reset, and
//    V=0 masks the output, so stale data is never visible.
//
// ============================================================================
module register_file #(
    parameter DATA_WIDTH      = 32,
    parameter DEPTH           = 256,
    parameter NUM_READ_PORTS  = 16,
    parameter NUM_WRITE_PORTS = 16,
    parameter ENFORCE_REPLACE = 1,
    parameter ADDR_WIDTH      = $clog2(DEPTH)
) (
    input clk, rst,
    input flush,                                   // clear all DV bits -> 00

    // Write ports
    input  [NUM_WRITE_PORTS-1:0]            write_en,
    input  [ADDR_WIDTH*NUM_WRITE_PORTS-1:0] write_addr,
    input  [DATA_WIDTH*NUM_WRITE_PORTS-1:0] write_data,
    output reg [NUM_WRITE_PORTS-1:0]        write_blocked, // write rejected (word still live)

    // Read ports
    input  [NUM_READ_PORTS-1:0]             read_en,   // this read consumes a use
    input  [NUM_READ_PORTS-1:0]             read_last, // this read is the final use
    input  [ADDR_WIDTH*NUM_READ_PORTS-1:0]  read_addr,
    output reg [DATA_WIDTH*NUM_READ_PORTS-1:0] read_data,
    output reg [NUM_READ_PORTS-1:0]         read_valid // V bit of the addressed word
);

    // ------------------------------------------------------------------
    // Storage: data + one D bit and one V bit per word
    // ------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] register [0:DEPTH-1];
    reg [DEPTH-1:0]      d_bit;
    reg [DEPTH-1:0]      v_bit;

    // ------------------------------------------------------------------
    // Combinational read (masked by V) and write-block detection
    // ------------------------------------------------------------------
    integer r, w;
    reg [ADDR_WIDTH-1:0] ra, wa;

    always @(*) begin
        for (r = 0; r < NUM_READ_PORTS; r = r + 1) begin
            ra = read_addr[r*ADDR_WIDTH +: ADDR_WIDTH];
            read_valid[r] = v_bit[ra];
            read_data[r*DATA_WIDTH +: DATA_WIDTH] =
                v_bit[ra] ? register[ra] : {DATA_WIDTH{1'b0}};
        end

        for (w = 0; w < NUM_WRITE_PORTS; w = w + 1) begin
            wa = write_addr[w*ADDR_WIDTH +: ADDR_WIDTH];
            write_blocked[w] = write_en[w] && (ENFORCE_REPLACE != 0) && v_bit[wa];
        end
    end

    // ------------------------------------------------------------------
    // Data array write (no reset)
    // ------------------------------------------------------------------
    integer k;
    reg [ADDR_WIDTH-1:0] wa_s;

    always @(posedge clk) begin
        if (!flush) begin
            for (k = 0; k < NUM_WRITE_PORTS; k = k + 1) begin
                wa_s = write_addr[k*ADDR_WIDTH +: ADDR_WIDTH];
                if (write_en[k] && !write_blocked[k])
                    register[wa_s] <= write_data[k*DATA_WIDTH +: DATA_WIDTH];
            end
        end
    end

    // ------------------------------------------------------------------
    // DV state machine. Reads are applied first, then accepted writes, so a
    // write (-> 01) always wins over a read (-> 11/10) of the same word.
    // ------------------------------------------------------------------
    integer i, j;
    reg [ADDR_WIDTH-1:0] ra_s, wa_t;
    reg [DEPTH-1:0] nd, nv;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            d_bit <= {DEPTH{1'b0}};
            v_bit <= {DEPTH{1'b0}};
        end
        else if (flush) begin
            d_bit <= {DEPTH{1'b0}};
            v_bit <= {DEPTH{1'b0}};
        end
        else begin
            nd = d_bit;
            nv = v_bit;

            // Reads: 01/11 -> 11, or -> 10 on last use. Reads of V=0 words ignored.
            for (i = 0; i < NUM_READ_PORTS; i = i + 1) begin
                ra_s = read_addr[i*ADDR_WIDTH +: ADDR_WIDTH];
                if (read_en[i] && v_bit[ra_s]) begin
                    nd[ra_s] = 1'b1;
                    if (read_last[i]) nv[ra_s] = 1'b0;
                end
            end

            // Accepted writes: -> 01
            for (j = 0; j < NUM_WRITE_PORTS; j = j + 1) begin
                wa_t = write_addr[j*ADDR_WIDTH +: ADDR_WIDTH];
                if (write_en[j] && !write_blocked[j]) begin
                    nd[wa_t] = 1'b0;
                    nv[wa_t] = 1'b1;
                end
            end

            d_bit <= nd;
            v_bit <= nv;
        end
    end

endmodule