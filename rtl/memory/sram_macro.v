// ============================================================================
// SRAM macro
//
//   - one port, shared for read and write, one access per cycle
//   - en = 0         : macro idle, rdata HOLDS its last value
//   - en = 1, we = 0 : read.  rdata is valid ONE cycle later (registered)
//   - en = 1, we = 1 : write. mem[addr] <= wdata at the clock edge
//   - read-before-write: on a write cycle rdata gets the OLD word. Consumers
//     must ignore rdata after a write (the bank wrapper does: rvalid = 0)
//   - no reset and no initial values: the array and rdata are plain
//     clocked storage, so synthesis can map this to a foundry SRAM macro,
//     block RAM, or (for small depths) registers. Contents are undefined
//     until written, like real silicon.
//
// DEPTH is in words. Default 4096 x 32 bit = 16 KB, so 8 macros = one 128 KB
// bank of the paper.
//
// ============================================================================
module sram_macro #(
    parameter DATA_WIDTH = 32,
    parameter DEPTH      = 4096,
    parameter ADDR_WIDTH = $clog2(DEPTH)
) (
    input  wire                  clk,
    input  wire                  en,
    input  wire                  we,
    input  wire [ADDR_WIDTH-1:0] addr,
    input  wire [DATA_WIDTH-1:0] wdata,
    output reg  [DATA_WIDTH-1:0] rdata
);

    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (en) begin
            rdata <= mem[addr];             // old data (read-before-write)
            if (we)
                mem[addr] <= wdata;
        end
    end

endmodule