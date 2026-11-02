// ============================================================================
// NMC-controlled register file with Dirty/Valid (DV) lifecycle tracking
//
// Paper (Sec. III-E.2): every data word carries two control bits, D and V:
//
//   DV  meaning
//   00  empty      - nothing loaded
//   01  loaded     - valid data, not used yet
//   11  in use     - data was used, but will be used again
//   10  consumed   - data was used, lifecycle over, may be replaced
//
//   write  (accepted)            : any replaceable word (00/10) -> 01
//   read   (read_en)             : 01/11 -> 11
//   read + read_last             : 01/11 -> 10
//   flush                        : all words -> 00
//
// Behaviour (unchanged from the first version, cycle-exact):
//  - Read is asynchronous, masked by V (read_data = 0, read_valid = 0 when V = 0).
//  - A write to a live word (V=1) is rejected and flagged on write_blocked
//    (ENFORCE_REPLACE=1).
//  - Several writes to the same address in one cycle: highest port wins.
//  - Several reads of the same address in one cycle: if ANY asserts read_last the
//    word becomes consumed.
//
// FPGA mapping (new)
//  BANKED_STORAGE = 1 (default): the data array is split into NUM_WRITE_PORTS banks,
//    word address a lives in bank a[log2(NWP)-1:0], bank-local address a[..:log2(NWP)].
//    Each bank has ONE write port, so it maps to distributed RAM (LUTRAM) with
//    asynchronous read, and every read port gets its own copy of the banks
//    (NUM_READ_PORTS replicas). CONTRACT: the words written in one cycle must fall in
//    different banks, i.e. they must lie in a window of NUM_WRITE_PORTS consecutive
//    addresses (true for the load generator: lane m writes dst0+m, and for the psum reload).
//    If two accepted writes of one cycle hit the same bank with different addresses,
//    the higher port wins and the other is lost.
//    DEPTH must be a multiple of NUM_WRITE_PORTS (power of two), and DEPTH > NUM_WRITE_PORTS.
//    With a single write port there is nothing to bank: the flat array is used (it is already LUTRAM-friendly).
//  BANKED_STORAGE = 0: one flat array (original structure), no restriction on write
//    addresses. Use it for tiny files (psum RF, 16 words) which synthesise to flip-flops.
//  TRACK_READS = 1 (default): read_en / read_last update the D bit and consume words.
//  TRACK_READS = 0: read_en / read_last are IGNORED. V only changes by write and flush,
//    D is never set. Removes read_ports x DEPTH update logic. Valid whenever the user never
//    consumes words with read_last (IFMap RF and weight RF in nmc_unit tie it to 0).
// ============================================================================
module register_file #(
    parameter DATA_WIDTH      = 32,
    parameter DEPTH           = 256,
    parameter NUM_READ_PORTS  = 16,
    parameter NUM_WRITE_PORTS = 16,
    parameter ENFORCE_REPLACE = 1,
    parameter BANKED_STORAGE  = 1,
    parameter TRACK_READS     = 1,
    parameter ADDR_WIDTH      = $clog2(DEPTH)
) (
    input clk, rst,
    input flush,                                   // clear all DV bits -> 00

    // Write ports
    input  [NUM_WRITE_PORTS-1:0]            write_en,
    input  [ADDR_WIDTH*NUM_WRITE_PORTS-1:0] write_addr,
    input  [DATA_WIDTH*NUM_WRITE_PORTS-1:0] write_data,
    output reg [NUM_WRITE_PORTS-1:0]        write_blocked, // write rejected (word still live)

    // Pre-check: "would a write to this address be accepted?"  (V = 0)
    input  [ADDR_WIDTH*NUM_WRITE_PORTS-1:0] chk_addr,
    output reg [NUM_WRITE_PORTS-1:0]        chk_free,

    // Read ports
    input  [NUM_READ_PORTS-1:0]             read_en,   // this read consumes a use
    input  [NUM_READ_PORTS-1:0]             read_last, // this read is the final use
    input  [ADDR_WIDTH*NUM_READ_PORTS-1:0]  read_addr,
    output reg [DATA_WIDTH*NUM_READ_PORTS-1:0] read_data,
    output reg [NUM_READ_PORTS-1:0]         read_valid // V bit of the addressed word
);

    reg [DEPTH-1:0] d_bit;
    reg [DEPTH-1:0] v_bit;

    // accepted writes (data side)
    wire [NUM_WRITE_PORTS-1:0] w_ok = write_en & ~write_blocked & {NUM_WRITE_PORTS{~flush}};

    // unmasked read data of every port
    wire [DATA_WIDTH*NUM_READ_PORTS-1:0] raw_rd;

    // ------------------------------------------------------------------
    // Data storage
    // ------------------------------------------------------------------
    genvar r, bk;
    generate
    if (BANKED_STORAGE != 0 && NUM_WRITE_PORTS > 1) begin : g_banked
        localparam NB  = NUM_WRITE_PORTS;
        localparam NBW = $clog2(NB);
        localparam BAW = ADDR_WIDTH - NBW;      // bank-local address bits
        localparam BD  = DEPTH / NB;            // words per bank

        // one write per bank per cycle; highest accepted port that targets the bank wins
        reg [NB-1:0]            bw_en;
        reg [NB*BAW-1:0]        bw_addr;
        reg [NB*DATA_WIDTH-1:0] bw_data;
        integer b, p;
        always @(*) begin
            bw_en = {NB{1'b0}}; bw_addr = {(NB*BAW){1'b0}}; bw_data = {(NB*DATA_WIDTH){1'b0}};
            for (b = 0; b < NB; b = b + 1)
                for (p = 0; p < NUM_WRITE_PORTS; p = p + 1)
                    if (w_ok[p] && (write_addr[p*ADDR_WIDTH +: NBW] == b)) begin
                        bw_en[b] = 1'b1;
                        bw_addr[b*BAW +: BAW]             = write_addr[p*ADDR_WIDTH + NBW +: BAW];
                        bw_data[b*DATA_WIDTH +: DATA_WIDTH] = write_data[p*DATA_WIDTH +: DATA_WIDTH];
                    end
        end

        // every read port owns a full copy of the banks (1 write port, async read -> LUTRAM)
        for (r = 0; r < NUM_READ_PORTS; r = r + 1) begin : g_rd
            wire [NB*DATA_WIDTH-1:0] bank_q;
            for (bk = 0; bk < NB; bk = bk + 1) begin : g_bank
                (* ram_style = "distributed" *) reg [DATA_WIDTH-1:0] mem [0:BD-1];
                always @(posedge clk)
                    if (bw_en[bk]) mem[bw_addr[bk*BAW +: BAW]] <= bw_data[bk*DATA_WIDTH +: DATA_WIDTH];
                assign bank_q[bk*DATA_WIDTH +: DATA_WIDTH] = mem[read_addr[r*ADDR_WIDTH + NBW +: BAW]];
            end
            assign raw_rd[r*DATA_WIDTH +: DATA_WIDTH] =
                bank_q[read_addr[r*ADDR_WIDTH +: NBW]*DATA_WIDTH +: DATA_WIDTH];
        end
    end else begin : g_flat
        reg [DATA_WIDTH-1:0] register [0:DEPTH-1];
        integer k;
        always @(posedge clk)
            for (k = 0; k < NUM_WRITE_PORTS; k = k + 1)
                if (w_ok[k])
                    register[write_addr[k*ADDR_WIDTH +: ADDR_WIDTH]] <= write_data[k*DATA_WIDTH +: DATA_WIDTH];
        for (r = 0; r < NUM_READ_PORTS; r = r + 1) begin : g_rd
            assign raw_rd[r*DATA_WIDTH +: DATA_WIDTH] = register[read_addr[r*ADDR_WIDTH +: ADDR_WIDTH]];
        end
    end
    endgenerate

    // ------------------------------------------------------------------
    // Combinational read (masked by V), write-block detection, free check
    // ------------------------------------------------------------------
    integer ri, w;
    reg [ADDR_WIDTH-1:0] ra, wa, ca;

    always @(*) begin
        for (ri = 0; ri < NUM_READ_PORTS; ri = ri + 1) begin
            ra = read_addr[ri*ADDR_WIDTH +: ADDR_WIDTH];
            read_valid[ri] = v_bit[ra];
            read_data[ri*DATA_WIDTH +: DATA_WIDTH] =
                v_bit[ra] ? raw_rd[ri*DATA_WIDTH +: DATA_WIDTH] : {DATA_WIDTH{1'b0}};
        end

        for (w = 0; w < NUM_WRITE_PORTS; w = w + 1) begin
            wa = write_addr[w*ADDR_WIDTH +: ADDR_WIDTH];
            write_blocked[w] = write_en[w] && (ENFORCE_REPLACE != 0) && v_bit[wa];
        end

        for (w = 0; w < NUM_WRITE_PORTS; w = w + 1) begin
            ca = chk_addr[w*ADDR_WIDTH +: ADDR_WIDTH];
            chk_free[w] = ~v_bit[ca];
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

            if (TRACK_READS != 0)
                for (i = 0; i < NUM_READ_PORTS; i = i + 1) begin
                    ra_s = read_addr[i*ADDR_WIDTH +: ADDR_WIDTH];
                    if (read_en[i] && v_bit[ra_s]) begin
                        nd[ra_s] = 1'b1;
                        if (read_last[i]) nv[ra_s] = 1'b0;
                    end
                end

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