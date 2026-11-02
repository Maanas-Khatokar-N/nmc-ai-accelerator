// Reference model: the ORIGINAL register_file (flat array, all-port tracking).
// Used only by tb/unit/register_file_equiv_tb.v. Not synthesised.
module register_file_ref #(
    parameter DATA_WIDTH = 32, DEPTH = 256, NUM_READ_PORTS = 16, NUM_WRITE_PORTS = 16,
    parameter ENFORCE_REPLACE = 1, parameter ADDR_WIDTH = $clog2(DEPTH)
) (
    input clk, rst, input flush,
    input  [NUM_WRITE_PORTS-1:0]            write_en,
    input  [ADDR_WIDTH*NUM_WRITE_PORTS-1:0] write_addr,
    input  [DATA_WIDTH*NUM_WRITE_PORTS-1:0] write_data,
    output reg [NUM_WRITE_PORTS-1:0]        write_blocked,
    input  [ADDR_WIDTH*NUM_WRITE_PORTS-1:0] chk_addr,
    output reg [NUM_WRITE_PORTS-1:0]        chk_free,
    input  [NUM_READ_PORTS-1:0]             read_en, read_last,
    input  [ADDR_WIDTH*NUM_READ_PORTS-1:0]  read_addr,
    output reg [DATA_WIDTH*NUM_READ_PORTS-1:0] read_data,
    output reg [NUM_READ_PORTS-1:0]         read_valid
);
    reg [DATA_WIDTH-1:0] register [0:DEPTH-1];
    reg [DEPTH-1:0] d_bit, v_bit;
    integer r, w;
    reg [ADDR_WIDTH-1:0] ra, wa, ca;
    always @(*) begin
        for (r = 0; r < NUM_READ_PORTS; r = r + 1) begin
            ra = read_addr[r*ADDR_WIDTH +: ADDR_WIDTH];
            read_valid[r] = v_bit[ra];
            read_data[r*DATA_WIDTH +: DATA_WIDTH] = v_bit[ra] ? register[ra] : {DATA_WIDTH{1'b0}};
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
    integer k;
    reg [ADDR_WIDTH-1:0] wa_s;
    always @(posedge clk) if (!flush)
        for (k = 0; k < NUM_WRITE_PORTS; k = k + 1) begin
            wa_s = write_addr[k*ADDR_WIDTH +: ADDR_WIDTH];
            if (write_en[k] && !write_blocked[k]) register[wa_s] <= write_data[k*DATA_WIDTH +: DATA_WIDTH];
        end
    integer i, j;
    reg [ADDR_WIDTH-1:0] ra_s, wa_t;
    reg [DEPTH-1:0] nd, nv;
    always @(posedge clk or posedge rst) begin
        if (rst) begin d_bit <= 0; v_bit <= 0; end
        else if (flush) begin d_bit <= 0; v_bit <= 0; end
        else begin
            nd = d_bit; nv = v_bit;
            for (i = 0; i < NUM_READ_PORTS; i = i + 1) begin
                ra_s = read_addr[i*ADDR_WIDTH +: ADDR_WIDTH];
                if (read_en[i] && v_bit[ra_s]) begin nd[ra_s] = 1'b1; if (read_last[i]) nv[ra_s] = 1'b0; end
            end
            for (j = 0; j < NUM_WRITE_PORTS; j = j + 1) begin
                wa_t = write_addr[j*ADDR_WIDTH +: ADDR_WIDTH];
                if (write_en[j] && !write_blocked[j]) begin nd[wa_t] = 1'b0; nv[wa_t] = 1'b1; end
            end
            d_bit <= nd; v_bit <= nv;
        end
    end
endmodule