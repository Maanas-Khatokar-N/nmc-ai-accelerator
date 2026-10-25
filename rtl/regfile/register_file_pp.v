// Ping-pong register file: two register_file halves.
//   wr_sel / flush_sel pick the half that is written / flushed (loader side)
//   rd_sel picks the half that is read and whose D/V bits are updated (compute side)
//   chk_addr / chk_free: per-word "is the destination free (V=0)?" query on the WRITE half
module register_file_pp #(
    parameter DATA_WIDTH      = 32,
    parameter DEPTH           = 256,
    parameter NUM_READ_PORTS  = 16,
    parameter NUM_WRITE_PORTS = 16,
    parameter ENFORCE_REPLACE = 1,
    parameter ADDR_WIDTH      = $clog2(DEPTH)
) (
    input clk, rst,
    input flush, flush_sel, wr_sel, rd_sel,

    input  [NUM_WRITE_PORTS-1:0]            write_en,
    input  [ADDR_WIDTH*NUM_WRITE_PORTS-1:0] write_addr,
    input  [DATA_WIDTH*NUM_WRITE_PORTS-1:0] write_data,
    output [NUM_WRITE_PORTS-1:0]            write_blocked,

    input  [ADDR_WIDTH*NUM_WRITE_PORTS-1:0] chk_addr,
    output [NUM_WRITE_PORTS-1:0]            chk_free,

    input  [NUM_READ_PORTS-1:0]             read_en,
    input  [NUM_READ_PORTS-1:0]             read_last,
    input  [ADDR_WIDTH*NUM_READ_PORTS-1:0]  read_addr,
    output [DATA_WIDTH*NUM_READ_PORTS-1:0]  read_data,
    output [NUM_READ_PORTS-1:0]             read_valid
);
    wire [NUM_WRITE_PORTS-1:0] blk_a, blk_b, cf_a, cf_b;
    wire [DATA_WIDTH*NUM_READ_PORTS-1:0] rd_a, rd_b;
    wire [NUM_READ_PORTS-1:0] rv_a, rv_b;

    register_file #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(DEPTH), .NUM_READ_PORTS(NUM_READ_PORTS),
                    .NUM_WRITE_PORTS(NUM_WRITE_PORTS), .ENFORCE_REPLACE(ENFORCE_REPLACE),
                    .ADDR_WIDTH(ADDR_WIDTH)) u_a (
        .clk(clk), .rst(rst), .flush(flush & ~flush_sel),
        .write_en(write_en & {NUM_WRITE_PORTS{~wr_sel}}), .write_addr(write_addr),
        .write_data(write_data), .write_blocked(blk_a),
        .chk_addr(chk_addr), .chk_free(cf_a),
        .read_en(read_en & {NUM_READ_PORTS{~rd_sel}}), .read_last(read_last),
        .read_addr(read_addr), .read_data(rd_a), .read_valid(rv_a));

    register_file #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(DEPTH), .NUM_READ_PORTS(NUM_READ_PORTS),
                    .NUM_WRITE_PORTS(NUM_WRITE_PORTS), .ENFORCE_REPLACE(ENFORCE_REPLACE),
                    .ADDR_WIDTH(ADDR_WIDTH)) u_b (
        .clk(clk), .rst(rst), .flush(flush & flush_sel),
        .write_en(write_en & {NUM_WRITE_PORTS{wr_sel}}), .write_addr(write_addr),
        .write_data(write_data), .write_blocked(blk_b),
        .chk_addr(chk_addr), .chk_free(cf_b),
        .read_en(read_en & {NUM_READ_PORTS{rd_sel}}), .read_last(read_last),
        .read_addr(read_addr), .read_data(rd_b), .read_valid(rv_b));

    assign write_blocked = wr_sel ? blk_b : blk_a;
    assign chk_free      = wr_sel ? cf_b  : cf_a;
    assign read_data     = rd_sel ? rd_b  : rd_a;
    assign read_valid    = rd_sel ? rv_b  : rv_a;
endmodule