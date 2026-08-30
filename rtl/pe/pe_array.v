module pe_array #(
    parameter NUM_PE = 16,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32
    parameter TOTAL_DATA_WIDTH = NUM_PE*DATA_WIDTH,
    parameter TOTAL_ACC_WIDTH = NUM_PE*ACC_WIDTH
) (
    input clk, rst, clear, en,
    input signed [TOTAL_DATA_WIDTH-1:0] a, b,
    output signed [TOTAL_ACC_WIDTH-1:0] acc_out
);

    genvar i;
    
    generate
        for (i = 1; i <= NUM_PE; i=i+1) begin : gen_pe
            pe #(
                .DATA_WIDTH(DATA_WIDTH),
                .ACC_WIDTH(ACC_WIDTH)
            ) pe_inst (
                .clk(clk), 
                .rst(rst), 
                .clear(clear), 
                .en(en), 
                .a(a[i*DATA_WIDTH - 1 -: DATA_WIDTH]), 
                .b(b[i*DATA_WIDTH - 1 -: DATA_WIDTH]), 
                .acc_out(acc_out[i*ACC_WIDTH - 1 -: ACC_WIDTH])
            );
        end
    endgenerate

endmodule