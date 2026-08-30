module pe #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32
)(
    input clk, rst, clear, en,
    input signed [DATA_WIDTH-1:0] a, b,
    output reg signed [ACC_WIDTH-1:0] acc_out
);

    mac #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) m0(
        .clk(clk), 
        .rst(rst), 
        .clear(clear), 
        .en(en), 
        .a(a), 
        .b(b), 
        .acc_out(acc_out)
    );
    
endmodule