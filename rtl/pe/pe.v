module pe (
    input clk, rst, clear, en,
    input signed [7:0] a, b,
    output signed [31:0] acc_out
);

    mac m0(
        .clk(clk), 
        .rst(rst), 
        .clear(clear), 
        .en(en), 
        .a(a), 
        .b(b), 
        .acc_out(acc_out)
    );
    
endmodule