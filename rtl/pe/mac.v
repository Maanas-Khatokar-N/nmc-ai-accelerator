module mac #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32
)(
    input clk, rst, clear, en,
    input signed [DATA_WIDTH-1:0] a, b,
    output signed [ACC_WIDTH-1:0] acc_out
);

    
    always @(posedge clk or posedge rst) begin
        
        //Asynchronous Reset
        if (rst) acc_out <= 0;

        else if (clear) acc_out <= 0;

        else begin
            if (en) acc_out <= acc_out + a*b;       //Automatically Sign extends
        end

    end

endmodule