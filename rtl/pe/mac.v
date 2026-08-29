module mac (
    input clk, rst, clear, en,
    input signed [7:0] a, b,
    output reg signed [31:0] acc_out
);
    
    always @(posedge clk or posedge rst) begin
        
        //Asynchronous Reset
        if (rst) acc_out <= 32'b0;

        else if (clear) acc_out <= 32'b0;

        else begin
            if (en) acc_out <= acc_out + a*b;       //Automatically Sign extends
        end

    end

endmodule