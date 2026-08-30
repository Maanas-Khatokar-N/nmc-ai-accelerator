module pe_array_tb #(
    parameter NUM_PE = 16,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32,
    parameter TOTAL_DATA_WIDTH = NUM_PE*DATA_WIDTH,
    parameter TOTAL_ACC_WIDTH = NUM_PE*ACC_WIDTH
);

    reg clk, rst, clear, en;
    reg signed [TOTAL_DATA_WIDTH-1:0] a, b;
    wire signed [TOTAL_ACC_WIDTH-1:0] acc_out;

    pe_array #(
        .NUM_PE(NUM_PE),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .TOTAL_DATA_WIDTH(TOTAL_DATA_WIDTH),
        .TOTAL_ACC_WIDTH(TOTAL_ACC_WIDTH)
    ) dut(
        .clk(clk), 
        .rst(rst), 
        .clear(clear), 
        .en(en), 
        .a(a), 
        .b(b), 
        .acc_out(acc_out)
    );


    initial begin
        clk <= 1'b0;
        rst <= 1'b1;
        clear = 0;
        en = 0;
        a = 0; b = 0;

        #3 rst <= 1'b0;
    end

    always #5 clk <= ~clk;

    initial begin
        #7 en <= 1'b1;
        a <= 128'd2;
        b <= 128'd3;

        #10
        a <= 4;
        b <= 5;

        #50 $finish;
    end

    initial begin
        $dumpfile("sim/waveform/pe_array_wave.vcd");
        $dumpvars(0, pe_array_tb);
    end
    
endmodule