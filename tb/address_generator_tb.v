`timescale 1ns/1ps

module address_generator_tb;

    parameter ADDR_WIDTH = 16;
    parameter DIM_WIDTH  = 16;
    parameter NUM_PE     = 16;

    reg clk;
    reg rst;
    reg start;
    reg enable;

    reg [DIM_WIDTH-1:0] ifmap_rows;
    reg [DIM_WIDTH-1:0] ifmap_cols;
    reg [DIM_WIDTH-1:0] kernel_rows;
    reg [DIM_WIDTH-1:0] kernel_cols;
    reg [DIM_WIDTH-1:0] input_channels;
    reg [DIM_WIDTH-1:0] output_channels;
    reg [DIM_WIDTH-1:0] stride;

    reg [ADDR_WIDTH-1:0] ifmap_base;
    reg [ADDR_WIDTH-1:0] weight_base;
    reg [ADDR_WIDTH-1:0] ofmap_base;

    wire [NUM_PE*ADDR_WIDTH-1:0] ifmap_addr;
    wire [ADDR_WIDTH-1:0] weight_addr;
    wire [NUM_PE*ADDR_WIDTH-1:0] ofmap_addr;

    wire valid;
    wire done;


    // DUT
    address_generator #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DIM_WIDTH(DIM_WIDTH),
        .NUM_PE(NUM_PE)
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .enable(enable),

        .ifmap_rows(ifmap_rows),
        .ifmap_cols(ifmap_cols),
        .kernel_rows(kernel_rows),
        .kernel_cols(kernel_cols),
        .input_channels(input_channels),
        .output_channels(output_channels),
        .stride(stride),

        .ifmap_base_addr(ifmap_base),
        .weight_base_addr(weight_base),
        .ofmap_base_addr(ofmap_base),

        .ifmap_addr(ifmap_addr),
        .weight_addr(weight_addr),
        .ofmap_addr(ofmap_addr),

        .valid(valid),
        .done(done)
    );


    // Clock
    always #5 clk = ~clk;


    integer i;

    initial begin

        clk = 0;
        rst = 1;
        start = 0;
        enable = 0;

        // Example layer
        ifmap_rows     = 8;
        ifmap_cols     = 8;

        kernel_rows    = 3;
        kernel_cols    = 3;

        input_channels = 2;
        output_channels = 2;

        stride = 1;

        // Base addresses
        ifmap_base  = 100;
        weight_base = 1000;
        ofmap_base  = 2000;


        // Reset
        #20;
        rst = 0;

        // Start AG
        #10;
        start = 1;
        enable = 1;

        #10;
        start = 0;


        // Monitor
        while (!done) begin

            @(posedge clk);

            if (valid) begin

                $display("\n--------------------------------");
                $display("Time = %0t", $time);

                $display("Weight Address = %0d", weight_addr);

                for (i = 0; i < NUM_PE; i = i + 1) begin
                    $display(
                        "PE%0d : IFMap Addr = %0d | OFMap Addr = %0d",
                        i,
                        ifmap_addr[i*ADDR_WIDTH +: ADDR_WIDTH],
                        ofmap_addr[i*ADDR_WIDTH +: ADDR_WIDTH]
                    );
                end
            end
        end


        $display("\n========== AG DONE ==========");

        #20;
        $finish;

    end

    initial begin
        $dumpfile("sim/waveform/address_generator_wave.vcd");
        $dumpvars(0, address_generator_tb);
    end

endmodule