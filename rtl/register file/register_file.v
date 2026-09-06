module register_file #(
    parameter DATA_WIDTH = 32,
    parameter DEPTH = 256,

    parameter NUM_READ_PORTS = 16,
    parameter NUM_WRITE_PORTS = 16,

    parameter ADDR_WIDTH = $clog2(DEPTH)
) (
    input clk, rst,

    //Write ports
    input [NUM_WRITE_PORTS-1:0] write_en,
    input [ADDR_WIDTH*NUM_WRITE_PORTS-1:0] write_addr,
    input [DATA_WIDTH*NUM_WRITE_PORTS-1:0] write_data,

    //Read ports
    input [ADDR_WIDTH*NUM_READ_PORTS-1:0] read_addr,
    output reg [DATA_WIDTH*NUM_READ_PORTS-1:0] read_data
);

    //Register size: DEPTH x DATA_WIDTH (eg. 256 x 32)
    reg [DATA_WIDTH-1:0] memory [0:DEPTH-1];


    //Asynchronous read
    integer i;

    always @(*) begin
        for (i = 0; i < NUM_READ_PORTS; i=i+1) begin

            //Read data for each port using read_addr 
            read_data[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] = memory[read_addr[(i+1)*ADDR_WIDTH-1 -: ADDR_WIDTH]];
        end
    end


    integer j;
    integer k;

    always @(posedge clk or posedge rst) begin

        //Reset
        if (rst) begin
            for (j = 0; j < DEPTH; j=j+1) begin
                memory[j] <= {DATA_WIDTH{1'b0}};       //Reset memory -> Read ports also gets reset
            end
        end

        //Synchronous write
        else begin
            for (k = 0; k < NUM_WRITE_PORTS; k=k+1) begin
                if (write_en[k]) begin

                    //Write data for each port using write_addr 
                    memory[write_addr[(k+1)*ADDR_WIDTH-1 -: ADDR_WIDTH]] <= write_data[(k+1)*DATA_WIDTH-1 -: DATA_WIDTH];
                end
            end
        end
    end

endmodule