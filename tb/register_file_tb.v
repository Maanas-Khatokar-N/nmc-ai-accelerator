module register_file_tb #(
    parameter DATA_WIDTH = 32,
    parameter DEPTH = 256,

    parameter NUM_READ_PORTS = 16,
    parameter NUM_WRITE_PORTS = 16,

    parameter ADDR_WIDTH = $clog2(DEPTH)
);

    reg clk, rst;

    reg [NUM_WRITE_PORTS-1:0] write_en;
    reg [ADDR_WIDTH*NUM_WRITE_PORTS-1:0] write_addr;
    reg [DATA_WIDTH*NUM_WRITE_PORTS-1:0] write_data;

    reg [ADDR_WIDTH*NUM_READ_PORTS-1:0] read_addr;
    wire [DATA_WIDTH*NUM_READ_PORTS-1:0] read_data;

    register_file #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH),
        .NUM_READ_PORTS(NUM_READ_PORTS),
        .NUM_WRITE_PORTS(NUM_WRITE_PORTS),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) dut(
        .clk(clk), 
        .rst(rst), 
        .write_en(write_en),
        .write_addr(write_addr),
        .write_data(write_data),
        .read_addr(read_addr),
        .read_data(read_data)
    );


    initial begin
        clk <= 1'b0;
        rst <= 1'b1;
        write_en = 0;
        write_addr = 0;
        write_data = 0;
        read_addr = 0;

        
    end

    always #5 clk <= ~clk;

    integer i;

    initial begin

        #12 rst = 0;
        
        #1 write_en = {NUM_WRITE_PORTS{1'b1}}; // Enable all write ports
        
        for (i = 0; i < NUM_WRITE_PORTS; i = i + 1) begin
            write_addr[i*ADDR_WIDTH +: ADDR_WIDTH] = i;              // Addr 0..15
            write_data[i*DATA_WIDTH +: DATA_WIDTH] = (i + 1) * 10;   // Data 10..160
        end

        #10 write_en = 0;

        for (i = 0; i < NUM_READ_PORTS; i = i + 1) begin
            read_addr[i*ADDR_WIDTH +: ADDR_WIDTH] = i;
        end

        #20 $finish;
    end

    initial begin
        $dumpfile("sim/waveform/register_file_wave.vcd");
        $dumpvars(0, register_file_tb);
    end
    
endmodule