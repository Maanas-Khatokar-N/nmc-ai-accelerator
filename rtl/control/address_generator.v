// AG ASSUMPTIONS:
// - 16 PEs arranged as a 4x4 output tile.
// - Weight-Unicast, Output-Stationary dataflow.
// - IFMap layout: [Channel][Row][Column].
// - Weight layout: [Output Channel][Input Channel][Kernel Row][Kernel Col].
// - OFMap layout: [Output Channel][Output Row][Output Column].
// - Kernel/stride/channels/dimensions are runtime configuration values.
// - Exact memory mapping and loop order are our implementation choices, not explicitly defined by the paper.
// - No padding is assumed currently.


module address_generator #(
    parameter ADDR_WIDTH = 16,
    parameter DIM_WIDTH  = 16,
    parameter NUM_PE     = 16,
    parameter PE_ROWS    = 4,
    parameter PE_COLS    = 4
)(
    input wire clk,
    input wire rst,

    input wire start,
    input wire enable,

    // Layer configuration
    input wire [DIM_WIDTH-1:0] ifmap_rows,
    input wire [DIM_WIDTH-1:0] ifmap_cols,

    input wire [DIM_WIDTH-1:0] kernel_rows,
    input wire [DIM_WIDTH-1:0] kernel_cols,

    input wire [DIM_WIDTH-1:0] input_channels,
    input wire [DIM_WIDTH-1:0] output_channels,

    input wire [DIM_WIDTH-1:0] stride,

    // Memory base addresses
    input wire [ADDR_WIDTH-1:0] ifmap_base_addr,
    input wire [ADDR_WIDTH-1:0] weight_base_addr,
    input wire [ADDR_WIDTH-1:0] ofmap_base_addr,

    // 16 IFMap addresses
    output reg [NUM_PE*ADDR_WIDTH-1:0] ifmap_addr,

    // One weight address - weight is broadcast
    output reg [ADDR_WIDTH-1:0] weight_addr,

    // 16 OFMap addresses
    output reg [NUM_PE*ADDR_WIDTH-1:0] ofmap_addr,

    output reg valid,
    output reg done
);

    // ------------------------------------------------------------
    // Counters
    // ------------------------------------------------------------

    reg [DIM_WIDTH-1:0] kernel_row;
    reg [DIM_WIDTH-1:0] kernel_col;

    reg [DIM_WIDTH-1:0] input_channel;
    reg [DIM_WIDTH-1:0] output_channel;

    reg [DIM_WIDTH-1:0] output_row;
    reg [DIM_WIDTH-1:0] output_col;

    reg running;


    // ------------------------------------------------------------
    // Derived output dimensions
    //
    // OFMap = (IFMap - Kernel) / Stride + 1
    // ------------------------------------------------------------

    wire [DIM_WIDTH-1:0] ofmap_rows;
    wire [DIM_WIDTH-1:0] ofmap_cols;

    assign ofmap_rows =
        ((ifmap_rows - kernel_rows) / stride) + 1;

    assign ofmap_cols =
        ((ifmap_cols - kernel_cols) / stride) + 1;


    // ------------------------------------------------------------
    // Address generation
    // ------------------------------------------------------------

    integer i;
    integer pe_row;
    integer pe_col;

    reg [ADDR_WIDTH-1:0] temp_ifmap_addr;
    reg [ADDR_WIDTH-1:0] temp_ofmap_addr;


    always @(posedge clk or posedge rst) begin

        if (rst) begin

            kernel_row   <= 0;
            kernel_col   <= 0;

            input_channel  <= 0;
            output_channel <= 0;

            output_row <= 0;
            output_col <= 0;

            ifmap_addr <= 0;
            weight_addr <= 0;
            ofmap_addr <= 0;

            valid <= 0;
            done  <= 0;

            running <= 0;
        end

        else begin

            valid <= 0;
            done  <= 0;


            // ----------------------------------------------------
            // Start
            // ----------------------------------------------------

            if (start && !running) begin

                kernel_row    <= 0;
                kernel_col    <= 0;

                input_channel  <= 0;
                output_channel <= 0;

                output_row <= 0;
                output_col <= 0;

                running <= 1;
            end


            // ----------------------------------------------------
            // Generate addresses
            // ----------------------------------------------------

            else if (running && enable) begin

                // ------------------------------------------------
                // Weight address
                //
                // [output_channel]
                // [input_channel]
                // [kernel_row]
                // [kernel_col]
                // ------------------------------------------------

                weight_addr <= weight_base_addr
                    + output_channel
                        * input_channels
                        * kernel_rows
                        * kernel_cols
                    + input_channel
                        * kernel_rows
                        * kernel_cols
                    + kernel_row
                        * kernel_cols
                    + kernel_col;


                // ------------------------------------------------
                // Generate 16 IFMap addresses
                // One address for each PE
                // ------------------------------------------------

                for (i = 0; i < NUM_PE; i = i + 1) begin

                    pe_row = i / PE_COLS;
                    pe_col = i % PE_COLS;

                    temp_ifmap_addr =
                        ifmap_base_addr
                        + input_channel
                            * ifmap_rows
                            * ifmap_cols
                        + (output_row * stride
                           + pe_row * stride
                           + kernel_row)
                            * ifmap_cols
                        + (output_col * stride
                           + pe_col * stride
                           + kernel_col);

                    ifmap_addr[
                        i*ADDR_WIDTH +: ADDR_WIDTH
                    ] <= temp_ifmap_addr;
                end


                // ------------------------------------------------
                // Generate 16 OFMap addresses
                //
                // Each PE corresponds to one output position
                // ------------------------------------------------

                for (i = 0; i < NUM_PE; i = i + 1) begin

                    pe_row = i / PE_COLS;
                    pe_col = i % PE_COLS;

                    temp_ofmap_addr =
                        ofmap_base_addr
                        + output_channel
                            * ofmap_rows
                            * ofmap_cols
                        + (output_row + pe_row)
                            * ofmap_cols
                        + (output_col + pe_col);

                    ofmap_addr[
                        i*ADDR_WIDTH +: ADDR_WIDTH
                    ] <= temp_ofmap_addr;
                end


                valid <= 1;


                // ------------------------------------------------
                // Move to next kernel element
                // ------------------------------------------------

                if (kernel_col < kernel_cols - 1) begin

                    kernel_col <= kernel_col + 1;

                end

                else begin

                    kernel_col <= 0;

                    if (kernel_row < kernel_rows - 1) begin

                        kernel_row <= kernel_row + 1;

                    end

                    else begin

                        kernel_row <= 0;


                        // ----------------------------------------
                        // One complete kernel processed for
                        // current input channel
                        // ----------------------------------------

                        if (input_channel < input_channels - 1) begin

                            input_channel <= input_channel + 1;

                        end

                        else begin

                            input_channel <= 0;


                            // ------------------------------------
                            // Move to next output tile
                            // ------------------------------------

                            if (output_col + PE_COLS < ofmap_cols) begin

                                output_col <= output_col + PE_COLS;

                            end

                            else begin

                                output_col <= 0;


                                if (output_row + PE_ROWS < ofmap_rows) begin

                                    output_row <= output_row + PE_ROWS;

                                end

                                else begin

                                    output_row <= 0;


                                    // ----------------------------
                                    // Next output channel
                                    // ----------------------------

                                    if (output_channel <
                                        output_channels - 1) begin

                                        output_channel <=
                                            output_channel + 1;

                                    end

                                    else begin

                                        // ------------------------
                                        // Entire layer completed
                                        // ------------------------

                                        output_channel <= 0;

                                        running <= 0;
                                        done <= 1;

                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

endmodule