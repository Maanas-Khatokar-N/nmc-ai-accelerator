// ============================================================================
// NMC cluster : NUM_BANKS x ( sram_bank + nmc_unit ), running concurrently
//
//   host / RISC-V core port ---> bank decode ---> sram_bank[b].core_*
//   nmc_unit[b] <----------------------------> sram_bank[b].nmc_*   (private)
//
// Paper Sec. III / Fig. 3: one NMC unit per bank, each a master on its own bank,
// each independently configured (own layer descriptor, own start). The units
// never touch another bank, so they need no arbitration between each other;
// the only contention is core-vs-NMC inside a bank (handled by sram_bank).
//
// Host address map (word addresses, one element per 32-bit word):
//     core_addr = { bank[BANK_W-1:0], word_in_bank[CORE_AW-1:0] }
// i.e. each bank is a contiguous region. This is the "one tile per bank" layout
// the bank-tiling strategy needs (a tile must live entirely in ONE bank, because
// its NMC unit can only see that bank). The low-order macro interleave INSIDE a
// bank is unchanged.
//
// Control:
//   unit_mask : which units take part in this round (static while running)
//   start     : 1-cycle pulse; starts every unit in unit_mask at the same edge
//   accumulate: per unit, static while running (Cin pass 1.. of a job)
//   irq       : 1-cycle pulse when every masked unit has raised done
//               (the "layer done" interrupt towards the RISC-V)
//   unit_busy : per unit, high from start until its own done
//
// Descriptor ports are packed, one field per bank slice [b*W +: W]. Every unit
// has its own descriptor because edge tiles / kernel tiles differ in size.
// ============================================================================
module nmc_cluster #(
    parameter NUM_BANKS       = 4,
    parameter ADDR_WIDTH      = 17,
    parameter DIM_WIDTH       = 16,
    parameter RF_AW           = 8,
    parameter NUM_MACROS      = 8,
    parameter DEPTH_PER_MACRO = 4096,
    parameter MACRO_AW        = $clog2(DEPTH_PER_MACRO),
    parameter CNT_W           = 48,
    parameter BANK_W          = $clog2(NUM_BANKS),
    parameter CORE_AW         = $clog2(NUM_MACROS) + MACRO_AW
)(
    input  wire                         clk,
    input  wire                         rst,

    // ---------------- control ----------------
    input  wire [NUM_BANKS-1:0]         unit_mask,
    input  wire                         start,
    input  wire [NUM_BANKS-1:0]         accumulate,
    output reg                          irq,
    output wire [NUM_BANKS-1:0]         unit_busy,

    // ---------------- per-unit layer descriptors (packed) ----------------
    input  wire [NUM_BANKS*DIM_WIDTH-1:0]  ifmap_cols,
    input  wire [NUM_BANKS*DIM_WIDTH-1:0]  kernel_rows,
    input  wire [NUM_BANKS*DIM_WIDTH-1:0]  kernel_cols,
    input  wire [NUM_BANKS*DIM_WIDTH-1:0]  input_channels,
    input  wire [NUM_BANKS*DIM_WIDTH-1:0]  output_channels,
    input  wire [NUM_BANKS*DIM_WIDTH-1:0]  stride,
    input  wire [NUM_BANKS*DIM_WIDTH-1:0]  ofmap_rows,
    input  wire [NUM_BANKS*DIM_WIDTH-1:0]  ofmap_cols,
    input  wire [NUM_BANKS*DIM_WIDTH-1:0]  weights_per_oc,
    input  wire [NUM_BANKS*ADDR_WIDTH-1:0] ifmap_ch_size,
    input  wire [NUM_BANKS*ADDR_WIDTH-1:0] ofmap_ch_size,
    input  wire [NUM_BANKS*ADDR_WIDTH-1:0] ifmap_row_step,
    input  wire [NUM_BANKS*ADDR_WIDTH-1:0] ifmap_base_addr,
    input  wire [NUM_BANKS*ADDR_WIDTH-1:0] weight_base_addr,
    input  wire [NUM_BANKS*ADDR_WIDTH-1:0] ofmap_base_addr,
    input  wire [NUM_BANKS*RF_AW-1:0]      rf_row_step,
    input  wire [NUM_BANKS*RF_AW-1:0]      rf_ch_stride,

    // ---------------- host (RISC-V / DMA) memory port ----------------
    input  wire                         core_req,
    input  wire                         core_we,
    input  wire [BANK_W+CORE_AW-1:0]    core_addr,
    input  wire [31:0]                  core_wdata,
    output wire [31:0]                  core_rdata,
    output wire                         core_rvalid,

    // ---------------- performance (packed per bank) ----------------
    output wire [NUM_BANKS-1:0]         perf_valid,
    output wire [NUM_BANKS*CNT_W-1:0]   perf_cyc_total,
    output wire [NUM_BANKS*CNT_W-1:0]   perf_cyc_mac,
    output wire [NUM_BANKS*CNT_W-1:0]   perf_cyc_stall,
    output wire [NUM_BANKS*CNT_W-1:0]   perf_cyc_mem_blocked,
    output wire [NUM_BANKS*CNT_W-1:0]   perf_mac_ops
);

    // ------------------------------------------------------------------
    // Round control: start the masked units together, irq when all are done
    // ------------------------------------------------------------------
    wire [NUM_BANKS-1:0] unit_done;
    reg  [NUM_BANKS-1:0] done_q;
    reg                  running;
    reg  [NUM_BANKS-1:0] busy_q;

    wire go = start && !running && (|unit_mask);
    wire [NUM_BANKS-1:0] unit_start = {NUM_BANKS{go}} & unit_mask;
    wire [NUM_BANKS-1:0] done_now   = done_q | unit_done;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            done_q <= 0; running <= 1'b0; irq <= 1'b0; busy_q <= 0;
        end else begin
            irq    <= 1'b0;
            busy_q <= (busy_q | unit_start) & ~unit_done;
            if (go) begin
                running <= 1'b1;
                done_q  <= 0;
            end else if (running) begin
                done_q <= done_now;
                if (&(done_now | ~unit_mask)) begin
                    running <= 1'b0;
                    irq     <= 1'b1;
                end
            end
        end
    end
    assign unit_busy = busy_q;

    // ------------------------------------------------------------------
    // Host port decode
    // ------------------------------------------------------------------
    wire [BANK_W-1:0] core_bank = core_addr[CORE_AW +: BANK_W];
    wire [CORE_AW-1:0] core_loc = core_addr[CORE_AW-1:0];
    reg  [BANK_W-1:0] core_bank_q;
    always @(posedge clk) core_bank_q <= core_bank;

    wire [NUM_BANKS*32-1:0] b_core_rdata;
    wire [NUM_BANKS-1:0]    b_core_rvalid;
    assign core_rdata  = b_core_rdata[core_bank_q*32 +: 32];
    assign core_rvalid = |b_core_rvalid;

    // ------------------------------------------------------------------
    // Banks + units
    // ------------------------------------------------------------------
    genvar b;
    generate for (b = 0; b < NUM_BANKS; b = b + 1) begin : g_bank

        wire [NUM_MACROS-1:0]          nmc_req, nmc_we, nmc_grant, nmc_blocked, nmc_rvalid;
        wire [NUM_MACROS*MACRO_AW-1:0] nmc_addr;
        wire [NUM_MACROS*32-1:0]       nmc_wdata, nmc_rdata;

        sram_bank #(.DATA_WIDTH(32), .NUM_MACROS(NUM_MACROS),
                    .DEPTH_PER_MACRO(DEPTH_PER_MACRO)) u_bank (
            .clk(clk), .rst(rst),
            .core_req(core_req && (core_bank == b)), .core_we(core_we),
            .core_addr(core_loc), .core_wdata(core_wdata),
            .core_rdata(b_core_rdata[b*32 +: 32]), .core_rvalid(b_core_rvalid[b]),
            .nmc_req(nmc_req), .nmc_we(nmc_we), .nmc_addr(nmc_addr), .nmc_wdata(nmc_wdata),
            .nmc_grant(nmc_grant), .nmc_blocked(nmc_blocked),
            .nmc_rdata(nmc_rdata), .nmc_rvalid(nmc_rvalid));

        nmc_unit #(.ADDR_WIDTH(ADDR_WIDTH), .DIM_WIDTH(DIM_WIDTH), .RF_AW(RF_AW),
                   .NUM_MACROS(NUM_MACROS), .MACRO_AW(MACRO_AW), .CNT_W(CNT_W)) u_unit (
            .clk(clk), .rst(rst), .start(unit_start[b]), .done(unit_done[b]),
            .accumulate(accumulate[b]),
            .ifmap_cols      (ifmap_cols      [b*DIM_WIDTH  +: DIM_WIDTH ]),
            .kernel_rows     (kernel_rows     [b*DIM_WIDTH  +: DIM_WIDTH ]),
            .kernel_cols     (kernel_cols     [b*DIM_WIDTH  +: DIM_WIDTH ]),
            .input_channels  (input_channels  [b*DIM_WIDTH  +: DIM_WIDTH ]),
            .output_channels (output_channels [b*DIM_WIDTH  +: DIM_WIDTH ]),
            .stride          (stride          [b*DIM_WIDTH  +: DIM_WIDTH ]),
            .ofmap_rows      (ofmap_rows      [b*DIM_WIDTH  +: DIM_WIDTH ]),
            .ofmap_cols      (ofmap_cols      [b*DIM_WIDTH  +: DIM_WIDTH ]),
            .weights_per_oc  (weights_per_oc  [b*DIM_WIDTH  +: DIM_WIDTH ]),
            .ifmap_ch_size   (ifmap_ch_size   [b*ADDR_WIDTH +: ADDR_WIDTH]),
            .ofmap_ch_size   (ofmap_ch_size   [b*ADDR_WIDTH +: ADDR_WIDTH]),
            .ifmap_row_step  (ifmap_row_step  [b*ADDR_WIDTH +: ADDR_WIDTH]),
            .ifmap_base_addr (ifmap_base_addr [b*ADDR_WIDTH +: ADDR_WIDTH]),
            .weight_base_addr(weight_base_addr[b*ADDR_WIDTH +: ADDR_WIDTH]),
            .ofmap_base_addr (ofmap_base_addr [b*ADDR_WIDTH +: ADDR_WIDTH]),
            .rf_row_step     (rf_row_step     [b*RF_AW      +: RF_AW     ]),
            .rf_ch_stride    (rf_ch_stride    [b*RF_AW      +: RF_AW     ]),
            .nmc_req(nmc_req), .nmc_we(nmc_we), .nmc_addr(nmc_addr), .nmc_wdata(nmc_wdata),
            .nmc_grant(nmc_grant), .nmc_rdata(nmc_rdata),
            .dbg_compute_start(), .dbg_compute_done(), .dbg_mac_en(), .dbg_stall(), .dbg_acc_out(),
            .perf_valid(perf_valid[b]),
            .perf_cyc_total      (perf_cyc_total      [b*CNT_W +: CNT_W]),
            .perf_cyc_load       (),
            .perf_cyc_compute    (),
            .perf_cyc_store      (),
            .perf_cyc_mac        (perf_cyc_mac        [b*CNT_W +: CNT_W]),
            .perf_cyc_stall      (perf_cyc_stall      [b*CNT_W +: CNT_W]),
            .perf_cyc_mem_blocked(perf_cyc_mem_blocked[b*CNT_W +: CNT_W]),
            .perf_mac_ops        (perf_mac_ops        [b*CNT_W +: CNT_W]));
    end endgenerate

endmodule