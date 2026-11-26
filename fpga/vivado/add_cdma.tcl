# ============================================================================
# Add an AXI CDMA to the nmc_system block design (PYNQ-Z2 speed-up, see fpga/FPGA_SPEEDUP.md)
#
#   DDR <-- S_AXI_HP0 <-- smartconnect_dma <-- axi_cdma_0/M_AXI
#                                 |
#                                 +--> smartconnect_0 (new Sxx) --> nmc_axi_wrapper_0/s_axi_mem
#   ARM GP0 --> smartconnect_0 (new Mxx) --> axi_cdma_0/S_AXI_LITE   (CDMA registers)
#
# Usage: open the project, then in the Tcl console
#     set FCLK_MHZ 66.666666         ;# optional, default 66.67 MHz (15 ns). 71.428571 = 14 ns (try later)
#     source <repo>/fpga/vivado/add_cdma.tcl
# then: Generate Block Design -> Run Synthesis/Implementation -> Generate Bitstream,
#       copy nmc.bit + nmc.hwh to the board.
# The ARM-side address map (cfg 0x4000_0000, mem 0x4008_0000) is not changed.
# ============================================================================
if {![info exists FCLK_MHZ]} { set FCLK_MHZ 66.666666 }

set bdf [lindex [get_files -quiet -filter {FILE_TYPE == "Block Designs"} *nmc_system.bd] 0]
if {$bdf eq ""} { set bdf [lindex [get_files -filter {FILE_TYPE == "Block Designs"}] 0] }
open_bd_design $bdf
current_bd_instance /

set ps   [get_bd_cells processing_system7_0]
set sc   [get_bd_cells smartconnect_0]
set wrap [get_bd_cells nmc_axi_wrapper_0]
if {$ps eq "" || $sc eq "" || $wrap eq ""} { error "expected cells processing_system7_0, smartconnect_0, nmc_axi_wrapper_0" }
if {[get_bd_cells -quiet axi_cdma_0] ne ""} { error "axi_cdma_0 already exists - nothing to do" }

# ---- PS: HP0 slave port + FCLK0 frequency ----
set_property -dict [list CONFIG.PCW_USE_S_AXI_HP0 {1} CONFIG.PCW_S_AXI_HP0_DATA_WIDTH {64} \
                         CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ $FCLK_MHZ] $ps

# ---- CDMA: simple mode, 32-bit, 256-beat bursts (s_axi_mem is 32 bit) ----
set cdma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_cdma axi_cdma_0]
set_property -dict [list CONFIG.C_INCLUDE_SG {0} CONFIG.C_M_AXI_DATA_WIDTH {32} \
                         CONFIG.C_M_AXI_MAX_BURST_LEN {256} CONFIG.C_INCLUDE_DRE {0} \
                         CONFIG.C_USE_DATAMOVER_LITE {0} CONFIG.C_ADDR_WIDTH {32}] $cdma

# ---- interconnect ----
set scd [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect smartconnect_dma]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] $scd

set nsi [get_property CONFIG.NUM_SI $sc]
set nmi [get_property CONFIG.NUM_MI $sc]
set_property -dict [list CONFIG.NUM_SI [expr {$nsi + 1}] CONFIG.NUM_MI [expr {$nmi + 1}]] $sc
set si_new [format "S%02d_AXI" $nsi]
set mi_new [format "M%02d_AXI" $nmi]

connect_bd_intf_net [get_bd_intf_pins axi_cdma_0/M_AXI]         [get_bd_intf_pins smartconnect_dma/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smartconnect_dma/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
connect_bd_intf_net [get_bd_intf_pins smartconnect_dma/M01_AXI] [get_bd_intf_pins smartconnect_0/$si_new]
connect_bd_intf_net [get_bd_intf_pins smartconnect_0/$mi_new]   [get_bd_intf_pins axi_cdma_0/S_AXI_LITE]

# ---- clock / reset: everything on FCLK_CLK0, same resets as the existing logic ----
set clknet [get_bd_nets -of_objects [get_bd_pins processing_system7_0/FCLK_CLK0]]
connect_bd_net -net $clknet [get_bd_pins axi_cdma_0/m_axi_aclk] [get_bd_pins axi_cdma_0/s_axi_lite_aclk] \
                            [get_bd_pins smartconnect_dma/aclk] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK]
set icrst [get_bd_nets -of_objects [get_bd_pins smartconnect_0/aresetn]]
connect_bd_net -net $icrst [get_bd_pins smartconnect_dma/aresetn]
set perst [get_bd_nets -of_objects [get_bd_pins nmc_axi_wrapper_0/aresetn]]
connect_bd_net -net $perst [get_bd_pins axi_cdma_0/s_axi_lite_aresetn]

# ---- addresses ----
assign_bd_address
# the CDMA must reach the SRAM window at the same address the ARM uses (0x4008_0000) and DDR at 0
foreach seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces axi_cdma_0/Data]] {
    set tgt [get_bd_addr_segs -of_objects $seg]
    if {[string match "*nmc_axi_wrapper_0/s_axi_mem*" $tgt]} {
        set_property offset 0x40080000 $seg
    }
}
puts "\n==== address map of axi_cdma_0/Data (s_axi_mem must be 0x40080000, HP0 DDR 0x00000000) ===="
foreach seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces axi_cdma_0/Data]] {
    puts [format "  %-60s 0x%08x  %s" $seg [get_property offset $seg] [get_property range $seg]]
}
puts "==== address map of processing_system7_0/Data ===="
foreach seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces processing_system7_0/Data]] {
    puts [format "  %-60s 0x%08x  %s" $seg [get_property offset $seg] [get_property range $seg]]
}

validate_bd_design
save_bd_design
puts "\nCDMA added. FCLK0 requested: $FCLK_MHZ MHz. Now: Generate Block Design, synthesis, implementation, bitstream."
