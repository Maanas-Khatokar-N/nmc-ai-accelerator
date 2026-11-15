
# ============================================================
# Group ALL failing setup endpoints by register name
# Reports worst slack and number of failing endpoints
#
# Execute AFTER opening the latest routed implementation:
# open_run impl_1
# source scripts/timing_endpoints.tcl
# ============================================================

set report_dir [file normalize "K:/IIT Dh/RnD/Vivado/nmc_accelerator/reports"]
file mkdir $report_dir

set out [file join $report_dir "timing_endpoints.txt"]

# Refresh timing information
update_timing

# Retrieve failing setup paths
set paths [get_timing_paths \
    -delay_type max \
    -max_paths 100000 \
    -nworst 1 \
    -slack_lesser_than 0.0 \
    -unique_pins]

array set worst {}
array set count {}

foreach p $paths {

    set ep  [get_property ENDPOINT_PIN $p]
    set slk [get_property SLACK $p]

    # Extract endpoint register/cell name
    set cell [join [lrange [split $ep "/"] 0 end-1] "/"]

    # Collapse bank/bit indices
    regsub -all {\[[0-9]+\]} $cell {[*]} key

    # Remove block-design hierarchy
    regsub {^.*/inst/} $key {} key

    # Initialize group
    if {![info exists count($key)]} {
        set count($key) 0
        set worst($key) $slk
    }

    incr count($key)

    # Update worst slack
    if {$slk < $worst($key)} {
        set worst($key) $slk
    }
}

# Write report (w truncates previous report)
set fh [open $out w]

puts $fh "Timing endpoint analysis"
puts $fh "Design: [current_design]"
puts $fh "Total failing paths: [llength $paths]"
puts $fh "Total register groups: [array size worst]"
puts $fh ""

puts $fh [format "%-15s %-10s %s" \
    "Worst_Slack" "Count" "Endpoint_Register_Group"]

set rows {}

foreach k [array names worst] {
    lappend rows [list $worst($k) $count($k) $k]
}

# Sort by worst slack (most negative first)
foreach r [lsort -real -index 0 $rows] {
    puts $fh [format "%-15.3f %-10d %s" {*}$r]
}

close $fh

puts "------------------------------------------"
puts "Timing endpoint analysis completed."
puts "Failing paths : [llength $paths]"
puts "Register groups: [array size worst]"
puts "Report: $out"
puts "------------------------------------------"
