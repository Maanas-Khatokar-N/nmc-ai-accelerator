# Groups ALL failing setup endpoints of the open (routed) design by register name and reports
# worst slack + count per group. The Vivado summary only shows the worst 10 paths.
#   Vivado Tcl console:  source scripts/timing_endpoints.tcl
set report_dir [file normalize "K:/IIT Dh/RnD/Vivado/nmc_accelerator/reports"]
file mkdir $report_dir
set out [file join $report_dir "timing_endpoints.txt"]
set paths [get_timing_paths -max_paths 5000 -nworst 1 -slack_lesser_than 0 -unique_pins]
array set worst {}
array set count {}
foreach p $paths {
    set ep   [get_property ENDPOINT_PIN $p]
    set slk  [get_property SLACK $p]
    set cell [join [lrange [split $ep "/"] 0 end-1] "/"]
    regsub -all {\[[0-9]+\]} $cell {[*]} key          ;# bank/bit indexes collapsed
    regsub {^.*/inst/} $key {} key                    ;# drop the block-design prefix
    if {![info exists count($key)]} { set count($key) 0; set worst($key) $slk }
    incr count($key)
    if {$slk < $worst($key)} { set worst($key) $slk }
}
set fh [open $out w]
puts $fh [format "%-12s %-8s %s" "worst_slack" "count" "endpoint register group"]
set rows {}
foreach k [array names worst] { lappend rows [list $worst($k) $count($k) $k] }
foreach r [lsort -real -index 0 $rows] { puts $fh [format "%-12.3f %-8d %s" {*}$r] }
close $fh
puts "wrote $out  ([llength $paths] failing endpoints, [array size worst] groups)"