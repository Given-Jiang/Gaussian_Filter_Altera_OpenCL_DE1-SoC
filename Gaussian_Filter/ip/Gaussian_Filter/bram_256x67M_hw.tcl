# TCL File Generated by Component Editor 12.0
# Wed Jul 25 19:01:12 EDT 2012
# DO NOT MODIFY


# 
# bram_256x67M "bram_256x67M" v1.0
# null 2012.07.25.19:01:12
# 
# 

# 
# request TCL package from ACDS 12.0
# 
package require -exact qsys 12.0


# 
# module bram_256x67M
# 
set_module_property NAME bram_256x67M
set_module_property VERSION 1.0
set_module_property INTERNAL false
set_module_property OPAQUE_ADDRESS_MAP true
set_module_property DISPLAY_NAME bram_256x67M
set_module_property INSTANTIATE_IN_SYSTEM_MODULE true
set_module_property EDITABLE true
set_module_property ANALYZE_HDL AUTO
set_module_property REPORT_TO_TALKBACK false
set_module_property ALLOW_GREYBOX_GENERATION false


# 
# file sets
# 
add_fileset QUARTUS_SYNTH QUARTUS_SYNTH "" ""
set_fileset_property QUARTUS_SYNTH TOP_LEVEL bram_256x67M
set_fileset_property QUARTUS_SYNTH ENABLE_RELATIVE_INCLUDE_PATHS false
add_fileset_file bram_256x67M.v VERILOG PATH bram_256x67M.v


# 
# parameters
# 
add_parameter DEPTH INTEGER 67108864
set_parameter_property DEPTH DEFAULT_VALUE 67108864
set_parameter_property DEPTH DISPLAY_NAME DEPTH
set_parameter_property DEPTH TYPE INTEGER
set_parameter_property DEPTH UNITS None
add_parameter NWORDS_A INTEGER 67108864
set_parameter_property NWORDS_A DEFAULT_VALUE 67108864
set_parameter_property NWORDS_A DISPLAY_NAME NWORDS_A
set_parameter_property NWORDS_A TYPE INTEGER
set_parameter_property NWORDS_A UNITS None
add_parameter ADDR_WIDTH INTEGER 26
set_parameter_property ADDR_WIDTH DEFAULT_VALUE 26
set_parameter_property ADDR_WIDTH DISPLAY_NAME ADDR_WIDTH
set_parameter_property ADDR_WIDTH TYPE INTEGER
set_parameter_property ADDR_WIDTH UNITS None


# 
# display items
# 


# 
# connection point avalon_slave_0
# 
add_interface avalon_slave_0 avalon end
set_interface_property avalon_slave_0 addressUnits WORDS
set_interface_property avalon_slave_0 associatedClock clock
set_interface_property avalon_slave_0 associatedReset reset
set_interface_property avalon_slave_0 bitsPerSymbol 8
set_interface_property avalon_slave_0 burstOnBurstBoundariesOnly false
set_interface_property avalon_slave_0 burstcountUnits WORDS
set_interface_property avalon_slave_0 explicitAddressSpan 0
set_interface_property avalon_slave_0 holdTime 0
set_interface_property avalon_slave_0 linewrapBursts false
set_interface_property avalon_slave_0 maximumPendingReadTransactions 0
set_interface_property avalon_slave_0 readLatency 2
set_interface_property avalon_slave_0 readWaitStates 0
set_interface_property avalon_slave_0 readWaitTime 0
set_interface_property avalon_slave_0 setupTime 0
set_interface_property avalon_slave_0 timingUnits Cycles
set_interface_property avalon_slave_0 writeWaitTime 0
set_interface_property avalon_slave_0 ENABLED true

add_interface_port avalon_slave_0 address address Input 26
add_interface_port avalon_slave_0 byteenable byteenable Input 32
add_interface_port avalon_slave_0 chipselect chipselect Input 1
add_interface_port avalon_slave_0 clken clken Input 1
add_interface_port avalon_slave_0 write write Input 1
add_interface_port avalon_slave_0 writedata writedata Input 256
add_interface_port avalon_slave_0 readdata readdata Output 256


# 
# connection point clock
# 
add_interface clock clock end
set_interface_property clock clockRate 0
set_interface_property clock ENABLED true

add_interface_port clock clk clk Input 1


# 
# connection point reset
# 
add_interface reset reset end
set_interface_property reset associatedClock clock
set_interface_property reset synchronousEdges DEASSERT
set_interface_property reset ENABLED true

add_interface_port reset reset reset Input 1

