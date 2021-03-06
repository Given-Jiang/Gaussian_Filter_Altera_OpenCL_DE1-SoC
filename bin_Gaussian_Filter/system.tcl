package require -exact qsys 12.1
set_validation_property AUTOMATIC_VALIDATION false

add_instance Gaussian_Filter_system Gaussian_Filter_system

add_connection acl_iface.kernel_clk Gaussian_Filter_system.clock_reset
add_connection acl_iface.kernel_clk2x Gaussian_Filter_system.clock_reset2x
add_connection acl_iface.kernel_reset Gaussian_Filter_system.clock_reset_reset

add_connection Gaussian_Filter_system.avm_memgmem0_port_0_0_rw acl_iface.kernel_mem0
add_connection acl_iface.kernel_irq Gaussian_Filter_system.kernel_irq

add_instance cra_root cra_ring_root
set_instance_parameter_value cra_root DATA_W 64
set_instance_parameter_value cra_root ADDR_W 4
set_instance_parameter_value cra_root ID_W 0
add_connection acl_iface.kernel_clk cra_root.clock
add_connection acl_iface.kernel_reset cra_root.reset
add_connection acl_iface.kernel_cra cra_root.cra_slave
set_connection_parameter_value acl_iface.kernel_cra/cra_root.cra_slave baseAddress "0x0"
add_instance avs_Gaussian_cra_cra_ring cra_ring_node
set_instance_parameter_value avs_Gaussian_cra_cra_ring DATA_W 64
set_instance_parameter_value avs_Gaussian_cra_cra_ring RING_ADDR_W 4
set_instance_parameter_value avs_Gaussian_cra_cra_ring CRA_ADDR_W 4
set_instance_parameter_value avs_Gaussian_cra_cra_ring ID_W 0
set_instance_parameter_value avs_Gaussian_cra_cra_ring ID 0
add_connection acl_iface.kernel_clk avs_Gaussian_cra_cra_ring.clock
add_connection acl_iface.kernel_reset avs_Gaussian_cra_cra_ring.reset
add_connection cra_root.ring_out avs_Gaussian_cra_cra_ring.ring_in
add_connection avs_Gaussian_cra_cra_ring.cra_master Gaussian_Filter_system.avs_Gaussian_cra
set_connection_parameter_value avs_Gaussian_cra_cra_ring.cra_master/Gaussian_Filter_system.avs_Gaussian_cra baseAddress "0x0"
add_connection avs_Gaussian_cra_cra_ring.ring_out cra_root.ring_in


save_system
