//Copyright (C) 1991-2012 Altera Corporation
//Your use of Altera Corporation's design tools, logic functions 
//and other software and tools, and its AMPP partner logic 
//functions, and any output files from any of the foregoing 
//(including device programming or simulation files), and any 
//associated documentation or information are expressly subject 
//to the terms and conditions of the Altera Program License 
//Subscription Agreement, Altera MegaCore Function License 
//Agreement, or other applicable license agreement, including, 
//without limitation, that your use is for the sole purpose of 
//programming logic devices manufactured by Altera and sold by 
//Altera or its authorized distributors.  Please refer to the 
//applicable agreement for further details.


//altfp_mult CBX_AUTO_BLACKBOX="ALL" DEDICATED_MULTIPLIER_CIRCUITRY="YES" DENORMAL_SUPPORT="NO" DEVICE_FAMILY="Stratix V" EXCEPTION_HANDLING="NO" PIPELINE=6 REDUCED_FUNCTIONALITY="NO" ROUNDING="TO_NEAREST" WIDTH_EXP=11 WIDTH_MAN=52 clk_en clock dataa datab result
//VERSION_BEGIN 12.0 cbx_alt_ded_mult_y 2012:05:31:20:08:01:SJ cbx_altbarrel_shift 2012:05:31:20:08:02:SJ cbx_altera_mult_add 2012:05:31:20:08:02:SJ cbx_altfp_mult 2012:05:31:20:08:02:SJ cbx_altmult_add 2012:05:31:20:08:02:SJ cbx_cycloneii 2012:05:31:20:08:02:SJ cbx_lpm_add_sub 2012:05:31:20:08:02:SJ cbx_lpm_compare 2012:05:31:20:08:02:SJ cbx_lpm_mult 2012:05:31:20:08:02:SJ cbx_mgl 2012:05:31:20:10:16:SJ cbx_padd 2012:05:31:20:08:02:SJ cbx_parallel_add 2012:05:31:20:08:02:SJ cbx_stratix 2012:05:31:20:08:02:SJ cbx_stratixii 2012:05:31:20:08:02:SJ cbx_util_mgl 2012:05:31:20:08:02:SJ  VERSION_END
// synthesis VERILOG_INPUT_VERSION VERILOG_2001
// altera message_off 10463


//synthesis_resources = lpm_add_sub 4 lpm_mult 1 reg 252 
//synopsys translate_off
`timescale 1 ps / 1 ps
//synopsys translate_on
module  acl_fp_mul_ll_s5_double_altfp_mult_abo
	( 
	clk_en,
	clock,
	dataa,
	datab,
	result) ;
	input   clk_en;
	input   clock;
	input   [63:0]  dataa;
	input   [63:0]  datab;
	output   [63:0]  result;
`ifndef ALTERA_RESERVED_QIS
// synopsys translate_off
`endif
	tri1   clk_en;
`ifndef ALTERA_RESERVED_QIS
// synopsys translate_on
`endif

	reg	dataa_exp_all_one_ff_p1;
	reg	dataa_exp_not_zero_ff_p1;
	reg	dataa_man_not_zero_ff_p1;
	reg	dataa_man_not_zero_ff_p2;
	reg	datab_exp_all_one_ff_p1;
	reg	datab_exp_not_zero_ff_p1;
	reg	datab_man_not_zero_ff_p1;
	reg	datab_man_not_zero_ff_p2;
	reg	[12:0]	delay_exp2_bias;
	reg	[12:0]	delay_exp3_bias;
	reg	[12:0]	delay_exp_bias;
	reg	delay_man_product_msb;
	reg	delay_man_product_msb_p0;
	reg	[11:0]	exp_add_p1;
	reg	[10:0]	exp_result_ff;
	reg	input_is_infinity_dffe_0;
	reg	input_is_infinity_dffe_1;
	reg	input_is_infinity_ff1;
	reg	input_is_infinity_ff2;
	reg	input_is_nan_dffe_0;
	reg	input_is_nan_dffe_1;
	reg	input_is_nan_ff1;
	reg	input_is_nan_ff2;
	reg	input_not_zero_dffe_0;
	reg	input_not_zero_dffe_1;
	reg	input_not_zero_ff1;
	reg	input_not_zero_ff2;
	reg	lsb_dffe;
	reg	[51:0]	man_result_ff;
	reg	[52:0]	man_round_p;
	reg	[53:0]	man_round_p2;
	reg	round_dffe;
	reg	[0:0]	sign_node_ff0;
	reg	[0:0]	sign_node_ff1;
	reg	[0:0]	sign_node_ff2;
	reg	[0:0]	sign_node_ff3;
	reg	[0:0]	sign_node_ff4;
	reg	[0:0]	sign_node_ff5;
	reg	sticky_dffe;
	wire  [11:0]   wire_exp_add_adder_result;
	wire  [12:0]   wire_exp_adj_adder_result;
	wire  [12:0]   wire_exp_bias_subtr_result;
	wire  [53:0]   wire_man_round_adder_result;
	wire  [105:0]   wire_man_product2_mult_result;
	wire aclr;
	wire  [12:0]  bias;
	wire  [10:0]  dataa_exp_all_one;
	wire  [10:0]  dataa_exp_not_zero;
	wire  [51:0]  dataa_man_not_zero;
	wire  [10:0]  datab_exp_all_one;
	wire  [10:0]  datab_exp_not_zero;
	wire  [51:0]  datab_man_not_zero;
	wire  [52:0]  delay_round_wire;
	wire  exp_is_inf;
	wire  exp_is_zero;
	wire  [12:0]  expmod;
	wire  [10:0]  inf_num;
	wire  lsb_bit;
	wire  [53:0]  man_shift_full;
	wire  [10:0]  result_exp_all_one;
	wire  [11:0]  result_exp_not_zero;
	wire  round_bit;
	wire  round_carry;
	wire  [51:0]  sticky_bit;

	// synopsys translate_off
	initial
		dataa_exp_all_one_ff_p1 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) dataa_exp_all_one_ff_p1 <= 1'b0;
		else if  (clk_en == 1'b1)   dataa_exp_all_one_ff_p1 <= dataa_exp_all_one[10];
	// synopsys translate_off
	initial
		dataa_exp_not_zero_ff_p1 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) dataa_exp_not_zero_ff_p1 <= 1'b0;
		else if  (clk_en == 1'b1)   dataa_exp_not_zero_ff_p1 <= dataa_exp_not_zero[10];
	// synopsys translate_off
	initial
		dataa_man_not_zero_ff_p1 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) dataa_man_not_zero_ff_p1 <= 1'b0;
		else if  (clk_en == 1'b1)   dataa_man_not_zero_ff_p1 <= dataa_man_not_zero[25];
	// synopsys translate_off
	initial
		dataa_man_not_zero_ff_p2 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) dataa_man_not_zero_ff_p2 <= 1'b0;
		else if  (clk_en == 1'b1)   dataa_man_not_zero_ff_p2 <= dataa_man_not_zero[51];
	// synopsys translate_off
	initial
		datab_exp_all_one_ff_p1 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) datab_exp_all_one_ff_p1 <= 1'b0;
		else if  (clk_en == 1'b1)   datab_exp_all_one_ff_p1 <= datab_exp_all_one[10];
	// synopsys translate_off
	initial
		datab_exp_not_zero_ff_p1 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) datab_exp_not_zero_ff_p1 <= 1'b0;
		else if  (clk_en == 1'b1)   datab_exp_not_zero_ff_p1 <= datab_exp_not_zero[10];
	// synopsys translate_off
	initial
		datab_man_not_zero_ff_p1 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) datab_man_not_zero_ff_p1 <= 1'b0;
		else if  (clk_en == 1'b1)   datab_man_not_zero_ff_p1 <= datab_man_not_zero[25];
	// synopsys translate_off
	initial
		datab_man_not_zero_ff_p2 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) datab_man_not_zero_ff_p2 <= 1'b0;
		else if  (clk_en == 1'b1)   datab_man_not_zero_ff_p2 <= datab_man_not_zero[51];
	// synopsys translate_off
	initial
		delay_exp2_bias = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) delay_exp2_bias <= 13'b0;
		else if  (clk_en == 1'b1)   delay_exp2_bias <= delay_exp_bias;
	// synopsys translate_off
	initial
		delay_exp3_bias = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) delay_exp3_bias <= 13'b0;
		else if  (clk_en == 1'b1)   delay_exp3_bias <= delay_exp2_bias;
	// synopsys translate_off
	initial
		delay_exp_bias = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) delay_exp_bias <= 13'b0;
		else if  (clk_en == 1'b1)   delay_exp_bias <= wire_exp_bias_subtr_result;
	// synopsys translate_off
	initial
		delay_man_product_msb = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) delay_man_product_msb <= 1'b0;
		else if  (clk_en == 1'b1)   delay_man_product_msb <= delay_man_product_msb_p0;
	// synopsys translate_off
	initial
		delay_man_product_msb_p0 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) delay_man_product_msb_p0 <= 1'b0;
		else if  (clk_en == 1'b1)   delay_man_product_msb_p0 <= wire_man_product2_mult_result[105];
	// synopsys translate_off
	initial
		exp_add_p1 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) exp_add_p1 <= 12'b0;
		else if  (clk_en == 1'b1)   exp_add_p1 <= wire_exp_add_adder_result;
	// synopsys translate_off
	initial
		exp_result_ff = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) exp_result_ff <= 11'b0;
		else if  (clk_en == 1'b1)   exp_result_ff <= ((inf_num & {11{((exp_is_inf | input_is_infinity_ff2) | input_is_nan_ff2)}}) | ((wire_exp_adj_adder_result[10:0] & {11{(~ exp_is_zero)}}) & {11{input_not_zero_ff2}}));
	// synopsys translate_off
	initial
		input_is_infinity_dffe_0 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) input_is_infinity_dffe_0 <= 1'b0;
		else if  (clk_en == 1'b1)   input_is_infinity_dffe_0 <= ((dataa_exp_all_one_ff_p1 & (~ (dataa_man_not_zero_ff_p1 | dataa_man_not_zero_ff_p2))) | (datab_exp_all_one_ff_p1 & (~ (datab_man_not_zero_ff_p1 | datab_man_not_zero_ff_p2))));
	// synopsys translate_off
	initial
		input_is_infinity_dffe_1 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) input_is_infinity_dffe_1 <= 1'b0;
		else if  (clk_en == 1'b1)   input_is_infinity_dffe_1 <= input_is_infinity_dffe_0;
	// synopsys translate_off
	initial
		input_is_infinity_ff1 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) input_is_infinity_ff1 <= 1'b0;
		else if  (clk_en == 1'b1)   input_is_infinity_ff1 <= input_is_infinity_dffe_1;
	// synopsys translate_off
	initial
		input_is_infinity_ff2 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) input_is_infinity_ff2 <= 1'b0;
		else if  (clk_en == 1'b1)   input_is_infinity_ff2 <= input_is_infinity_ff1;
	// synopsys translate_off
	initial
		input_is_nan_dffe_0 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) input_is_nan_dffe_0 <= 1'b0;
		else if  (clk_en == 1'b1)   input_is_nan_dffe_0 <= ((dataa_exp_all_one_ff_p1 & (dataa_man_not_zero_ff_p1 | dataa_man_not_zero_ff_p2)) | (datab_exp_all_one_ff_p1 & (datab_man_not_zero_ff_p1 | datab_man_not_zero_ff_p2)));
	// synopsys translate_off
	initial
		input_is_nan_dffe_1 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) input_is_nan_dffe_1 <= 1'b0;
		else if  (clk_en == 1'b1)   input_is_nan_dffe_1 <= input_is_nan_dffe_0;
	// synopsys translate_off
	initial
		input_is_nan_ff1 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) input_is_nan_ff1 <= 1'b0;
		else if  (clk_en == 1'b1)   input_is_nan_ff1 <= input_is_nan_dffe_1;
	// synopsys translate_off
	initial
		input_is_nan_ff2 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) input_is_nan_ff2 <= 1'b0;
		else if  (clk_en == 1'b1)   input_is_nan_ff2 <= input_is_nan_ff1;
	// synopsys translate_off
	initial
		input_not_zero_dffe_0 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) input_not_zero_dffe_0 <= 1'b0;
		else if  (clk_en == 1'b1)   input_not_zero_dffe_0 <= (dataa_exp_not_zero_ff_p1 & datab_exp_not_zero_ff_p1);
	// synopsys translate_off
	initial
		input_not_zero_dffe_1 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) input_not_zero_dffe_1 <= 1'b0;
		else if  (clk_en == 1'b1)   input_not_zero_dffe_1 <= input_not_zero_dffe_0;
	// synopsys translate_off
	initial
		input_not_zero_ff1 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) input_not_zero_ff1 <= 1'b0;
		else if  (clk_en == 1'b1)   input_not_zero_ff1 <= input_not_zero_dffe_1;
	// synopsys translate_off
	initial
		input_not_zero_ff2 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) input_not_zero_ff2 <= 1'b0;
		else if  (clk_en == 1'b1)   input_not_zero_ff2 <= input_not_zero_ff1;
	// synopsys translate_off
	initial
		lsb_dffe = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) lsb_dffe <= 1'b0;
		else if  (clk_en == 1'b1)   lsb_dffe <= lsb_bit;
	// synopsys translate_off
	initial
		man_result_ff = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) man_result_ff <= 52'b0;
		else if  (clk_en == 1'b1)   man_result_ff <= {((((((delay_round_wire[51] & input_not_zero_ff2) & (~ input_is_infinity_ff2)) & (~ exp_is_inf)) & (~ exp_is_zero)) | (input_is_infinity_ff2 & (~ input_not_zero_ff2))) | input_is_nan_ff2), (((((delay_round_wire[50:0] & {51{input_not_zero_ff2}}) & {51{(~ input_is_infinity_ff2)}}) & {51{(~ exp_is_inf)}}) & {51{(~ exp_is_zero)}}) & {51{(~ input_is_nan_ff2)}})};
	// synopsys translate_off
	initial
		man_round_p = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) man_round_p <= 53'b0;
		else if  (clk_en == 1'b1)   man_round_p <= man_shift_full[53:1];
	// synopsys translate_off
	initial
		man_round_p2 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) man_round_p2 <= 54'b0;
		else if  (clk_en == 1'b1)   man_round_p2 <= wire_man_round_adder_result;
	// synopsys translate_off
	initial
		round_dffe = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) round_dffe <= 1'b0;
		else if  (clk_en == 1'b1)   round_dffe <= round_bit;
	// synopsys translate_off
	initial
		sign_node_ff0 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) sign_node_ff0 <= 1'b0;
		else if  (clk_en == 1'b1)   sign_node_ff0 <= (dataa[63] ^ datab[63]);
	// synopsys translate_off
	initial
		sign_node_ff1 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) sign_node_ff1 <= 1'b0;
		else if  (clk_en == 1'b1)   sign_node_ff1 <= sign_node_ff0[0:0];
	// synopsys translate_off
	initial
		sign_node_ff2 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) sign_node_ff2 <= 1'b0;
		else if  (clk_en == 1'b1)   sign_node_ff2 <= sign_node_ff1[0:0];
	// synopsys translate_off
	initial
		sign_node_ff3 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) sign_node_ff3 <= 1'b0;
		else if  (clk_en == 1'b1)   sign_node_ff3 <= sign_node_ff2[0:0];
	// synopsys translate_off
	initial
		sign_node_ff4 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) sign_node_ff4 <= 1'b0;
		else if  (clk_en == 1'b1)   sign_node_ff4 <= sign_node_ff3[0:0];
	// synopsys translate_off
	initial
		sign_node_ff5 = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) sign_node_ff5 <= 1'b0;
		else if  (clk_en == 1'b1)   sign_node_ff5 <= sign_node_ff4[0:0];
	// synopsys translate_off
	initial
		sticky_dffe = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) sticky_dffe <= 1'b0;
		else if  (clk_en == 1'b1)   sticky_dffe <= sticky_bit[51];
	lpm_add_sub   exp_add_adder
	( 
	.aclr(aclr),
	.cin(1'b0),
	.clken(clk_en),
	.clock(clock),
	.cout(),
	.dataa({1'b0, dataa[62:52]}),
	.datab({1'b0, datab[62:52]}),
	.overflow(),
	.result(wire_exp_add_adder_result)
	`ifndef FORMAL_VERIFICATION
	// synopsys translate_off
	`endif
	,
	.add_sub(1'b1)
	`ifndef FORMAL_VERIFICATION
	// synopsys translate_on
	`endif
	);
	defparam
		exp_add_adder.lpm_pipeline = 1,
		exp_add_adder.lpm_width = 12,
		exp_add_adder.lpm_type = "lpm_add_sub";
	lpm_add_sub   exp_adj_adder
	( 
	.cin(1'b0),
	.cout(),
	.dataa(delay_exp3_bias),
	.datab(expmod),
	.overflow(),
	.result(wire_exp_adj_adder_result)
	`ifndef FORMAL_VERIFICATION
	// synopsys translate_off
	`endif
	,
	.aclr(1'b0),
	.add_sub(1'b1),
	.clken(1'b1),
	.clock(1'b0)
	`ifndef FORMAL_VERIFICATION
	// synopsys translate_on
	`endif
	);
	defparam
		exp_adj_adder.lpm_pipeline = 0,
		exp_adj_adder.lpm_width = 13,
		exp_adj_adder.lpm_type = "lpm_add_sub";
	lpm_add_sub   exp_bias_subtr
	( 
	.cout(),
	.dataa({1'b0, exp_add_p1[11:0]}),
	.datab({bias[12:0]}),
	.overflow(),
	.result(wire_exp_bias_subtr_result)
	`ifndef FORMAL_VERIFICATION
	// synopsys translate_off
	`endif
	,
	.aclr(1'b0),
	.add_sub(1'b1),
	.cin(),
	.clken(1'b1),
	.clock(1'b0)
	`ifndef FORMAL_VERIFICATION
	// synopsys translate_on
	`endif
	);
	defparam
		exp_bias_subtr.lpm_direction = "SUB",
		exp_bias_subtr.lpm_pipeline = 0,
		exp_bias_subtr.lpm_representation = "UNSIGNED",
		exp_bias_subtr.lpm_width = 13,
		exp_bias_subtr.lpm_type = "lpm_add_sub";
	lpm_add_sub   man_round_adder
	( 
	.cout(),
	.dataa({1'b0, man_round_p}),
	.datab({{53{1'b0}}, round_carry}),
	.overflow(),
	.result(wire_man_round_adder_result)
	`ifndef FORMAL_VERIFICATION
	// synopsys translate_off
	`endif
	,
	.aclr(1'b0),
	.add_sub(1'b1),
	.cin(),
	.clken(1'b1),
	.clock(1'b0)
	`ifndef FORMAL_VERIFICATION
	// synopsys translate_on
	`endif
	);
	defparam
		man_round_adder.lpm_pipeline = 0,
		man_round_adder.lpm_width = 54,
		man_round_adder.lpm_type = "lpm_add_sub";
	lpm_mult   man_product2_mult
	( 
	.aclr(aclr),
	.clken(clk_en),
	.clock(clock),
	.dataa({1'b1, dataa[51:0]}),
	.datab({1'b1, datab[51:0]}),
	.result(wire_man_product2_mult_result)
	`ifndef FORMAL_VERIFICATION
	// synopsys translate_off
	`endif
	,
	.sum({1{1'b0}})
	`ifndef FORMAL_VERIFICATION
	// synopsys translate_on
	`endif
	);
	defparam
		man_product2_mult.lpm_pipeline = 3,
		man_product2_mult.lpm_representation = "UNSIGNED",
		man_product2_mult.lpm_widtha = 53,
		man_product2_mult.lpm_widthb = 53,
		man_product2_mult.lpm_widthp = 106,
		man_product2_mult.lpm_widths = 1,
		man_product2_mult.lpm_type = "lpm_mult",
		man_product2_mult.lpm_hint = "DEDICATED_MULTIPLIER_CIRCUITRY=YES";
	assign
		aclr = 1'b0,
		bias = {{3{1'b0}}, {10{1'b1}}},
		dataa_exp_all_one = {(dataa[62] & dataa_exp_all_one[9]), (dataa[61] & dataa_exp_all_one[8]), (dataa[60] & dataa_exp_all_one[7]), (dataa[59] & dataa_exp_all_one[6]), (dataa[58] & dataa_exp_all_one[5]), (dataa[57] & dataa_exp_all_one[4]), (dataa[56] & dataa_exp_all_one[3]), (dataa[55] & dataa_exp_all_one[2]), (dataa[54] & dataa_exp_all_one[1]), (dataa[53] & dataa_exp_all_one[0]), dataa[52]},
		dataa_exp_not_zero = {(dataa[62] | dataa_exp_not_zero[9]), (dataa[61] | dataa_exp_not_zero[8]), (dataa[60] | dataa_exp_not_zero[7]), (dataa[59] | dataa_exp_not_zero[6]), (dataa[58] | dataa_exp_not_zero[5]), (dataa[57] | dataa_exp_not_zero[4]), (dataa[56] | dataa_exp_not_zero[3]), (dataa[55] | dataa_exp_not_zero[2]), (dataa[54] | dataa_exp_not_zero[1]), (dataa[53] | dataa_exp_not_zero[0]), dataa[52]},
		dataa_man_not_zero = {(dataa[51] | dataa_man_not_zero[50]), (dataa[50] | dataa_man_not_zero[49]), (dataa[49] | dataa_man_not_zero[48]), (dataa[48] | dataa_man_not_zero[47]), (dataa[47] | dataa_man_not_zero[46]), (dataa[46] | dataa_man_not_zero[45]), (dataa[45] | dataa_man_not_zero[44]), (dataa[44] | dataa_man_not_zero[43]), (dataa[43] | dataa_man_not_zero[42]), (dataa[42] | dataa_man_not_zero[41]), (dataa[41] | dataa_man_not_zero[40]), (dataa[40] | dataa_man_not_zero[39]), (dataa[39] | dataa_man_not_zero[38]), (dataa[38] | dataa_man_not_zero[37]), (dataa[37] | dataa_man_not_zero[36]), (dataa[36] | dataa_man_not_zero[35]), (dataa[35] | dataa_man_not_zero[34]), (dataa[34] | dataa_man_not_zero[33]), (dataa[33] | dataa_man_not_zero[32]), (dataa[32] | dataa_man_not_zero[31]), (dataa[31] | dataa_man_not_zero[30]), (dataa[30] | dataa_man_not_zero[29]), (dataa[29] | dataa_man_not_zero[28]), (dataa[28] | dataa_man_not_zero[27]), (dataa[27] | dataa_man_not_zero[26]), dataa[26], (dataa[25] | dataa_man_not_zero[24]), (dataa[24] | dataa_man_not_zero[23]), (dataa[23] | dataa_man_not_zero[22]), (dataa[22] | dataa_man_not_zero[21]), (dataa[21] | dataa_man_not_zero[20]), (dataa[20] | dataa_man_not_zero[19]), (dataa[19] | dataa_man_not_zero[18]), (dataa[18] | dataa_man_not_zero[17]), (dataa[17] | dataa_man_not_zero[16]), (dataa[16] | dataa_man_not_zero[15]), (dataa[15] | dataa_man_not_zero[14]), (dataa[14] | dataa_man_not_zero[13]), (dataa[13] | dataa_man_not_zero[12]), (dataa[12] | dataa_man_not_zero[11]), (dataa[11] | dataa_man_not_zero[10]), (dataa[10] | dataa_man_not_zero[9]), (dataa[9] | dataa_man_not_zero[8]), (dataa[8] | dataa_man_not_zero[7]), (dataa[7] | dataa_man_not_zero[6]), (dataa[6] | dataa_man_not_zero[5]), (dataa[5] | dataa_man_not_zero[4]), (dataa[4] | dataa_man_not_zero[3]), (dataa[3] | dataa_man_not_zero[2]), (dataa[2] | dataa_man_not_zero[1]), (dataa[1] | dataa_man_not_zero[0]), dataa[0]},
		datab_exp_all_one = {(datab[62] & datab_exp_all_one[9]), (datab[61] & datab_exp_all_one[8]), (datab[60] & datab_exp_all_one[7]), (datab[59] & datab_exp_all_one[6]), (datab[58] & datab_exp_all_one[5]), (datab[57] & datab_exp_all_one[4]), (datab[56] & datab_exp_all_one[3]), (datab[55] & datab_exp_all_one[2]), (datab[54] & datab_exp_all_one[1]), (datab[53] & datab_exp_all_one[0]), datab[52]},
		datab_exp_not_zero = {(datab[62] | datab_exp_not_zero[9]), (datab[61] | datab_exp_not_zero[8]), (datab[60] | datab_exp_not_zero[7]), (datab[59] | datab_exp_not_zero[6]), (datab[58] | datab_exp_not_zero[5]), (datab[57] | datab_exp_not_zero[4]), (datab[56] | datab_exp_not_zero[3]), (datab[55] | datab_exp_not_zero[2]), (datab[54] | datab_exp_not_zero[1]), (datab[53] | datab_exp_not_zero[0]), datab[52]},
		datab_man_not_zero = {(datab[51] | datab_man_not_zero[50]), (datab[50] | datab_man_not_zero[49]), (datab[49] | datab_man_not_zero[48]), (datab[48] | datab_man_not_zero[47]), (datab[47] | datab_man_not_zero[46]), (datab[46] | datab_man_not_zero[45]), (datab[45] | datab_man_not_zero[44]), (datab[44] | datab_man_not_zero[43]), (datab[43] | datab_man_not_zero[42]), (datab[42] | datab_man_not_zero[41]), (datab[41] | datab_man_not_zero[40]), (datab[40] | datab_man_not_zero[39]), (datab[39] | datab_man_not_zero[38]), (datab[38] | datab_man_not_zero[37]), (datab[37] | datab_man_not_zero[36]), (datab[36] | datab_man_not_zero[35]), (datab[35] | datab_man_not_zero[34]), (datab[34] | datab_man_not_zero[33]), (datab[33] | datab_man_not_zero[32]), (datab[32] | datab_man_not_zero[31]), (datab[31] | datab_man_not_zero[30]), (datab[30] | datab_man_not_zero[29]), (datab[29] | datab_man_not_zero[28]), (datab[28] | datab_man_not_zero[27]), (datab[27] | datab_man_not_zero[26]), datab[26], (datab[25] | datab_man_not_zero[24]), (datab[24] | datab_man_not_zero[23]), (datab[23] | datab_man_not_zero[22]), (datab[22] | datab_man_not_zero[21]), (datab[21] | datab_man_not_zero[20]), (datab[20] | datab_man_not_zero[19]), (datab[19] | datab_man_not_zero[18]), (datab[18] | datab_man_not_zero[17]), (datab[17] | datab_man_not_zero[16]), (datab[16] | datab_man_not_zero[15]), (datab[15] | datab_man_not_zero[14]), (datab[14] | datab_man_not_zero[13]), (datab[13] | datab_man_not_zero[12]), (datab[12] | datab_man_not_zero[11]), (datab[11] | datab_man_not_zero[10]), (datab[10] | datab_man_not_zero[9]), (datab[9] | datab_man_not_zero[8]), (datab[8] | datab_man_not_zero[7]), (datab[7] | datab_man_not_zero[6]), (datab[6] | datab_man_not_zero[5]), (datab[5] | datab_man_not_zero[4]), (datab[4] | datab_man_not_zero[3]), (datab[3] | datab_man_not_zero[2]), (datab[2] | datab_man_not_zero[1]), (datab[1] | datab_man_not_zero[0]), datab[0]},
		delay_round_wire = ((man_round_p2[52:0] & {53{(~ man_round_p2[53])}}) | (man_round_p2[53:1] & {53{man_round_p2[53]}})),
		exp_is_inf = (((~ wire_exp_adj_adder_result[12]) & wire_exp_adj_adder_result[11]) | ((~ wire_exp_adj_adder_result[11]) & result_exp_all_one[10])),
		exp_is_zero = (wire_exp_adj_adder_result[12] | (~ result_exp_not_zero[11])),
		expmod = {{11{1'b0}}, (delay_man_product_msb & man_round_p2[53]), (delay_man_product_msb ^ man_round_p2[53])},
		inf_num = {11{1'b1}},
		lsb_bit = man_shift_full[1],
		man_shift_full = ((wire_man_product2_mult_result[104:51] & {54{(~ wire_man_product2_mult_result[105])}}) | (wire_man_product2_mult_result[105:52] & {54{wire_man_product2_mult_result[105]}})),
		result = {sign_node_ff5[0:0], exp_result_ff[10:0], man_result_ff[51:0]},
		result_exp_all_one = {(result_exp_all_one[9] & wire_exp_adj_adder_result[10]), (result_exp_all_one[8] & wire_exp_adj_adder_result[9]), (result_exp_all_one[7] & wire_exp_adj_adder_result[8]), (result_exp_all_one[6] & wire_exp_adj_adder_result[7]), (result_exp_all_one[5] & wire_exp_adj_adder_result[6]), (result_exp_all_one[4] & wire_exp_adj_adder_result[5]), (result_exp_all_one[3] & wire_exp_adj_adder_result[4]), (result_exp_all_one[2] & wire_exp_adj_adder_result[3]), (result_exp_all_one[1] & wire_exp_adj_adder_result[2]), (result_exp_all_one[0] & wire_exp_adj_adder_result[1]), wire_exp_adj_adder_result[0]},
		result_exp_not_zero = {(result_exp_not_zero[10] | wire_exp_adj_adder_result[11]), (result_exp_not_zero[9] | wire_exp_adj_adder_result[10]), (result_exp_not_zero[8] | wire_exp_adj_adder_result[9]), (result_exp_not_zero[7] | wire_exp_adj_adder_result[8]), (result_exp_not_zero[6] | wire_exp_adj_adder_result[7]), (result_exp_not_zero[5] | wire_exp_adj_adder_result[6]), (result_exp_not_zero[4] | wire_exp_adj_adder_result[5]), (result_exp_not_zero[3] | wire_exp_adj_adder_result[4]), (result_exp_not_zero[2] | wire_exp_adj_adder_result[3]), (result_exp_not_zero[1] | wire_exp_adj_adder_result[2]), (result_exp_not_zero[0] | wire_exp_adj_adder_result[1]), wire_exp_adj_adder_result[0]},
		round_bit = man_shift_full[0],
		round_carry = (round_dffe & (lsb_dffe | sticky_dffe)),
		sticky_bit = {(sticky_bit[50] | (wire_man_product2_mult_result[105] & wire_man_product2_mult_result[51])), (sticky_bit[49] | wire_man_product2_mult_result[50]), (sticky_bit[48] | wire_man_product2_mult_result[49]), (sticky_bit[47] | wire_man_product2_mult_result[48]), (sticky_bit[46] | wire_man_product2_mult_result[47]), (sticky_bit[45] | wire_man_product2_mult_result[46]), (sticky_bit[44] | wire_man_product2_mult_result[45]), (sticky_bit[43] | wire_man_product2_mult_result[44]), (sticky_bit[42] | wire_man_product2_mult_result[43]), (sticky_bit[41] | wire_man_product2_mult_result[42]), (sticky_bit[40] | wire_man_product2_mult_result[41]), (sticky_bit[39] | wire_man_product2_mult_result[40]), (sticky_bit[38] | wire_man_product2_mult_result[39]), (sticky_bit[37] | wire_man_product2_mult_result[38]), (sticky_bit[36] | wire_man_product2_mult_result[37]), (sticky_bit[35] | wire_man_product2_mult_result[36]), (sticky_bit[34] | wire_man_product2_mult_result[35]), (sticky_bit[33] | wire_man_product2_mult_result[34]), (sticky_bit[32] | wire_man_product2_mult_result[33]), (sticky_bit[31] | wire_man_product2_mult_result[32]), (sticky_bit[30] | wire_man_product2_mult_result[31]), (sticky_bit[29] | wire_man_product2_mult_result[30]), (sticky_bit[28] | wire_man_product2_mult_result[29]), (sticky_bit[27] | wire_man_product2_mult_result[28]), (sticky_bit[26] | wire_man_product2_mult_result[27]), (sticky_bit[25] | wire_man_product2_mult_result[26]), (sticky_bit[24] | wire_man_product2_mult_result[25]), (sticky_bit[23] | wire_man_product2_mult_result[24]), (sticky_bit[22] | wire_man_product2_mult_result[23]), (sticky_bit[21] | wire_man_product2_mult_result[22]), (sticky_bit[20] | wire_man_product2_mult_result[21]), (sticky_bit[19] | wire_man_product2_mult_result[20]), (sticky_bit[18] | wire_man_product2_mult_result[19]), (sticky_bit[17] | wire_man_product2_mult_result[18]), (sticky_bit[16] | wire_man_product2_mult_result[17]), (sticky_bit[15] | wire_man_product2_mult_result[16]), (sticky_bit[14] | wire_man_product2_mult_result[15]
), (sticky_bit[13] | wire_man_product2_mult_result[14]), (sticky_bit[12] | wire_man_product2_mult_result[13]), (sticky_bit[11] | wire_man_product2_mult_result[12]), (sticky_bit[10] | wire_man_product2_mult_result[11]), (sticky_bit[9] | wire_man_product2_mult_result[10]), (sticky_bit[8] | wire_man_product2_mult_result[9]), (sticky_bit[7] | wire_man_product2_mult_result[8]), (sticky_bit[6] | wire_man_product2_mult_result[7]), (sticky_bit[5] | wire_man_product2_mult_result[6]), (sticky_bit[4] | wire_man_product2_mult_result[5]), (sticky_bit[3] | wire_man_product2_mult_result[4]), (sticky_bit[2] | wire_man_product2_mult_result[3]), (sticky_bit[1] | wire_man_product2_mult_result[2]), (sticky_bit[0] | wire_man_product2_mult_result[1]), wire_man_product2_mult_result[0]};
endmodule //acl_fp_mul_ll_s5_double_altfp_mult_abo
//VALID FILE


// synopsys translate_off
`timescale 1 ps / 1 ps
// synopsys translate_on
module acl_fp_mul_ll_s5_double (
	enable,
	clock,
	dataa,
	datab,
	result);

	input	  enable;
	input	  clock;
	input	[63:0]  dataa;
	input	[63:0]  datab;
	output	[63:0]  result;

	wire [63:0] sub_wire0;
	wire [63:0] result = sub_wire0[63:0];

	acl_fp_mul_ll_s5_double_altfp_mult_abo	acl_fp_mul_ll_s5_double_altfp_mult_abo_component (
				.clk_en (enable),
				.clock (clock),
				.datab (datab),
				.dataa (dataa),
				.result (sub_wire0));

endmodule

// ============================================================
// CNX file retrieval info
// ============================================================
// Retrieval info: LIBRARY: altera_mf altera_mf.altera_mf_components.all
// Retrieval info: PRIVATE: FPM_FORMAT STRING "Double"
// Retrieval info: PRIVATE: INTENDED_DEVICE_FAMILY STRING "Stratix V"
// Retrieval info: CONSTANT: DEDICATED_MULTIPLIER_CIRCUITRY STRING "YES"
// Retrieval info: CONSTANT: DENORMAL_SUPPORT STRING "NO"
// Retrieval info: CONSTANT: EXCEPTION_HANDLING STRING "NO"
// Retrieval info: CONSTANT: INTENDED_DEVICE_FAMILY STRING "UNUSED"
// Retrieval info: CONSTANT: LPM_HINT STRING "UNUSED"
// Retrieval info: CONSTANT: LPM_TYPE STRING "altfp_mult"
// Retrieval info: CONSTANT: PIPELINE NUMERIC "6"
// Retrieval info: CONSTANT: REDUCED_FUNCTIONALITY STRING "NO"
// Retrieval info: CONSTANT: ROUNDING STRING "TO_NEAREST"
// Retrieval info: CONSTANT: WIDTH_EXP NUMERIC "11"
// Retrieval info: CONSTANT: WIDTH_MAN NUMERIC "52"
// Retrieval info: USED_PORT: clk_en 0 0 0 0 INPUT NODEFVAL "clk_en"
// Retrieval info: CONNECT: @clk_en 0 0 0 0 clk_en 0 0 0 0
// Retrieval info: USED_PORT: clock 0 0 0 0 INPUT NODEFVAL "clock"
// Retrieval info: CONNECT: @clock 0 0 0 0 clock 0 0 0 0
// Retrieval info: USED_PORT: dataa 0 0 64 0 INPUT NODEFVAL "dataa[63..0]"
// Retrieval info: CONNECT: @dataa 0 0 64 0 dataa 0 0 64 0
// Retrieval info: USED_PORT: datab 0 0 64 0 INPUT NODEFVAL "datab[63..0]"
// Retrieval info: CONNECT: @datab 0 0 64 0 datab 0 0 64 0
// Retrieval info: USED_PORT: result 0 0 64 0 OUTPUT NODEFVAL "result[63..0]"
// Retrieval info: CONNECT: result 0 0 64 0 @result 0 0 64 0
// Retrieval info: GEN_FILE: TYPE_NORMAL acl_fp_mul_ll_s5_double.v TRUE FALSE
// Retrieval info: GEN_FILE: TYPE_NORMAL acl_fp_mul_ll_s5_double.qip TRUE FALSE
// Retrieval info: GEN_FILE: TYPE_NORMAL acl_fp_mul_ll_s5_double.bsf TRUE TRUE
// Retrieval info: GEN_FILE: TYPE_NORMAL acl_fp_mul_ll_s5_double_inst.v TRUE TRUE
// Retrieval info: GEN_FILE: TYPE_NORMAL acl_fp_mul_ll_s5_double_bb.v TRUE TRUE
// Retrieval info: GEN_FILE: TYPE_NORMAL acl_fp_mul_ll_s5_double.inc TRUE TRUE
// Retrieval info: GEN_FILE: TYPE_NORMAL acl_fp_mul_ll_s5_double.cmp TRUE TRUE
// Retrieval info: LIB_FILE: lpm
