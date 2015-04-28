// megafunction wizard: %ALTFP_ABS%
// GENERATION: STANDARD
// VERSION: WM1.0
// MODULE: ALTFP_ABS 

// ============================================================
// File Name: acl_fp_fabs_double.v
// Megafunction Name(s):
// 			ALTFP_ABS
//
// Simulation Library Files(s):
// 			
// ============================================================
// ************************************************************
// THIS IS A WIZARD-GENERATED FILE. DO NOT EDIT THIS FILE!
//
// 10.0 Build 262 08/18/2010 SP 1 SJ Full Version
// ************************************************************

// (C) 1992-2014 Altera Corporation. All rights reserved.                         
// Your use of Altera Corporation's design tools, logic functions and other       
// software and tools, and its AMPP partner logic functions, and any output       
// files any of the foregoing (including device programming or simulation         
// files), and any associated documentation or information are expressly subject  
// to the terms and conditions of the Altera Program License Subscription         
// Agreement, Altera MegaCore Function License Agreement, or other applicable     
// license agreement, including, without limitation, that your use is for the     
// sole purpose of programming logic devices manufactured by Altera and sold by   
// Altera or its authorized distributors.  Please refer to the applicable         
// agreement for further details.                                                 
    



//altfp_abs CBX_AUTO_BLACKBOX="ALL" DEVICE_FAMILY="Stratix IV" PIPELINE=1 WIDTH_EXP=11 WIDTH_MAN=52 clk_en clock data result
//VERSION_BEGIN 10.0SP1 cbx_altfp_abs 2010:08:18:21:07:09:SJ cbx_mgl 2010:08:18:21:11:11:SJ  VERSION_END
// synthesis VERILOG_INPUT_VERSION VERILOG_2001
// altera message_off 10463


//synthesis_resources = reg 63 
//synopsys translate_off
`timescale 1 ps / 1 ps
//synopsys translate_on
module  acl_fp_fabs_double_altfp_abs_v4b
	( 
	clk_en,
	clock,
	data,
	result) ;
	input   clk_en;
	input   clock;
	input   [63:0]  data;
	output   [63:0]  result;
`ifndef ALTERA_RESERVED_QIS
// synopsys translate_off
`endif
	tri1   clk_en;
	tri0   clock;
`ifndef ALTERA_RESERVED_QIS
// synopsys translate_on
`endif

	reg	[62:0]	result_pipe;
	wire aclr;
	wire  [63:0]  data_w;
	wire  gnd_w;

	// synopsys translate_off
	initial
		result_pipe = 0;
	// synopsys translate_on
	always @ ( posedge clock or  posedge aclr)
		if (aclr == 1'b1) result_pipe <= 63'b0;
		else if  (clk_en == 1'b1)   result_pipe <= data_w[62:0];
	assign
		aclr = 1'b0,
		data_w = {gnd_w, data[62:0]},
		gnd_w = 1'b0,
		result = {data_w[63], result_pipe[62:0]};
endmodule //acl_fp_fabs_double_altfp_abs_v4b
//VALID FILE


// synopsys translate_off
`timescale 1 ps / 1 ps
// synopsys translate_on
module acl_fp_fabs_double (
	enable,
	clock,
	dataa,
	result);

	input	  enable;
	input	  clock;
	input	[63:0]  dataa;
	output	[63:0]  result;

	wire [63:0] sub_wire0;
	wire [63:0] result = sub_wire0[63:0];

	acl_fp_fabs_double_altfp_abs_v4b	acl_fp_fabs_double_altfp_abs_v4b_component (
				.clk_en (enable),
				.clock (clock),
				.data (dataa),
				.result (sub_wire0));

endmodule

// ============================================================
// CNX file retrieval info
// ============================================================
// Retrieval info: LIBRARY: altera_mf altera_mf.altera_mf_components.all
// Retrieval info: PRIVATE: INTENDED_DEVICE_FAMILY STRING "Stratix IV"
// Retrieval info: CONSTANT: INTENDED_DEVICE_FAMILY STRING "UNUSED"
// Retrieval info: CONSTANT: LPM_HINT STRING "UNUSED"
// Retrieval info: CONSTANT: LPM_TYPE STRING "altfp_abs"
// Retrieval info: CONSTANT: PIPELINE NUMERIC "1"
// Retrieval info: CONSTANT: WIDTH_EXP NUMERIC "11"
// Retrieval info: CONSTANT: WIDTH_MAN NUMERIC "52"
// Retrieval info: USED_PORT: clk_en 0 0 0 0 INPUT NODEFVAL "clk_en"
// Retrieval info: CONNECT: @clk_en 0 0 0 0 clk_en 0 0 0 0
// Retrieval info: USED_PORT: clock 0 0 0 0 INPUT NODEFVAL "clock"
// Retrieval info: CONNECT: @clock 0 0 0 0 clock 0 0 0 0
// Retrieval info: USED_PORT: data 0 0 64 0 INPUT NODEFVAL "data[63..0]"
// Retrieval info: CONNECT: @data 0 0 64 0 data 0 0 64 0
// Retrieval info: USED_PORT: result 0 0 64 0 OUTPUT NODEFVAL "result[63..0]"
// Retrieval info: CONNECT: result 0 0 64 0 @result 0 0 64 0
// Retrieval info: GEN_FILE: TYPE_NORMAL acl_fp_fabs_double.v TRUE FALSE
// Retrieval info: GEN_FILE: TYPE_NORMAL acl_fp_fabs_double.qip TRUE FALSE
// Retrieval info: GEN_FILE: TYPE_NORMAL acl_fp_fabs_double.bsf TRUE TRUE
// Retrieval info: GEN_FILE: TYPE_NORMAL acl_fp_fabs_double_inst.v TRUE TRUE
// Retrieval info: GEN_FILE: TYPE_NORMAL acl_fp_fabs_double_bb.v TRUE TRUE
// Retrieval info: GEN_FILE: TYPE_NORMAL acl_fp_fabs_double.inc TRUE TRUE
// Retrieval info: GEN_FILE: TYPE_NORMAL acl_fp_fabs_double.cmp TRUE TRUE