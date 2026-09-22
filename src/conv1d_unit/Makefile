SIM ?= verilator
TOPLEVEL_LANG = verilog
VERILOG_SOURCES = conv1d_unit.sv ../../systemverilog_fp_unit/rtl/fp_pkg.sv ../../systemverilog_fp_unit/rtl/fp_unit.sv ../../systemverilog_fp_unit/rtl/fp_add.sv ../../systemverilog_fp_unit/rtl/fp_mul.sv ../../systemverilog_fp_unit/rtl/fp_div.sv ../../systemverilog_fp_unit/rtl/fp_sqrt.sv ../../systemverilog_fp_unit/rtl/fp_fma.sv ../../systemverilog_fp_unit/rtl/fp_minmax.sv ../../systemverilog_fp_unit/rtl/fp_cmp.sv ../../systemverilog_fp_unit/rtl/fp_totalorder.sv ../../systemverilog_fp_unit/rtl/fp_roundint.sv ../../systemverilog_fp_unit/rtl/fp_add_e8m0.sv ../../systemverilog_fp_unit/rtl/fp_mul_e8m0.sv ../../systemverilog_fp_unit/rtl/fp_div_e8m0.sv ../../systemverilog_fp_unit/rtl/fp_sqrt_e8m0.sv ../../systemverilog_fp_unit/rtl/fp_fma_e8m0.sv ../../systemverilog_fp_unit/rtl/fp_roundint_e8m0.sv
TOPLEVEL = conv1d_unit
MODULE = tb_conv1d_unit
COMPILE_ARGS = -Wall -I../../systemverilog_fp_unit/rtl
include $(shell cocotb-config --makefiles)/Makefile.sim
