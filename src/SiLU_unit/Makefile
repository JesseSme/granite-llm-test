SIM ?= verilator
TOPLEVEL_LANG = verilog
VERILOG_SOURCES = ../sigmoid_unit/sigmoid_unit.sv silu_unit.sv
TOPLEVEL = silu_unit
MODULE = tb_silu_unit
COMPILE_ARGS = -Wall
include $(shell cocotb-config --makefiles)/Makefile.sim
