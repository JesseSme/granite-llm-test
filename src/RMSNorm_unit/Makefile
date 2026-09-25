# Makefile for RMSNorm unit cocotb tests

SIM ?= verilator
TOPLEVEL_LANG ?= verilog

# Source files
RTL_DIR = systemverilog_fp_unit/rtl
RTL_SRCS = $(RTL_DIR)/fp_pkg.sv \
           $(RTL_DIR)/fp_add.sv \
           $(RTL_DIR)/fp_mul.sv \
           $(RTL_DIR)/fp_div.sv \
           $(RTL_DIR)/fp_sqrt.sv \
           $(RTL_DIR)/fp_fma.sv \
           $(RTL_DIR)/fp_minmax.sv \
           $(RTL_DIR)/fp_cmp.sv \
           $(RTL_DIR)/fp_totalorder.sv \
           $(RTL_DIR)/fp_roundint.sv \
           $(RTL_DIR)/fp_unit.sv \
           rmsnorm_unit.sv

# Test module
TOPLEVEL = rmsnorm_unit
MODULE = tb_rmsnorm_unit

# cocotb
COMPILE_ARGS = -Wall
SIM_ARGS = --timing

.PHONY: test clean lint

test:
	uv run -m cocotb.runner \
		--verilog-sim $(SIM) \
		--top-module $(TOPLEVEL) \
		--python-module $(MODULE) \
		--verilog-sources $(RTL_SRCS) \
		--sim-args "$(SIM_ARGS)" \
		--compile-args "$(COMPILE_ARGS)"

lint:
	verilator --lint-only -Wall $(RTL_SRCS)

clean:
	rm -rf sim_build results.xml __pycache__
