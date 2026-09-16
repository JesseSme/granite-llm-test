# Default configuration: Attention Q/O projection (768 -> 768).
# The cocotb runner (run_test.py) is the working path in this environment and
# supports parameterized and in-loop builds.

SIM ?= verilator
TOPLEVEL_LANG = verilog

FP_DIR = /home/jese/tinyllm/systemverilog_fp_unit/rtl

VERILOG_SOURCES = \
  $(FP_DIR)/fp_pkg.sv \
  $(FP_DIR)/fp_add.sv \
  $(FP_DIR)/fp_mul.sv \
  $(FP_DIR)/fp_div.sv \
  $(FP_DIR)/fp_sqrt.sv \
  $(FP_DIR)/fp_fma.sv \
  $(FP_DIR)/fp_minmax.sv \
  $(FP_DIR)/fp_cmp.sv \
  $(FP_DIR)/fp_roundint.sv \
  $(FP_DIR)/fp_totalorder.sv \
  $(FP_DIR)/fp32_to_bf16_round.sv \
  $(FP_DIR)/fp_unit.sv \
  matrix_unit.sv

TOPLEVEL = matrix_unit
MODULE = tb_matrix_unit
COMPILE_ARGS = -Wall --top-module matrix_unit
include $(shell cocotb-config --makefiles)/Makefile.sim
