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
  $(FP_DIR)/fp_unit.sv \
  embedding_lookup_unit.sv

TOPLEVEL = embedding_lookup_unit
MODULE = tb_embedding_lookup_unit
COMPILE_ARGS = -Wall --top-module embedding_lookup_unit
include $(shell cocotb-config --makefiles)/Makefile.sim
