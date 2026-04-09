#!/bin/bash
# run_test_compute_unit.sh
# Compiles and runs the compute_unit diagnostic test.
set -e
set -o pipefail

echo "=== compute_unit: Testing (DIAGNOSTIC) ==="
echo ""

mkdir -p build

iverilog -g2012 -Wall \
    src/const.sv \
    src/op_const.sv \
    src/float/float_params.sv \
    src/assert.sv \
    src/int/chunked_add_task.sv \
    src/int/chunked_sub_task.sv \
    src/float/float_add_pipeline.sv \
    src/float/float_mul_pipeline.sv \
    src/int/int_div_regfile.sv \
    src/int/mul_pipeline_32bit.sv \
    src/generated/mul_pipeline_cycle_24bit_2bpc.sv \
    src/generated/mul_pipeline_cycle_32bit_2bpc.sv \
    src/mem_large.sv \
    src/core.sv \
    src/global_mem_controller.sv \
    src/mem_arbiter.sv \
    src/compute_unit.sv \
    test/behav/test_compute_unit.sv \
    -o build/test_compute_unit

echo "Compilation OK"
echo ""

./build/test_compute_unit

echo ""
echo "=== Test complete ==="
