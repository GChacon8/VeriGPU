#!/bin/bash
# run_test_gpu_die_multicore.sh
# CP-6 test: full gpu_die with batching.
set -e
set -o pipefail

echo "=== CP-6: Testing gpu_die with batching ==="
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
    src/gpu_controller.sv \
    src/gpu_die.sv \
    test/behav/test_gpu_die_multicore.sv \
    -o build/test_gpu_die_multicore

echo "Compilation OK"
echo ""

./build/test_gpu_die_multicore

echo ""
echo "=== Test complete ==="
