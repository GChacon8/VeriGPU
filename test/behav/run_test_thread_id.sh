#!/bin/bash
# run_test_thread_id.sh
# Compiles and runs the CP-1 thread_id test.
# Run from the repository root: bash test/behav/run_test_thread_id.sh

set -e
set -o pipefail

echo "=== CP-1: Testing thread_id port on core ==="
echo ""

mkdir -p build

# Compile with iverilog
# Note: include paths may need adjustment depending on your generated files.
# If mul_pipeline_cycle_24bit_2bpc.sv doesn't exist, check src/generated/ for
# the actual filename and update accordingly.
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
    test/behav/core_and_mem.sv \
    test/behav/test_thread_id.sv \
    -o build/test_thread_id

echo "Compilation OK"
echo ""

# Run the simulation
./build/test_thread_id

echo ""
echo "=== Test complete ==="