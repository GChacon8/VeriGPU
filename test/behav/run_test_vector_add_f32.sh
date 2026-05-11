#!/bin/bash
# run_test_vector_add_f32.sh — Compile and run float vector add test
#
# Tests FADD.S instruction running on 4 RISC-V cores with memory arbitration.
# This is the float equivalent of run_test_vector_add.sh.
#
# Run from repository root:
#   bash test/behav/run_test_vector_add_f32.sh

set -e
set -o pipefail

cd "$(dirname "$0")/../.."

echo "=== CP-2 (HW Roadmap): Float Vector Add Test ==="
echo ""

mkdir -p build

echo "Compiling with iverilog..."
iverilog -g2012 \
    src/assert.sv \
    src/const.sv \
    src/op_const.sv \
    src/int/chunked_add_task.sv \
    src/int/chunked_sub_task.sv \
    src/float/float_params.sv \
    src/float/float_add_pipeline.sv \
    src/float/float_mul_pipeline.sv \
    src/generated/mul_pipeline_cycle_24bit_2bpc.sv \
    src/generated/mul_pipeline_cycle_32bit_2bpc.sv \
    src/int/int_div_regfile.sv \
    src/int/mul_pipeline_32bit.sv \
    src/core.sv \
    src/mem_arbiter.sv \
    src/compute_unit.sv \
    src/mem_16mb.sv \
    src/global_mem_controller.sv \
    src/gpu_controller.sv \
    src/gpu_die.sv \
    test/behav/test_vector_add_f32.sv \
    -o build/test_vector_add_f32

echo "Compilation OK"
echo ""

echo "Running simulation..."
echo ""
./build/test_vector_add_f32
