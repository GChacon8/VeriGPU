#!/bin/bash
# ============================================================
# run_test_math_lib.sh 
# ============================================================

set -e
set -o pipefail

echo "=== Testing software math library (fdiv, fsqrt, fexp, flog, ftanh) ==="
echo ""

mkdir -p build

# Step 1: Assemble the test program
echo "Step 1: Assembling test program..."
python3 verigpu/assembler.py \
    --in-asm examples/direct/test_math_lib.asm \
    --out-hex build/test_math_lib.hex \
    --offset 128 \
    --quiet
echo "  Assembly OK"
echo ""

# Step 2: Compile the testbench with iverilog
echo "Step 2: Compiling testbench..."
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
    test/behav/test_math_lib.sv \
    -o build/test_math_lib
echo "  Compilation OK"
echo ""

# Step 3: Run the simulation
echo "Step 3: Running simulation..."
echo ""
./build/test_math_lib

echo ""
echo "=== Test complete ==="
