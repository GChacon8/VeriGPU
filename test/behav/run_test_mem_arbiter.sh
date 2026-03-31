#!/bin/bash
# run_test_mem_arbiter.sh
# Compiles and runs the CP-2 memory arbiter test.
# Run from the repository root: bash test/behav/run_test_mem_arbiter.sh

set -e
set -o pipefail

echo "=== CP-2: Testing mem_arbiter (round-robin) ==="
echo ""

mkdir -p build

# The arbiter test only needs the memory subsystem — no cores, no float/int units.
iverilog -g2012 -Wall \
    src/const.sv \
    src/assert.sv \
    src/mem_large.sv \
    src/global_mem_controller.sv \
    src/mem_arbiter.sv \
    test/behav/test_mem_arbiter.sv \
    -o build/test_mem_arbiter

echo "Compilation OK"
echo ""

./build/test_mem_arbiter

echo ""
echo "=== Test complete ==="