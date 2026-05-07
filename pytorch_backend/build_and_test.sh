#!/bin/bash
# build_and_test.sh — Build and test CP-7 (autograd)
# Run from repository root: bash pytorch_backend/build_and_test.sh
set -e
set -o pipefail

echo "=== CP-7: VeriGPU autograd — Build & Test ==="
echo ""

python3 -c "import torch" 2>/dev/null || {
    echo "ERROR: PyTorch not found. source .venv/bin/activate"; exit 1; }

echo "Step 1: Building C++ extension..."
cd pytorch_backend
pip install -e . --no-build-isolation 2>&1 | tail -3
echo "  Build OK"
echo ""
cd ..

echo "Step 2: Running CP-7 tests..."
python3 pytorch_backend/test_autograd.py

echo ""
echo "=== Done ==="
