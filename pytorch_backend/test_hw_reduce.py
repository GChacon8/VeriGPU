#!/usr/bin/env python3
"""
test_hw_reduce.py — CP-9 (HW Roadmap): Sum reduction on hardware.

Tests sum() and mean() running on the RISC-V cores via a sequential
sum kernel (thread 0 loops through all elements).

Run from repository root:
    VERIGPU_USE_HW=1 python3 pytorch_backend/test_hw_reduce.py
"""

import torch
import subprocess
import sys
import os
import types

# ── Backend setup ──────────────────────────────────────────────────────

print("=== CP-9 (HW Roadmap): Sum Reduction on Hardware ===")
print()

# All kernels
kernels = {
    "vadd_f32":             "kernels/vector_add_f32_param.asm",
    "vsub_f32":             "kernels/vector_sub_f32.asm",
    "vmul_f32":             "kernels/vector_mul_f32.asm",
    "vneg_f32":             "kernels/vector_neg_f32.asm",
    "vabs_f32":             "kernels/vector_abs_f32.asm",
    "vdiv_f32":             "kernels/vector_div_f32.asm",
    "vrelu_f32":            "kernels/vector_relu_f32.asm",
    "vthreshold_bwd_f32":   "kernels/vector_threshold_bwd_f32.asm",
    "vsum_f32":             "kernels/vector_sum_f32.asm",
}

print("Assembling kernels...")
os.makedirs("build", exist_ok=True)
for name, asm_path in kernels.items():
    if not os.path.exists(asm_path):
        print(f"  ERROR: {asm_path} not found")
        sys.exit(1)
    hex_path = f"build/{os.path.basename(asm_path).replace('.asm', '.hex')}"
    result = subprocess.run(
        ["python3", "verigpu/assembler.py", "--in-asm", asm_path, "--out-hex", hex_path],
        capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR assembling {name}:\n{result.stderr}")
        sys.exit(1)
print(f"  All {len(kernels)} kernels assembled OK")
print()

try:
    import _verigpu_C
except ImportError:
    print("ERROR: _verigpu_C not found.")
    sys.exit(1)

_mod = types.ModuleType("torch.verigpu")
_mod.__path__ = []
sys.modules["torch.verigpu"] = _mod
torch.utils.rename_privateuse1_backend("verigpu")

if not _verigpu_C.is_hw_mode():
    print("ERROR: Run with VERIGPU_USE_HW=1")
    sys.exit(1)

print("Loading kernels...")
for name, asm_path in kernels.items():
    hex_path = f"build/{os.path.basename(asm_path).replace('.asm', '.hex')}"
    with open(hex_path) as f:
        words = [int(line.strip(), 16) for line in f if line.strip()]
    _verigpu_C.load_kernel(name, words)
print()

# ── Tests ──────────────────────────────────────────────────────────────

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"PASS: {name}")
        passed += 1
    else:
        print(f"FAIL: {name}")
        failed += 1

def gpu(data):
    return torch.tensor(data, dtype=torch.float32).to("verigpu")

# ── SUM ───────────────────────────────────────────────────────────────

print("── SUM ──")

try:
    t = gpu([1.0, 2.0, 3.0, 4.0])
    result = t.sum().item()
    check(f"sum([1,2,3,4]) = {result} (expected 10.0)", abs(result - 10.0) < 1e-4)
except Exception as e:
    import traceback; traceback.print_exc()
    check(f"sum basic (exception)", False)

try:
    t = gpu([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    result = t.sum().item()
    check(f"sum([1..8]) = {result} (expected 36.0)", abs(result - 36.0) < 1e-4)
except Exception as e:
    check(f"sum 8-elem (exception: {e})", False)

try:
    t = gpu([-1.0, 2.0, -3.0, 4.0])
    result = t.sum().item()
    check(f"sum([-1,2,-3,4]) = {result} (expected 2.0)", abs(result - 2.0) < 1e-4)
except Exception as e:
    check(f"sum mixed (exception: {e})", False)

try:
    t = gpu([0.0, 0.0, 0.0, 0.0])
    result = t.sum().item()
    check(f"sum([0,0,0,0]) = {result} (expected 0.0)", abs(result) < 1e-6)
except Exception as e:
    check(f"sum zeros (exception: {e})", False)

try:
    torch.manual_seed(42)
    data = torch.randn(16)
    gpu_sum = data.to("verigpu").sum().item()
    cpu_sum = data.sum().item()
    diff = abs(gpu_sum - cpu_sum)
    check(f"sum(16 random) matches CPU (diff={diff:.6f})", diff < 1e-3)
except Exception as e:
    check(f"sum random (exception: {e})", False)

# ── MEAN ──────────────────────────────────────────────────────────────

print("── MEAN ──")

try:
    t = gpu([1.0, 2.0, 3.0, 4.0])
    result = t.mean().item()
    check(f"mean([1,2,3,4]) = {result} (expected 2.5)", abs(result - 2.5) < 1e-4)
except Exception as e:
    check(f"mean basic (exception: {e})", False)

try:
    t = gpu([10.0, 20.0, 30.0, 40.0])
    result = t.mean().item()
    check(f"mean([10,20,30,40]) = {result} (expected 25.0)", abs(result - 25.0) < 1e-4)
except Exception as e:
    check(f"mean large (exception: {e})", False)

try:
    torch.manual_seed(42)
    data = torch.randn(16)
    gpu_mean = data.to("verigpu").mean().item()
    cpu_mean = data.mean().item()
    diff = abs(gpu_mean - cpu_mean)
    check(f"mean(16 random) matches CPU (diff={diff:.6f})", diff < 1e-3)
except Exception as e:
    check(f"mean random (exception: {e})", False)

# ── MSE LOSS (sum + mean pipeline) ───────────────────────────────────

print("── MSE LOSS (mul + mean on HW) ──")

try:
    pred = gpu([0.5, 1.5, 2.5, 3.5])
    target = gpu([0.0, 1.0, 2.0, 3.0])
    diff = pred - target                    # sub kernel
    sq = diff * diff                         # mul kernel
    loss = sq.mean().item()                  # sum kernel + host divide
    expected = ((0.25 + 0.25 + 0.25 + 0.25) / 4)
    check(f"MSE loss = {loss:.4f} (expected {expected:.4f})", abs(loss - expected) < 1e-3)
except Exception as e:
    check(f"MSE loss (exception: {e})", False)

# ── REGRESSION ────────────────────────────────────────────────────────

print("── REGRESSION ──")

try:
    a = gpu([1.0, 2.0, 3.0, 4.0])
    b = gpu([10.0, 20.0, 30.0, 40.0])
    check("add still works", torch.allclose((a+b).cpu(), torch.tensor([11., 22., 33., 44.])))
    check("mul still works", torch.allclose((a*b).cpu(), torch.tensor([10., 40., 90., 160.])))
    check("relu still works", torch.allclose(torch.relu(gpu([-1.,2.,-3.,4.])).cpu(),
                                              torch.tensor([0., 2., 0., 4.])))
except Exception as e:
    check(f"regression (exception: {e})", False)

# ── Summary ───────────────────────────────────────────────────────────

print()
if failed == 0:
    print(f"========================================")
    print(f"  ALL {passed} TESTS PASSED")
    print(f"  sum/mean reduction on RISC-V!")
    print(f"========================================")
else:
    print(f"  {passed} passed, {failed} FAILED")
    sys.exit(1)
