#!/usr/bin/env python3
"""
test_hw_matmul.py — CP-10 (HW Roadmap): Matrix multiplication on hardware.

Tests torch.mm running on RISC-V cores. Each core computes one row
of the output matrix, with batching for matrices larger than 4 rows.

Also tests addmm (used by nn.Linear) which calls mm internally.

Run from repository root:
    VERIGPU_USE_HW=1 python3 pytorch_backend/test_hw_matmul.py
"""

import torch
import subprocess
import sys
import os
import types

# ── Backend setup ──────────────────────────────────────────────────────

print("=== CP-10 (HW Roadmap): Matmul on Hardware ===")
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
    "vmm_f32":              "kernels/matmul_f32.asm",
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

def check_mm(name, a_data, b_data, atol=1e-3):
    try:
        a_cpu = torch.tensor(a_data, dtype=torch.float32)
        b_cpu = torch.tensor(b_data, dtype=torch.float32)
        expected = torch.mm(a_cpu, b_cpu)

        a_gpu = a_cpu.to("verigpu")
        b_gpu = b_cpu.to("verigpu")
        result = torch.mm(a_gpu, b_gpu).cpu()

        ok = torch.allclose(result, expected, atol=atol)
        if not ok:
            print(f"  got:      {result}")
            print(f"  expected: {expected}")
            diff = (result - expected).abs().max().item()
            print(f"  max diff: {diff}")
        check(name, ok)
    except Exception as e:
        import traceback; traceback.print_exc()
        check(f"{name} (exception)", False)

# ── MM ────────────────────────────────────────────────────────────────

print("── MM (matrix multiplication) ──")

# Test 1: 2x2 × 2x2
check_mm("2x2 @ 2x2",
    [[1.0, 2.0], [3.0, 4.0]],
    [[5.0, 6.0], [7.0, 8.0]])
# Expected: [[19, 22], [43, 50]]

# Test 2: 4x2 × 2x3 (1 batch, 4 rows)
check_mm("4x2 @ 2x3",
    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 3.0]],
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# Row 0: [1,2,3], Row 1: [4,5,6], Row 2: [5,7,9], Row 3: [14,19,24]

# Test 3: 8x2 × 2x2 (2 batches)
check_mm("8x2 @ 2x2 (2 batches)",
    [[float(i), float(i+1)] for i in range(8)],
    [[1.0, 0.0], [0.0, 1.0]])  # identity-ish

# Test 4: 1x4 × 4x1 (dot product)
check_mm("1x4 @ 4x1 (dot product)",
    [[1.0, 2.0, 3.0, 4.0]],
    [[1.0], [2.0], [3.0], [4.0]])
# Expected: [[30]]

# Test 5: Random matrices vs CPU
try:
    torch.manual_seed(42)
    a_cpu = torch.randn(4, 3)
    b_cpu = torch.randn(3, 5)
    expected = torch.mm(a_cpu, b_cpu)
    result = torch.mm(a_cpu.to("verigpu"), b_cpu.to("verigpu")).cpu()
    diff = (result - expected).abs().max().item()
    check(f"random 4x3 @ 3x5 vs CPU (max diff={diff:.6f})", diff < 0.01)
except Exception as e:
    check(f"random mm (exception: {e})", False)

# ── ADDMM (used by nn.Linear) ────────────────────────────────────────

print("── ADDMM (nn.Linear path) ──")

try:
    # addmm(bias, input, weight.T) = bias + input @ weight.T
    bias = gpu([1.0, 2.0])
    inp  = gpu([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # 3x2
    weight = gpu([[1.0, 0.0], [0.0, 1.0]])              # 2x2 (identity)
    result = torch.addmm(bias, inp, weight.t()).cpu()
    expected = torch.tensor([[2.0, 2.0], [1.0, 3.0], [2.0, 3.0]])
    ok = torch.allclose(result, expected, atol=1e-3)
    if not ok:
        print(f"  got: {result}")
        print(f"  exp: {expected}")
    check("addmm (bias + input @ weight)", ok)
except Exception as e:
    import traceback; traceback.print_exc()
    check(f"addmm (exception)", False)

# ── nn.Linear ─────────────────────────────────────────────────────────

print("── nn.Linear (end-to-end) ──")

try:
    import torch.nn as nn
    torch.manual_seed(42)

    layer = nn.Linear(3, 2)
    layer = layer.to("verigpu")

    x = torch.randn(4, 3).to("verigpu")
    result = layer(x).cpu()

    # Compare with CPU
    torch.manual_seed(42)
    layer_cpu = nn.Linear(3, 2)
    x_cpu = x.cpu()
    expected = layer_cpu(x_cpu)

    diff = (result - expected).abs().max().item()
    check(f"nn.Linear(3,2) forward (max diff={diff:.6f})", diff < 0.01)
except Exception as e:
    import traceback; traceback.print_exc()
    check(f"nn.Linear (exception)", False)

# ── REGRESSION ────────────────────────────────────────────────────────

print("── REGRESSION ──")

try:
    a = gpu([1.0, 2.0, 3.0, 4.0])
    b = gpu([10.0, 20.0, 30.0, 40.0])
    check("add", torch.allclose((a+b).cpu(), torch.tensor([11., 22., 33., 44.])))
    check("sub", torch.allclose((a-b).cpu(), torch.tensor([-9., -18., -27., -36.])))
    check("mul", torch.allclose((a*b).cpu(), torch.tensor([10., 40., 90., 160.])))
    check("sum", abs(a.sum().item() - 10.0) < 1e-4)
    check("relu", torch.allclose(torch.relu(gpu([-1., 2., -3., 4.])).cpu(),
                                  torch.tensor([0., 2., 0., 4.])))
except Exception as e:
    check(f"regression (exception: {e})", False)

# ── Summary ───────────────────────────────────────────────────────────

print()
if failed == 0:
    print(f"========================================")
    print(f"  ALL {passed} TESTS PASSED")
    print(f"  matmul on RISC-V hardware!")
    print(f"========================================")
else:
    print(f"  {passed} passed, {failed} FAILED")
    sys.exit(1)
