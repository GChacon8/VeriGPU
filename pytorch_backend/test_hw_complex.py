#!/usr/bin/env python3
"""
test_hw_complex.py — CP-8 (HW Roadmap): Complex kernels on hardware.

Tests div (Newton-Raphson), relu (conditional), and threshold_backward
(relu gradient) running on RISC-V cores.

Run from repository root:
    VERIGPU_USE_HW=1 python3 pytorch_backend/test_hw_complex.py
"""

import torch
import subprocess
import sys
import os
import types

# ── Backend setup ──────────────────────────────────────────────────────

print("=== CP-8 (HW Roadmap): Complex Kernels on Hardware ===")
print()

# All kernels (including previous CPs for regression + chaining)
kernels = {
    "vadd_f32":             "kernels/vector_add_f32_param.asm",
    "vsub_f32":             "kernels/vector_sub_f32.asm",
    "vmul_f32":             "kernels/vector_mul_f32.asm",
    "vneg_f32":             "kernels/vector_neg_f32.asm",
    "vabs_f32":             "kernels/vector_abs_f32.asm",
    "vdiv_f32":             "kernels/vector_div_f32.asm",
    "vrelu_f32":            "kernels/vector_relu_f32.asm",
    "vthreshold_bwd_f32":   "kernels/vector_threshold_bwd_f32.asm",
}

print("Step 1: Assembling kernels...")
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
    print("ERROR: _verigpu_C not found. Build first.")
    sys.exit(1)

_mod = types.ModuleType("torch.verigpu")
_mod.__path__ = []
sys.modules["torch.verigpu"] = _mod
torch.utils.rename_privateuse1_backend("verigpu")

if not _verigpu_C.is_hw_mode():
    print("ERROR: Run with VERIGPU_USE_HW=1")
    sys.exit(1)

print("Step 2: Loading kernels to GPU memory...")
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

def check_close(name, got_gpu, expected_cpu, atol=1e-4):
    try:
        got = got_gpu.cpu()
        ok = torch.allclose(got, expected_cpu, atol=atol)
        if not ok:
            diff = (got - expected_cpu).abs().max().item()
            print(f"  got:      {got.tolist()}")
            print(f"  expected: {expected_cpu.tolist()}")
            print(f"  max diff: {diff}")
        check(name, ok)
    except Exception as e:
        import traceback; traceback.print_exc()
        check(f"{name} (exception)", False)

def gpu(data):
    return torch.tensor(data, dtype=torch.float32).to("verigpu")

print("Step 3: Running tests...")
print()

# ── DIV ───────────────────────────────────────────────────────────────

print("── DIV (Newton-Raphson on hardware) ──")

check_close("div: [10,20,30,40] / [2,4,5,8]",
    gpu([10.0, 20.0, 30.0, 40.0]) / gpu([2.0, 4.0, 5.0, 8.0]),
    torch.tensor([5.0, 5.0, 6.0, 5.0]),
    atol=1e-2)  # Newton-Raphson has ~1e-3 error

check_close("div: 8 elements (2 batches)",
    gpu([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]) /
    gpu([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
    torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
    atol=1e-2)

check_close("div: with negatives",
    gpu([-6.0, 9.0, -12.0, 15.0]) / gpu([3.0, -3.0, 4.0, -5.0]),
    torch.tensor([-2.0, -3.0, -3.0, -3.0]),
    atol=1e-2)

# Compare with CPU (higher tolerance for Newton-Raphson)
try:
    torch.manual_seed(42)
    a_cpu = torch.randn(16) * 10
    b_cpu = torch.randn(16).abs().clamp(min=0.5)  # avoid near-zero divisors
    expected = a_cpu / b_cpu
    result = (a_cpu.to("verigpu") / b_cpu.to("verigpu")).cpu()
    diff = (result - expected).abs().max().item()
    check(f"random 16-elem div vs CPU (max diff={diff:.6f})", diff < 0.05)
except Exception as e:
    check(f"random div (exception: {e})", False)

# ── RELU ──────────────────────────────────────────────────────────────

print("── RELU (conditional branch on hardware) ──")

check_close("relu: [1,-2,3,-4]",
    torch.relu(gpu([1.0, -2.0, 3.0, -4.0])),
    torch.tensor([1.0, 0.0, 3.0, 0.0]))

check_close("relu: all positive",
    torch.relu(gpu([1.0, 2.0, 3.0, 4.0])),
    torch.tensor([1.0, 2.0, 3.0, 4.0]))

check_close("relu: all negative",
    torch.relu(gpu([-1.0, -2.0, -3.0, -4.0])),
    torch.tensor([0.0, 0.0, 0.0, 0.0]))

check_close("relu: with zero",
    torch.relu(gpu([-1.0, 0.0, 1.0, 0.0])),
    torch.tensor([0.0, 0.0, 1.0, 0.0]))

check_close("relu: 8 elements (2 batches)",
    torch.relu(gpu([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])),
    torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]))

# ── THRESHOLD BACKWARD ───────────────────────────────────────────────

print("── THRESHOLD BACKWARD (relu gradient on hardware) ──")

# threshold_backward is called by autograd during backward through relu.
# We test it directly via the ATen function.
try:
    grad    = gpu([1.0, 2.0, 3.0, 4.0])
    inp     = gpu([0.5, -1.0, 2.0, -0.5])
    # result should be: [1,0,3,0] because inp>0 for elements 0,2
    result  = torch.ops.aten.threshold_backward(grad, inp, 0.0)
    expected = torch.tensor([1.0, 0.0, 3.0, 0.0])
    check_close("threshold_bwd: basic", result, expected)
except Exception as e:
    check(f"threshold_bwd basic (exception: {e})", False)

try:
    grad    = gpu([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    inp     = gpu([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    result  = torch.ops.aten.threshold_backward(grad, inp, 0.0)
    expected = torch.tensor([0.0, 20.0, 0.0, 40.0, 0.0, 60.0, 0.0, 80.0])
    check_close("threshold_bwd: 8 elements alternating", result, expected)
except Exception as e:
    check(f"threshold_bwd 8-elem (exception: {e})", False)

# ── AUTOGRAD THROUGH RELU ────────────────────────────────────────────

print("── AUTOGRAD THROUGH RELU ──")

try:
    x = gpu([1.0, -2.0, 3.0, -4.0]).detach().requires_grad_(True)
    y = torch.relu(x)
    y.sum().backward()
    grad = x.grad.cpu()
    expected = torch.tensor([1.0, 0.0, 1.0, 0.0])
    check_close("relu backward via autograd", grad, expected)
except Exception as e:
    check(f"relu autograd (exception: {e})", False)

# ── CHAINING ALL OPS ──────────────────────────────────────────────────

print("── CHAINING (all kernels) ──")

try:
    a = gpu([4.0, -2.0, 6.0, -8.0])
    b = gpu([2.0, 2.0, 3.0, 4.0])
    # relu(a/b + abs(a)) = relu([2,-1,2,-2] + [4,2,6,8]) = relu([6,1,8,6]) = [6,1,8,6]
    result = torch.relu(a / b + torch.abs(a))
    expected = torch.tensor([6.0, 1.0, 8.0, 6.0])
    check_close("chained: relu(a/b + abs(a))", result, expected, atol=0.05)
except Exception as e:
    check(f"chaining (exception: {e})", False)

# ── Summary ───────────────────────────────────────────────────────────

print()
if failed == 0:
    print(f"========================================")
    print(f"  ALL {passed} TESTS PASSED")
    print(f"  div/relu/threshold_bwd on RISC-V!")
    print(f"========================================")
else:
    print(f"  {passed} passed, {failed} FAILED")
    sys.exit(1)
