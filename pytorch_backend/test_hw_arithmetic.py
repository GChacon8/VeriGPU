#!/usr/bin/env python3
"""
test_hw_arithmetic.py — CP-7 (HW Roadmap): Arithmetic kernels on hardware.

Tests sub, mul, neg, abs running on RISC-V cores (plus add from CP-6).

Run from repository root:
    VERIGPU_USE_HW=1 python3 pytorch_backend/test_hw_arithmetic.py
"""

import torch
import subprocess
import sys
import os
import types

# ── Backend setup ──────────────────────────────────────────────────────

print("=== CP-7 (HW Roadmap): Arithmetic Kernels on Hardware ===")
print()

# Step 1: Assemble all kernels
kernels = {
    "vadd_f32": "kernels/vector_add_f32_param.asm",
    "vsub_f32": "kernels/vector_sub_f32.asm",
    "vmul_f32": "kernels/vector_mul_f32.asm",
    "vneg_f32": "kernels/vector_neg_f32.asm",
    "vabs_f32": "kernels/vector_abs_f32.asm",
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
        print(f"  ERROR assembling {name}: {result.stderr}")
        sys.exit(1)
    print(f"  {name}: assembled OK")
print()

# Step 2: Load backend
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

# Step 3: Load all kernels into GPU memory
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
            print(f"  got:      {got}")
            print(f"  expected: {expected_cpu}")
        check(name, ok)
    except Exception as e:
        check(f"{name} (exception: {e})", False)

def gpu(data):
    return torch.tensor(data, dtype=torch.float32).to("verigpu")

print("Step 3: Running tests...")
print()

# ── ADD (regression from CP-6) ────────────────────────────────────────

print("── ADD (CP-6 regression) ──")

check_close("add: [1,2,3,4] + [10,20,30,40]",
    gpu([1.0, 2.0, 3.0, 4.0]) + gpu([10.0, 20.0, 30.0, 40.0]),
    torch.tensor([11.0, 22.0, 33.0, 44.0]))

# ── SUB ───────────────────────────────────────────────────────────────

print("── SUB ──")

check_close("sub: [10,20,30,40] - [1,2,3,4]",
    gpu([10.0, 20.0, 30.0, 40.0]) - gpu([1.0, 2.0, 3.0, 4.0]),
    torch.tensor([9.0, 18.0, 27.0, 36.0]))

check_close("sub: 8 elements (2 batches)",
    gpu([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]) -
    gpu([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    torch.tensor([9.0, 18.0, 27.0, 36.0, 45.0, 54.0, 63.0, 72.0]))

check_close("sub: negative result",
    gpu([1.0, 2.0]) - gpu([10.0, 20.0]),
    torch.tensor([-9.0, -18.0]))

# ── MUL ───────────────────────────────────────────────────────────────

print("── MUL ──")

check_close("mul: [2,3,4,5] * [10,20,30,40]",
    gpu([2.0, 3.0, 4.0, 5.0]) * gpu([10.0, 20.0, 30.0, 40.0]),
    torch.tensor([20.0, 60.0, 120.0, 200.0]))

check_close("mul: with negatives",
    gpu([-1.0, 2.0, -3.0, 4.0]) * gpu([10.0, -20.0, 30.0, -40.0]),
    torch.tensor([-10.0, -40.0, -90.0, -160.0]))

check_close("mul: 8 elements (2 batches)",
    gpu([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]) *
    gpu([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
    torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]))

# ── NEG ───────────────────────────────────────────────────────────────

print("── NEG ──")

check_close("neg: -[1, -2, 3, 0]",
    -gpu([1.0, -2.0, 3.0, 0.0]),
    torch.tensor([-1.0, 2.0, -3.0, 0.0]))

check_close("neg: 8 elements",
    -gpu([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]))

# ── ABS ───────────────────────────────────────────────────────────────

print("── ABS ──")

check_close("abs: |[-1, 2, -3, 0]|",
    torch.abs(gpu([-1.0, 2.0, -3.0, 0.0])),
    torch.tensor([1.0, 2.0, 3.0, 0.0]))

check_close("abs: 8 elements",
    torch.abs(gpu([-1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0])),
    torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))

# ── CHAINING ──────────────────────────────────────────────────────────

print("── CHAINING (multiple kernels) ──")

try:
    a = gpu([1.0, 2.0, 3.0, 4.0])
    b = gpu([10.0, 20.0, 30.0, 40.0])
    c = (a + b) * a - b         # [11,22,33,44]*[1,2,3,4] - [10,20,30,40]
    expected = torch.tensor([1.0, 24.0, 69.0, 136.0])
    check_close("chained: (a+b)*a - b", c, expected)
except Exception as e:
    check(f"chaining (exception: {e})", False)

try:
    a = gpu([-3.0, 5.0, -7.0, 9.0])
    result = torch.abs(-a)      # abs(neg(a)) = abs([3,-5,7,-9]) = [3,5,7,9]
    expected = torch.tensor([3.0, 5.0, 7.0, 9.0])
    check_close("chained: abs(neg(a))", result, expected)
except Exception as e:
    check(f"chaining unary (exception: {e})", False)

# ── COMPARE WITH CPU ──────────────────────────────────────────────────

print("── CPU COMPARISON ──")

try:
    torch.manual_seed(42)
    a_cpu = torch.randn(16)
    b_cpu = torch.randn(16)

    a_gpu = a_cpu.to("verigpu")
    b_gpu = b_cpu.to("verigpu")

    for op_name, op in [("add", lambda a,b: a+b),
                         ("sub", lambda a,b: a-b),
                         ("mul", lambda a,b: a*b)]:
        gpu_result = op(a_gpu, b_gpu).cpu()
        cpu_result = op(a_cpu, b_cpu)
        ok = torch.allclose(gpu_result, cpu_result, atol=1e-4)
        check(f"random 16-elem {op_name} matches CPU", ok)

    for op_name, op in [("neg", lambda a: -a),
                         ("abs", lambda a: torch.abs(a))]:
        gpu_result = op(a_gpu).cpu()
        cpu_result = op(a_cpu)
        ok = torch.allclose(gpu_result, cpu_result, atol=1e-4)
        check(f"random 16-elem {op_name} matches CPU", ok)

except Exception as e:
    check(f"CPU comparison (exception: {e})", False)

# ── Summary ───────────────────────────────────────────────────────────

print()
if failed == 0:
    print(f"========================================")
    print(f"  ALL {passed} TESTS PASSED")
    print(f"  sub/mul/neg/abs on RISC-V hardware!")
    print(f"========================================")
else:
    print(f"  {passed} passed, {failed} FAILED")
    sys.exit(1)
