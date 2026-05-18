#!/usr/bin/env python3
"""
test_hw_add.py — CP-6 (HW Roadmap): First operation on RISC-V hardware!

Tests that torch.add dispatches to the vector_add_f32 kernel running
on the actual RISC-V cores (simulated via Verilator).

Run from repository root:
    VERIGPU_USE_HW=1 python3 pytorch_backend/test_hw_add.py
"""

import torch
import subprocess
import sys
import os
import types

# ── Step 1: Assemble the kernel ───────────────────────────────────────

print("=== CP-6 (HW Roadmap): First Add on Hardware ===")
print()

kernel_asm = "kernels/vector_add_f32_param.asm"
kernel_hex = "build/vector_add_f32_param.hex"

if not os.path.exists(kernel_asm):
    print(f"ERROR: {kernel_asm} not found")
    sys.exit(1)

print(f"Step 1: Assembling kernel...")
os.makedirs("build", exist_ok=True)
result = subprocess.run(
    ["python3", "verigpu/assembler.py",
     "--in-asm", kernel_asm,
     "--out-hex", kernel_hex],
    capture_output=True, text=True)

if result.returncode != 0:
    print(f"ERROR: Assembly failed:\n{result.stderr}")
    sys.exit(1)

# Load hex file
with open(kernel_hex) as f:
    kernel_words = [int(line.strip(), 16) for line in f if line.strip()]

print(f"  Assembled {len(kernel_words)} instructions")
print(f"  Hex: {[hex(w) for w in kernel_words[:5]]}...")
print()

# ── Step 2: Load backend ─────────────────────────────────────────────

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
    print("ERROR: HW mode not active. Run with:")
    print("  VERIGPU_USE_HW=1 python3 pytorch_backend/test_hw_add.py")
    sys.exit(1)

print(f"Step 2: Backend loaded (HW mode active)")
print()

# ── Step 3: Load kernel into GPU memory ──────────────────────────────

print(f"Step 3: Loading kernel to GPU memory...")
_verigpu_C.load_kernel("vadd_f32", kernel_words)
print()

# ── Step 4: Tests ────────────────────────────────────────────────────

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

print("Step 4: Running tests...")
print()
print("── HARDWARE ADD ──")

# Test 1: Basic 4-element add (1 batch)
try:
    a = torch.tensor([1.0, 2.0, 3.0, 4.0]).to("verigpu")
    b = torch.tensor([10.0, 20.0, 30.0, 40.0]).to("verigpu")
    c = a + b
    result = c.cpu()
    expected = torch.tensor([11.0, 22.0, 33.0, 44.0])
    ok = torch.allclose(result, expected, atol=1e-5)
    if not ok:
        print(f"  got:      {result}")
        print(f"  expected: {expected}")
    check("4-element float add (1 batch, 4 cores)", ok)
except Exception as e:
    import traceback
    traceback.print_exc()
    check(f"4-element add (exception: {e})", False)

# Test 2: 8-element add (2 batches)
try:
    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).to("verigpu")
    b = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]).to("verigpu")
    c = a + b
    result = c.cpu()
    expected = torch.tensor([11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0])
    ok = torch.allclose(result, expected, atol=1e-5)
    if not ok:
        print(f"  got:      {result}")
        print(f"  expected: {expected}")
    check("8-element float add (2 batches)", ok)
except Exception as e:
    check(f"8-element add (exception: {e})", False)

# Test 3: Verify result stays on device
try:
    a = torch.tensor([1.0, 2.0]).to("verigpu")
    b = torch.tensor([3.0, 4.0]).to("verigpu")
    c = a + b
    check("result on verigpu device", c.device.type == "verigpu")
except Exception as e:
    check(f"device check (exception: {e})", False)

# Test 4: Compare with CPU
try:
    a_cpu = torch.randn(16)
    b_cpu = torch.randn(16)
    expected = a_cpu + b_cpu

    a_gpu = a_cpu.to("verigpu")
    b_gpu = b_cpu.to("verigpu")
    c_gpu = a_gpu + b_gpu
    result = c_gpu.cpu()

    ok = torch.allclose(result, expected, atol=1e-4)
    if not ok:
        diff = (result - expected).abs().max().item()
        print(f"  max diff: {diff}")
    check("16 random floats match CPU", ok)
except Exception as e:
    check(f"random floats (exception: {e})", False)

# Test 5: Chained adds on device
try:
    a = torch.tensor([1.0, 2.0, 3.0, 4.0]).to("verigpu")
    b = torch.tensor([10.0, 10.0, 10.0, 10.0]).to("verigpu")
    c = a + b      # [11, 12, 13, 14] — kernel launch 1
    d = c + b      # [21, 22, 23, 24] — kernel launch 2
    result = d.cpu()
    expected = torch.tensor([21.0, 22.0, 23.0, 24.0])
    check("chained adds (2 kernel launches)", torch.allclose(result, expected, atol=1e-5))
except Exception as e:
    check(f"chained adds (exception: {e})", False)

print()
print("── FALLBACK TO HOST (non-float, scalar, alpha!=1) ──")

# Test 6: Integer add falls back to host (no float kernel)
try:
    a = torch.tensor([1, 2, 3], dtype=torch.int32).to("verigpu")
    b = torch.tensor([10, 20, 30], dtype=torch.int32).to("verigpu")
    c = (a + b).cpu()
    check("int32 add (host fallback)", torch.equal(c, torch.tensor([11, 22, 33], dtype=torch.int32)))
except Exception as e:
    check(f"int32 fallback (exception: {e})", False)

# Test 7: Add with alpha falls back to host
try:
    a = torch.tensor([1.0, 2.0]).to("verigpu")
    b = torch.tensor([10.0, 20.0]).to("verigpu")
    c = torch.add(a, b, alpha=2.0).cpu()
    check("add with alpha=2 (host fallback)", torch.allclose(c, torch.tensor([21.0, 42.0])))
except Exception as e:
    check(f"alpha fallback (exception: {e})", False)

print()
print("── REGRESSION ──")

try:
    check("round-trip", torch.equal(
        torch.tensor([1.0, 2.0]).to("verigpu").cpu(), torch.tensor([1.0, 2.0])))
except Exception as e:
    check(f"round-trip (exception: {e})", False)

try:
    check("mul (host)", torch.equal(
        (torch.tensor([2.0, 3.0]).to("verigpu") * torch.tensor([4.0, 5.0]).to("verigpu")).cpu(),
        torch.tensor([8.0, 15.0])))
except Exception as e:
    check(f"mul (exception: {e})", False)

# ── Summary ───────────────────────────────────────────────────────────

print()
if failed == 0:
    print(f"========================================")
    print(f"  ALL {passed} TESTS PASSED")
    print(f"  PyTorch add runs on RISC-V hardware!")
    print(f"========================================")
else:
    print(f"  {passed} passed, {failed} FAILED")
    sys.exit(1)
