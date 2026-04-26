#!/usr/bin/env python3
"""
test_backend.py — Verification tests for VeriGPU PyTorch backend (CP-3).

Run (from the repository root, with venv activated):
    python3 pytorch_backend/test_backend.py
"""

import torch
import sys
import types

# ── Load the C++ extension ─────────────────────────────────────────────

try:
    import _verigpu_C
except ImportError:
    print("ERROR: _verigpu_C extension not found.")
    print("  Did you build it? Run from pytorch_backend/:")
    print("    pip install -e . --no-build-isolation")
    sys.exit(1)

# ── Register backend ──────────────────────────────────────────────────
# PyTorch 2.11+ tries to "import torch.verigpu" internally.
# We must register a stub module BEFORE calling rename_privateuse1_backend.

_mod = types.ModuleType("torch.verigpu")
_mod.__path__ = []
sys.modules["torch.verigpu"] = _mod

torch.utils.rename_privateuse1_backend("verigpu")

print(f"PyTorch version: {torch.__version__}")
print(f"VeriGPU backend available: {_verigpu_C.is_available()}")
print(f"VeriGPU device count: {_verigpu_C.device_count()}")
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

# Test 1: Device exists and is recognized
try:
    dev = torch.device("verigpu")
    check("device 'verigpu' is recognized", dev.type == "verigpu")
except Exception as e:
    check(f"device 'verigpu' is recognized (exception: {e})", False)

# Test 2: Create empty tensor on device
try:
    t = torch.empty(4, 3, device="verigpu")
    check("torch.empty on verigpu",
          t.device.type == "verigpu" and t.shape == (4, 3))
except Exception as e:
    check(f"torch.empty on verigpu (exception: {e})", False)

# Test 3: Float32 round-trip
try:
    original = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    on_gpu = original.to("verigpu")
    back = on_gpu.cpu()
    check("float32 round-trip (CPU → VeriGPU → CPU)",
          on_gpu.device.type == "verigpu" and torch.equal(original, back))
except Exception as e:
    check(f"float32 round-trip (exception: {e})", False)

# Test 4: Int32 round-trip
try:
    original = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
    on_gpu = original.to("verigpu")
    back = on_gpu.cpu()
    check("int32 round-trip",
          torch.equal(original, back))
except Exception as e:
    check(f"int32 round-trip (exception: {e})", False)

# Test 5: 2D tensor round-trip
try:
    original = torch.tensor([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0]])
    on_gpu = original.to("verigpu")
    back = on_gpu.cpu()
    check("2D tensor round-trip",
          back.shape == (2, 3) and torch.equal(original, back))
except Exception as e:
    check(f"2D tensor round-trip (exception: {e})", False)

# Test 6: Large tensor round-trip
try:
    original = torch.randn(256, 256)
    on_gpu = original.to("verigpu")
    back = on_gpu.cpu()
    check("large tensor (256x256) round-trip",
          torch.equal(original, back))
except Exception as e:
    check(f"large tensor round-trip (exception: {e})", False)

# Test 7: torch.zeros on device
try:
    t = torch.zeros(3, 4, device="verigpu")
    back = t.cpu()
    expected = torch.zeros(3, 4)
    check("torch.zeros on verigpu",
          t.device.type == "verigpu" and torch.equal(back, expected))
except Exception as e:
    check(f"torch.zeros on verigpu (exception: {e})", False)

# Test 8: torch.ones on device
try:
    t = torch.ones(2, 5, device="verigpu")
    back = t.cpu()
    expected = torch.ones(2, 5)
    check("torch.ones on verigpu",
          torch.equal(back, expected))
except Exception as e:
    check(f"torch.ones on verigpu (exception: {e})", False)

# Test 9: Multiple tensors, independent lifetimes
try:
    a = torch.tensor([1.0, 2.0]).to("verigpu")
    b = torch.tensor([3.0, 4.0]).to("verigpu")
    a_back = a.cpu()
    b_back = b.cpu()
    check("multiple independent tensors",
          torch.equal(a_back, torch.tensor([1.0, 2.0])) and
          torch.equal(b_back, torch.tensor([3.0, 4.0])))
except Exception as e:
    check(f"multiple independent tensors (exception: {e})", False)

# Test 10: Correct dtype preservation
try:
    for dtype, name in [(torch.float32, "float32"),
                         (torch.float64, "float64"),
                         (torch.int32, "int32"),
                         (torch.int64, "int64")]:
        t = torch.empty(4, device="verigpu", dtype=dtype)
        ok = (t.dtype == dtype)
        if not ok:
            check(f"dtype preservation ({name})", False)
            break
    else:
        check("dtype preservation (float32/64, int32/64)", True)
except Exception as e:
    check(f"dtype preservation (exception: {e})", False)

# ── Summary ────────────────────────────────────────────────────────────

print()
if failed == 0:
    print(f"========================================")
    print(f"  ALL {passed} TESTS PASSED")
    print(f"========================================")
else:
    print(f"  {passed} passed, {failed} FAILED")
    sys.exit(1)
    