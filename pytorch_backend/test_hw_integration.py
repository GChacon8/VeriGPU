#!/usr/bin/env python3
"""
test_hw_integration.py — CP-4 (HW Roadmap): Verify runtime integration.

Tests:
  1. Extension reports HW availability correctly
  2. Host mode still works (all CP-3 tests pass)
  3. HW mode initializes the Verilator simulation
  4. Host-mode ops still work after HW init

Run from repository root with venv activated:
    python3 pytorch_backend/test_hw_integration.py

To test HW mode:
    VERIGPU_USE_HW=1 python3 pytorch_backend/test_hw_integration.py
"""

import torch
import sys
import os
import types

# ── Backend setup ──────────────────────────────────────────────────────
try:
    import _verigpu_C
except ImportError:
    print("ERROR: _verigpu_C not found. Build first:")
    print("  cd pytorch_backend && pip install -e . --no-build-isolation && cd ..")
    sys.exit(1)

_mod = types.ModuleType("torch.verigpu")
_mod.__path__ = []
sys.modules["torch.verigpu"] = _mod
torch.utils.rename_privateuse1_backend("verigpu")

hw_requested = os.environ.get("VERIGPU_USE_HW", "0") == "1"
hw_available = _verigpu_C.hw_available()
hw_active = _verigpu_C.is_hw_mode()

print(f"PyTorch {torch.__version__}")
print(f"========================================")
print(f"  CP-4 (HW Roadmap): Runtime Integration")
print(f"========================================")
print(f"  HW available (compiled with runtime): {hw_available}")
print(f"  HW requested (VERIGPU_USE_HW=1):      {hw_requested}")
print(f"  HW active:                             {hw_active}")
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

# ── Test 1: HW availability detection ─────────────────────────────────

print("── HW DETECTION ──")

check("hw_available() returns bool", isinstance(hw_available, bool))
check("is_hw_mode() returns bool", isinstance(hw_active, bool))

if hw_requested and hw_available:
    check("HW mode activated when requested", hw_active == True)
elif hw_requested and not hw_available:
    check("HW mode not active (runtime not linked)", hw_active == False)
else:
    check("HW mode not active (not requested)", hw_active == False)

# ── Test 2: Host-mode ops still work ──────────────────────────────────

print("── HOST OPS (must work in both modes) ──")

try:
    t = torch.empty(4, 3, device="verigpu")
    check("torch.empty on verigpu", t.device.type == "verigpu" and t.shape == (4, 3))
except Exception as e:
    check(f"torch.empty (exception: {e})", False)

try:
    original = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    on_gpu = original.to("verigpu")
    back = on_gpu.cpu()
    check("float32 round-trip (CPU → VeriGPU → CPU)", torch.equal(original, back))
except Exception as e:
    check(f"round-trip (exception: {e})", False)

try:
    a = torch.tensor([1.0, 2.0, 3.0]).to("verigpu")
    b = torch.tensor([10.0, 20.0, 30.0]).to("verigpu")
    c = a + b
    expected = torch.tensor([11.0, 22.0, 33.0])
    check("add on verigpu", torch.equal(c.cpu(), expected))
except Exception as e:
    check(f"add (exception: {e})", False)

try:
    a = torch.tensor([2.0, 3.0]).to("verigpu")
    b = torch.tensor([4.0, 5.0]).to("verigpu")
    c = a * b
    expected = torch.tensor([8.0, 15.0])
    check("mul on verigpu", torch.equal(c.cpu(), expected))
except Exception as e:
    check(f"mul (exception: {e})", False)

try:
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to("verigpu")
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]]).to("verigpu")
    c = torch.mm(a, b)
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
    check("mm on verigpu", torch.equal(c.cpu(), expected))
except Exception as e:
    check(f"mm (exception: {e})", False)

try:
    check("torch.zeros", torch.equal(
        torch.zeros(3, device="verigpu").cpu(), torch.zeros(3)))
    check("torch.ones", torch.equal(
        torch.ones(3, device="verigpu").cpu(), torch.ones(3)))
except Exception as e:
    check(f"zeros/ones (exception: {e})", False)

# ── Test 3: Autograd still works ──────────────────────────────────────

print("── AUTOGRAD ──")

try:
    a = torch.tensor([1.0, 2.0, 3.0]).to("verigpu").detach().requires_grad_(True)
    b = torch.tensor([4.0, 5.0, 6.0]).to("verigpu").detach().requires_grad_(True)
    c = (a * b).sum()
    c.backward()
    check("autograd: mul backward works", 
          a.grad is not None and torch.allclose(a.grad.cpu(), torch.tensor([4.0, 5.0, 6.0])))
except Exception as e:
    check(f"autograd (exception: {e})", False)

# ── Summary ────────────────────────────────────────────────────────────

print()
mode_str = "HARDWARE (Verilator)" if hw_active else "HOST CPU"
if failed == 0:
    print(f"========================================")
    print(f"  ALL {passed} TESTS PASSED")
    print(f"  Mode: {mode_str}")
    print(f"========================================")
else:
    print(f"  {passed} passed, {failed} FAILED")
    print(f"  Mode: {mode_str}")
    sys.exit(1)
