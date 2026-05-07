#!/usr/bin/env python3
"""
test_add.py — Verification tests for VeriGPU tensor add (CP-4).

Run (from the repository root, with venv activated):
    python3 pytorch_backend/test_add.py
"""

import torch
import sys
import types

# ── Backend setup ──────────────────────────────────────────────────────

try:
    import _verigpu_C
except ImportError:
    print("ERROR: _verigpu_C not found. Run: cd pytorch_backend && pip install -e . --no-build-isolation")
    sys.exit(1)

_mod = types.ModuleType("torch.verigpu")
_mod.__path__ = []
sys.modules["torch.verigpu"] = _mod
torch.utils.rename_privateuse1_backend("verigpu")

print(f"PyTorch {torch.__version__} — CP-4: tensor add tests")
print()

# ── Test framework ─────────────────────────────────────────────────────

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

# ── Tests ──────────────────────────────────────────────────────────────

# Test 1: Basic float add
try:
    a = torch.tensor([1.0, 2.0, 3.0, 4.0]).to("verigpu")
    b = torch.tensor([10.0, 20.0, 30.0, 40.0]).to("verigpu")
    c = a + b
    expected = torch.tensor([11.0, 22.0, 33.0, 44.0])
    check("basic float add (a + b)",
          c.device.type == "verigpu" and torch.equal(c.cpu(), expected))
except Exception as e:
    check(f"basic float add (exception: {e})", False)

# Test 2: torch.add function
try:
    a = torch.tensor([1.0, 2.0]).to("verigpu")
    b = torch.tensor([3.0, 4.0]).to("verigpu")
    c = torch.add(a, b)
    expected = torch.tensor([4.0, 6.0])
    check("torch.add(a, b)",
          torch.equal(c.cpu(), expected))
except Exception as e:
    check(f"torch.add (exception: {e})", False)

# Test 3: Add with alpha parameter
try:
    a = torch.tensor([10.0, 20.0]).to("verigpu")
    b = torch.tensor([1.0, 2.0]).to("verigpu")
    c = torch.add(a, b, alpha=3.0)  # result = a + 3*b
    expected = torch.tensor([13.0, 26.0])
    check("add with alpha (a + 3*b)",
          torch.equal(c.cpu(), expected))
except Exception as e:
    check(f"add with alpha (exception: {e})", False)

# Test 4: In-place add (a += b)
try:
    a = torch.tensor([1.0, 2.0, 3.0]).to("verigpu")
    b = torch.tensor([10.0, 20.0, 30.0]).to("verigpu")
    a.add_(b)
    expected = torch.tensor([11.0, 22.0, 33.0])
    check("in-place add (a += b)",
          torch.equal(a.cpu(), expected))
except Exception as e:
    check(f"in-place add (exception: {e})", False)

# Test 5: In-place add with alpha (a += alpha * b)
try:
    a = torch.tensor([100.0, 200.0]).to("verigpu")
    b = torch.tensor([1.0, 2.0]).to("verigpu")
    a.add_(b, alpha=-0.5)  # a = a + (-0.5)*b
    expected = torch.tensor([99.5, 199.0])
    check("in-place add with alpha (a += -0.5*b)",
          torch.equal(a.cpu(), expected))
except Exception as e:
    check(f"in-place add with alpha (exception: {e})", False)

# Test 6: 2D tensor add
try:
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to("verigpu")
    b = torch.tensor([[10.0, 20.0], [30.0, 40.0]]).to("verigpu")
    c = a + b
    expected = torch.tensor([[11.0, 22.0], [33.0, 44.0]])
    check("2D tensor add",
          torch.equal(c.cpu(), expected))
except Exception as e:
    check(f"2D tensor add (exception: {e})", False)

# Test 7: Add scalar to tensor
try:
    a = torch.tensor([1.0, 2.0, 3.0]).to("verigpu")
    c = a + 10.0
    expected = torch.tensor([11.0, 12.0, 13.0])
    check("tensor + scalar",
          torch.equal(c.cpu(), expected))
except Exception as e:
    check(f"tensor + scalar (exception: {e})", False)

# Test 8: Integer add
try:
    a = torch.tensor([1, 2, 3, 4], dtype=torch.int32).to("verigpu")
    b = torch.tensor([10, 20, 30, 40], dtype=torch.int32).to("verigpu")
    c = a + b
    expected = torch.tensor([11, 22, 33, 44], dtype=torch.int32)
    check("int32 add",
          torch.equal(c.cpu(), expected))
except Exception as e:
    check(f"int32 add (exception: {e})", False)

# Test 9: Result stays on device
try:
    a = torch.tensor([1.0, 2.0]).to("verigpu")
    b = torch.tensor([3.0, 4.0]).to("verigpu")
    c = a + b
    d = c + a  # chain operations on device
    expected = torch.tensor([5.0, 8.0])
    check("chained adds (result stays on device)",
          d.device.type == "verigpu" and torch.equal(d.cpu(), expected))
except Exception as e:
    check(f"chained adds (exception: {e})", False)

# Test 10: Larger tensor (256 elements)
try:
    a_cpu = torch.arange(256, dtype=torch.float32)
    b_cpu = torch.ones(256, dtype=torch.float32) * 1000
    a = a_cpu.to("verigpu")
    b = b_cpu.to("verigpu")
    c = a + b
    expected = a_cpu + b_cpu
    check("large tensor add (256 elements)",
          torch.equal(c.cpu(), expected))
except Exception as e:
    check(f"large tensor add (exception: {e})", False)

# ── CP-3 regression: make sure old tests still pass ────────────────────

# Test 11: Round-trip still works
try:
    original = torch.tensor([1.0, 2.0, 3.0])
    back = original.to("verigpu").cpu()
    check("CP-3 regression: round-trip",
          torch.equal(original, back))
except Exception as e:
    check(f"CP-3 regression: round-trip (exception: {e})", False)

# Test 12: zeros/ones still work
try:
    z = torch.zeros(4, device="verigpu").cpu()
    o = torch.ones(4, device="verigpu").cpu()
    check("CP-3 regression: zeros/ones",
          torch.equal(z, torch.zeros(4)) and torch.equal(o, torch.ones(4)))
except Exception as e:
    check(f"CP-3 regression: zeros/ones (exception: {e})", False)

# ── Summary ────────────────────────────────────────────────────────────

print()
if failed == 0:
    print(f"========================================")
    print(f"  ALL {passed} TESTS PASSED")
    print(f"========================================")
else:
    print(f"  {passed} passed, {failed} FAILED")
    sys.exit(1)
