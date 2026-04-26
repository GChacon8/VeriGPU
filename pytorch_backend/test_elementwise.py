#!/usr/bin/env python3
"""
test_elementwise.py — Tests for CP-5 element-wise operations.

Run from repository root with venv activated:
    python3 pytorch_backend/test_elementwise.py
"""

import torch
import sys
import types

# ── Backend setup ──────────────────────────────────────────────────────
try:
    import _verigpu_C
except ImportError:
    print("ERROR: _verigpu_C not found. Build first.")
    sys.exit(1)

_mod = types.ModuleType("torch.verigpu")
_mod.__path__ = []
sys.modules["torch.verigpu"] = _mod
torch.utils.rename_privateuse1_backend("verigpu")

print(f"PyTorch {torch.__version__} — CP-5: element-wise ops tests")
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

def check_close(name, got_gpu, expected_cpu, atol=1e-6):
    """Compare GPU tensor against CPU expected value."""
    try:
        got_cpu = got_gpu.cpu()
        ok = torch.allclose(got_cpu, expected_cpu, atol=atol)
        if not ok:
            print(f"  got:      {got_cpu}")
            print(f"  expected: {expected_cpu}")
        check(name, ok)
    except Exception as e:
        check(f"{name} (exception: {e})", False)

# ── Helper: create GPU tensors ─────────────────────────────────────────

def gpu(data):
    return torch.tensor(data, dtype=torch.float32).to("verigpu")

# ====================================================================
# SUB tests
# ====================================================================
print("── SUB ──")

check_close("a - b",
    gpu([10.0, 20.0, 30.0]) - gpu([1.0, 2.0, 3.0]),
    torch.tensor([9.0, 18.0, 27.0]))

check_close("a - scalar",
    gpu([10.0, 20.0]) - 5.0,
    torch.tensor([5.0, 15.0]))

try:
    a = gpu([10.0, 20.0])
    a.sub_(gpu([1.0, 2.0]))
    check_close("in-place sub (a -= b)", a, torch.tensor([9.0, 18.0]))
except Exception as e:
    check(f"in-place sub (exception: {e})", False)

check_close("sub with alpha (a - 2*b)",
    torch.sub(gpu([10.0, 20.0]), gpu([1.0, 2.0]), alpha=2.0),
    torch.tensor([8.0, 16.0]))

# ====================================================================
# MUL tests
# ====================================================================
print("── MUL ──")

check_close("a * b",
    gpu([2.0, 3.0, 4.0]) * gpu([10.0, 20.0, 30.0]),
    torch.tensor([20.0, 60.0, 120.0]))

check_close("a * scalar",
    gpu([1.0, 2.0, 3.0]) * 5.0,
    torch.tensor([5.0, 10.0, 15.0]))

try:
    a = gpu([2.0, 3.0])
    a.mul_(gpu([10.0, 20.0]))
    check_close("in-place mul (a *= b)", a, torch.tensor([20.0, 60.0]))
except Exception as e:
    check(f"in-place mul (exception: {e})", False)

# ====================================================================
# DIV tests
# ====================================================================
print("── DIV ──")

check_close("a / b",
    gpu([10.0, 20.0, 30.0]) / gpu([2.0, 5.0, 10.0]),
    torch.tensor([5.0, 4.0, 3.0]))

check_close("a / scalar",
    gpu([10.0, 20.0, 30.0]) / 10.0,
    torch.tensor([1.0, 2.0, 3.0]))

try:
    a = gpu([10.0, 20.0])
    a.div_(gpu([2.0, 5.0]))
    check_close("in-place div (a /= b)", a, torch.tensor([5.0, 4.0]))
except Exception as e:
    check(f"in-place div (exception: {e})", False)

# ====================================================================
# NEG tests
# ====================================================================
print("── NEG ──")

check_close("-a (neg)",
    -gpu([1.0, -2.0, 3.0, 0.0]),
    torch.tensor([-1.0, 2.0, -3.0, 0.0]))

check_close("torch.neg",
    torch.neg(gpu([5.0, -5.0])),
    torch.tensor([-5.0, 5.0]))

# ====================================================================
# ABS tests
# ====================================================================
print("── ABS ──")

check_close("abs",
    torch.abs(gpu([-1.0, 2.0, -3.0, 0.0])),
    torch.tensor([1.0, 2.0, 3.0, 0.0]))

# ====================================================================
# RELU tests
# ====================================================================
print("── RELU ──")

check_close("relu (positive stays)",
    torch.relu(gpu([1.0, 2.0, 3.0])),
    torch.tensor([1.0, 2.0, 3.0]))

check_close("relu (negative → 0)",
    torch.relu(gpu([-1.0, -2.0, -3.0])),
    torch.tensor([0.0, 0.0, 0.0]))

check_close("relu (mixed)",
    torch.relu(gpu([-2.0, 0.0, 3.0, -0.5, 1.5])),
    torch.tensor([0.0, 0.0, 3.0, 0.0, 1.5]))

try:
    a = gpu([-1.0, 2.0, -3.0])
    a.relu_()
    check_close("in-place relu_", a, torch.tensor([0.0, 2.0, 0.0]))
except Exception as e:
    check(f"in-place relu_ (exception: {e})", False)

# ====================================================================
# 2D operations (shape preservation)
# ====================================================================
print("── 2D SHAPES ──")

a2d = gpu([[1.0, 2.0], [3.0, 4.0]])
b2d = gpu([[10.0, 20.0], [30.0, 40.0]])

check_close("2D sub", a2d - b2d, torch.tensor([[-9.0, -18.0], [-27.0, -36.0]]))
check_close("2D mul", a2d * b2d, torch.tensor([[10.0, 40.0], [90.0, 160.0]]))
check_close("2D div", b2d / a2d, torch.tensor([[10.0, 10.0], [10.0, 10.0]]))

# ====================================================================
# Chaining multiple operations
# ====================================================================
print("── CHAINING ──")

try:
    x = gpu([1.0, 2.0, 3.0])
    # (x * 2 + 1) / 3  →  [1.0, 1.666.., 2.333..]
    result = (x * 2.0 + 1.0) / 3.0
    expected = torch.tensor([1.0, 5.0/3.0, 7.0/3.0])
    check_close("chained ops: (x*2+1)/3", result, expected)
except Exception as e:
    check(f"chained ops (exception: {e})", False)

try:
    x = gpu([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = torch.relu(x * 2.0 - 1.0)  # [-5, -3, -1, 1, 3] → relu → [0,0,0,1,3]
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 3.0])
    check_close("chained: relu(x*2-1)", result, expected)
except Exception as e:
    check(f"chained relu (exception: {e})", False)

# ====================================================================
# Regression
# ====================================================================
print("── REGRESSION ──")

check_close("CP-4: add still works",
    gpu([1.0, 2.0]) + gpu([3.0, 4.0]),
    torch.tensor([4.0, 6.0]))

try:
    back = torch.tensor([1.0, 2.0]).to("verigpu").cpu()
    check("CP-3: round-trip", torch.equal(back, torch.tensor([1.0, 2.0])))
except Exception as e:
    check(f"CP-3: round-trip (exception: {e})", False)

try:
    z = torch.zeros(3, device="verigpu").cpu()
    o = torch.ones(3, device="verigpu").cpu()
    check("CP-3: zeros/ones",
          torch.equal(z, torch.zeros(3)) and torch.equal(o, torch.ones(3)))
except Exception as e:
    check(f"CP-3: zeros/ones (exception: {e})", False)

# ── Summary ────────────────────────────────────────────────────────────

print()
if failed == 0:
    print(f"========================================")
    print(f"  ALL {passed} TESTS PASSED")
    print(f"========================================")
else:
    print(f"  {passed} passed, {failed} FAILED")
    sys.exit(1)
