#!/usr/bin/env python3
"""
test_autograd.py — Tests for CP-7: backward pass and autograd.

Tests that gradients flow correctly through:
  - element-wise ops (add, sub, mul, div)
  - relu
  - matrix multiplication (mm)
  - sum / mean
  - combined patterns (linear layer, MSE loss)

Run from repository root with venv activated:
    python3 pytorch_backend/test_autograd.py
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

print(f"PyTorch {torch.__version__} — CP-7: autograd tests")
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

def check_close(name, got, expected, atol=1e-5):
    try:
        g = got.cpu() if got.device.type != 'cpu' else got
        e = expected.cpu() if expected.device.type != 'cpu' else expected
        ok = torch.allclose(g.float(), e.float(), atol=atol)
        if not ok:
            print(f"  got:      {g}")
            print(f"  expected: {e}")
        check(name, ok)
    except Exception as ex:
        check(f"{name} (exception: {ex})", False)

# ── Helper ─────────────────────────────────────────────────────────────

def gpu_grad(data):
    """Create a float32 tensor on verigpu with requires_grad=True."""
    return torch.tensor(data, dtype=torch.float32, requires_grad=True).to("verigpu")

# ====================================================================
# View ops (prerequisites for autograd)
# ====================================================================
print("── VIEW OPS ──")

# t() — transpose
try:
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to("verigpu")
    at = a.t()
    check("t() shape", at.shape == (3, 2))
    check_close("t() values", at, torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]))
except Exception as e:
    check(f"t() (exception: {e})", False)

# reshape
try:
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).to("verigpu")
    b = a.reshape(2, 3)
    check("reshape shape", b.shape == (2, 3))
    check_close("reshape values", b, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
except Exception as e:
    check(f"reshape (exception: {e})", False)

# _local_scalar_dense (via .item())
try:
    a = torch.tensor([42.0]).to("verigpu")
    val = a.item()
    check("item() extracts scalar", abs(val - 42.0) < 1e-6)
except Exception as e:
    check(f"item() (exception: {e})", False)

# threshold_backward
try:
    grad = torch.tensor([1.0, 1.0, 1.0, 1.0]).to("verigpu")
    inp  = torch.tensor([-1.0, 0.0, 0.5, 2.0]).to("verigpu")
    result = torch.ops.aten.threshold_backward(grad, inp, 0.0)
    expected = torch.tensor([0.0, 0.0, 1.0, 1.0])
    check_close("threshold_backward", result, expected)
except Exception as e:
    check(f"threshold_backward (exception: {e})", False)

# ====================================================================
# Autograd: element-wise ops
# ====================================================================
print("── AUTOGRAD: ELEMENT-WISE ──")

# add backward: d(a+b)/da = 1, d(a+b)/db = 1
try:
    a = gpu_grad([1.0, 2.0, 3.0])
    b = gpu_grad([4.0, 5.0, 6.0])
    c = a + b
    loss = c.sum()
    loss.backward()
    check_close("add grad_a", a.grad, torch.ones(3))
    check_close("add grad_b", b.grad, torch.ones(3))
except Exception as e:
    check(f"add backward (exception: {e})", False)

# mul backward: d(a*b)/da = b, d(a*b)/db = a
try:
    a = gpu_grad([2.0, 3.0])
    b = gpu_grad([4.0, 5.0])
    c = a * b
    loss = c.sum()
    loss.backward()
    check_close("mul grad_a (=b)", a.grad, torch.tensor([4.0, 5.0]))
    check_close("mul grad_b (=a)", b.grad, torch.tensor([2.0, 3.0]))
except Exception as e:
    check(f"mul backward (exception: {e})", False)

# sub backward: d(a-b)/da = 1, d(a-b)/db = -1
try:
    a = gpu_grad([10.0, 20.0])
    b = gpu_grad([1.0, 2.0])
    c = a - b
    loss = c.sum()
    loss.backward()
    check_close("sub grad_a", a.grad, torch.ones(2))
    check_close("sub grad_b", b.grad, -torch.ones(2))
except Exception as e:
    check(f"sub backward (exception: {e})", False)

# relu backward
try:
    a = gpu_grad([-1.0, 0.0, 1.0, 2.0])
    c = torch.relu(a)
    loss = c.sum()
    loss.backward()
    expected_grad = torch.tensor([0.0, 0.0, 1.0, 1.0])
    check_close("relu grad", a.grad, expected_grad)
except Exception as e:
    check(f"relu backward (exception: {e})", False)

# ====================================================================
# Autograd: reduction
# ====================================================================
print("── AUTOGRAD: REDUCTION ──")

# sum backward: gradient expands to input shape
try:
    a = gpu_grad([1.0, 2.0, 3.0])
    loss = a.sum()
    loss.backward()
    check_close("sum grad (all ones)", a.grad, torch.ones(3))
except Exception as e:
    check(f"sum backward (exception: {e})", False)

# mean backward: gradient = 1/n for each element
try:
    a = gpu_grad([1.0, 2.0, 3.0, 4.0])
    loss = a.mean()
    loss.backward()
    check_close("mean grad (1/n)", a.grad, torch.full((4,), 0.25))
except Exception as e:
    check(f"mean backward (exception: {e})", False)

# ====================================================================
# Autograd: matmul
# ====================================================================
print("── AUTOGRAD: MATMUL ──")

# mm backward:
#   grad_A = grad_output @ B.T
#   grad_B = A.T @ grad_output
try:
    A = gpu_grad([[1.0, 2.0], [3.0, 4.0]])
    B = gpu_grad([[5.0, 6.0], [7.0, 8.0]])
    C = torch.mm(A, B)
    loss = C.sum()
    loss.backward()

    # grad_output is all ones (from .sum())
    # grad_A = ones(2,2) @ B.T = [[5+7, 6+8], [5+7, 6+8]] = [[12,14],[12,14]]
    expected_grad_A = torch.tensor([[12.0, 14.0], [12.0, 14.0]])
    # grad_B = A.T @ ones(2,2) = [[1+3, 1+3], [2+4, 2+4]] = [[4,4],[6,6]]
    expected_grad_B = torch.tensor([[4.0, 4.0], [6.0, 6.0]])

    check_close("mm grad_A", A.grad, expected_grad_A)
    check_close("mm grad_B", B.grad, expected_grad_B)
except Exception as e:
    check(f"mm backward (exception: {e})", False)

# ====================================================================
# Autograd: combined pattern (linear layer + MSE)
# ====================================================================
print("── AUTOGRAD: COMBINED ──")

# Simulate: y = relu(x @ w + b), loss = mean((y - target)^2)
try:
    x = gpu_grad([[1.0, 0.5]])       # [1, 2]
    w = gpu_grad([[0.3, -0.2],
                  [0.1,  0.4]])      # [2, 2]
    b = gpu_grad([0.1, -0.1])        # [2]

    # Forward
    h = torch.mm(x, w)               # [1, 2]
    h_b = h + b                      # [1, 2] + [2] broadcast
    y = torch.relu(h_b)              # [1, 2]

    target = torch.tensor([[1.0, 0.0]]).to("verigpu")
    diff = y - target
    loss = (diff * diff).mean()

    # Backward
    loss.backward()

    # Verify gradients exist and have correct shape
    check("combined: x.grad exists", x.grad is not None)
    check("combined: w.grad exists", w.grad is not None)
    check("combined: b.grad exists", b.grad is not None)
    check("combined: x.grad shape", x.grad.shape == (1, 2))
    check("combined: w.grad shape", w.grad.shape == (2, 2))
    check("combined: b.grad shape", b.grad.shape == (2,))

    # Verify against CPU computation
    x_cpu = torch.tensor([[1.0, 0.5]], requires_grad=True)
    w_cpu = torch.tensor([[0.3, -0.2], [0.1, 0.4]], requires_grad=True)
    b_cpu = torch.tensor([0.1, -0.1], requires_grad=True)
    h_cpu = torch.mm(x_cpu, w_cpu) + b_cpu
    y_cpu = torch.relu(h_cpu)
    target_cpu = torch.tensor([[1.0, 0.0]])
    diff_cpu = y_cpu - target_cpu
    loss_cpu = (diff_cpu * diff_cpu).mean()
    loss_cpu.backward()

    check_close("combined: x.grad matches CPU", x.grad, x_cpu.grad)
    check_close("combined: w.grad matches CPU", w.grad, w_cpu.grad)
    check_close("combined: b.grad matches CPU", b.grad, b_cpu.grad)
    check_close("combined: loss matches CPU", loss, loss_cpu)

except Exception as e:
    import traceback
    traceback.print_exc()
    check(f"combined backward (exception: {e})", False)

# ====================================================================
# Gradient accumulation (multiple backward)
# ====================================================================
print("── GRAD ACCUMULATION ──")

try:
    a = gpu_grad([1.0, 2.0])
    b = gpu_grad([3.0, 4.0])

    # First backward
    c1 = (a * b).sum()
    c1.backward()
    grad1 = a.grad.cpu().clone()

    # Second backward (gradients accumulate)
    c2 = (a * b).sum()
    c2.backward()
    grad2 = a.grad.cpu().clone()

    # After 2 backwards, grad should be 2x the single-pass grad
    check_close("grad accumulation (2x)", grad2, 2 * grad1)
except Exception as e:
    check(f"grad accumulation (exception: {e})", False)

# ====================================================================
# REGRESSION
# ====================================================================
print("── REGRESSION ──")

try:
    r = torch.tensor([1.0, 2.0]).to("verigpu").cpu()
    check("CP-3: round-trip", torch.equal(r, torch.tensor([1.0, 2.0])))
except Exception as e:
    check(f"CP-3 (exception: {e})", False)

try:
    r = (torch.tensor([1.0, 2.0]).to("verigpu") + torch.tensor([3.0, 4.0]).to("verigpu")).cpu()
    check("CP-4: add", torch.equal(r, torch.tensor([4.0, 6.0])))
except Exception as e:
    check(f"CP-4 (exception: {e})", False)

try:
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to("verigpu")
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]]).to("verigpu")
    c = torch.mm(a, b).cpu()
    check("CP-6: mm", torch.equal(c, torch.tensor([[19.0, 22.0], [43.0, 50.0]])))
except Exception as e:
    check(f"CP-6 (exception: {e})", False)

# ── Summary ────────────────────────────────────────────────────────────

print()
if failed == 0:
    print(f"========================================")
    print(f"  ALL {passed} TESTS PASSED")
    print(f"========================================")
else:
    print(f"  {passed} passed, {failed} FAILED")
    sys.exit(1)
