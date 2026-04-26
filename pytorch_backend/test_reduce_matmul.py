#!/usr/bin/env python3
"""
test_reduce_matmul.py — Tests for CP-6: sum, mean, mm, addmm.

Run from repository root with venv activated:
    python3 pytorch_backend/test_reduce_matmul.py
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

print(f"PyTorch {torch.__version__} — CP-6: reduction & matmul tests")
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

def check_close(name, got_gpu, expected_cpu, atol=1e-5):
    try:
        got_cpu = got_gpu.cpu()
        ok = torch.allclose(got_cpu, expected_cpu.float(), atol=atol)
        if not ok:
            print(f"  got:      {got_cpu}")
            print(f"  expected: {expected_cpu}")
        check(name, ok)
    except Exception as e:
        check(f"{name} (exception: {e})", False)

def gpu(data):
    return torch.tensor(data, dtype=torch.float32).to("verigpu")

# ====================================================================
# SUM — full reduction
# ====================================================================
print("── SUM (full) ──")

check_close("sum([1,2,3,4])",
    gpu([1.0, 2.0, 3.0, 4.0]).sum(),
    torch.tensor(10.0))

check_close("sum of zeros",
    gpu([0.0, 0.0, 0.0]).sum(),
    torch.tensor(0.0))

check_close("sum with negatives",
    gpu([-1.0, 2.0, -3.0, 4.0]).sum(),
    torch.tensor(2.0))

check_close("sum 2D tensor (all elements)",
    gpu([[1.0, 2.0], [3.0, 4.0]]).sum(),
    torch.tensor(10.0))

# ====================================================================
# SUM — along dimension
# ====================================================================
print("── SUM (dim) ──")

try:
    t = gpu([[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0]])

    # sum over dim=0 (columns): [5, 7, 9]
    r0 = t.sum(dim=0)
    check_close("sum dim=0 (over rows)",
        r0, torch.tensor([5.0, 7.0, 9.0]))

    # sum over dim=1 (rows): [6, 15]
    r1 = t.sum(dim=1)
    check_close("sum dim=1 (over cols)",
        r1, torch.tensor([6.0, 15.0]))

    # keepdim=True: shape preserved
    r0k = t.sum(dim=0, keepdim=True)
    check("sum dim=0 keepdim shape", r0k.shape == (1, 3))
    check_close("sum dim=0 keepdim values",
        r0k, torch.tensor([[5.0, 7.0, 9.0]]))

except Exception as e:
    check(f"sum dim (exception: {e})", False)

# ====================================================================
# MEAN — full reduction
# ====================================================================
print("── MEAN ──")

check_close("mean([1,2,3,4])",
    gpu([1.0, 2.0, 3.0, 4.0]).mean(),
    torch.tensor(2.5))

check_close("mean 2D",
    gpu([[2.0, 4.0], [6.0, 8.0]]).mean(),
    torch.tensor(5.0))

# ====================================================================
# MEAN — along dimension
# ====================================================================
print("── MEAN (dim) ──")

try:
    t = gpu([[1.0, 2.0],
             [3.0, 4.0],
             [5.0, 6.0]])

    # mean over dim=0: [3, 4]
    check_close("mean dim=0",
        t.mean(dim=0),
        torch.tensor([3.0, 4.0]))

    # mean over dim=1: [1.5, 3.5, 5.5]
    check_close("mean dim=1",
        t.mean(dim=1),
        torch.tensor([1.5, 3.5, 5.5]))

except Exception as e:
    check(f"mean dim (exception: {e})", False)

# ====================================================================
# MM — matrix multiplication
# ====================================================================
print("── MM ──")

# Basic 2x2 × 2x2
try:
    a = gpu([[1.0, 2.0],
             [3.0, 4.0]])
    b = gpu([[5.0, 6.0],
             [7.0, 8.0]])
    c = torch.mm(a, b)
    expected = torch.tensor([[19.0, 22.0],
                              [43.0, 50.0]])
    check_close("mm 2x2 @ 2x2", c, expected)
except Exception as e:
    check(f"mm 2x2 (exception: {e})", False)

# Non-square: 2x3 × 3x2 → 2x2
try:
    a = gpu([[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0]])
    b = gpu([[7.0, 8.0],
             [9.0, 10.0],
             [11.0, 12.0]])
    c = torch.mm(a, b)
    expected = torch.tensor([[58.0, 64.0],
                              [139.0, 154.0]])
    check_close("mm 2x3 @ 3x2", c, expected)
except Exception as e:
    check(f"mm 2x3 (exception: {e})", False)

# Identity matrix
try:
    a = gpu([[1.0, 2.0],
             [3.0, 4.0]])
    eye = gpu([[1.0, 0.0],
               [0.0, 1.0]])
    c = torch.mm(a, eye)
    check_close("mm with identity", c, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
except Exception as e:
    check(f"mm identity (exception: {e})", False)

# 1×N × N×1 → 1×1 (dot product as matrix)
try:
    a = gpu([[1.0, 2.0, 3.0]])
    b = gpu([[4.0], [5.0], [6.0]])
    c = torch.mm(a, b)
    check_close("mm 1x3 @ 3x1 (dot product)", c, torch.tensor([[32.0]]))
except Exception as e:
    check(f"mm dot (exception: {e})", False)

# Larger matrix
try:
    a_cpu = torch.randn(8, 16)
    b_cpu = torch.randn(16, 4)
    a_gpu = a_cpu.to("verigpu")
    b_gpu = b_cpu.to("verigpu")
    c_gpu = torch.mm(a_gpu, b_gpu)
    c_expected = torch.mm(a_cpu, b_cpu)
    check_close("mm 8x16 @ 16x4 (random)", c_gpu, c_expected, atol=1e-4)
except Exception as e:
    check(f"mm large (exception: {e})", False)

# ====================================================================
# ADDMM — bias + matmul
# ====================================================================
print("── ADDMM ──")

# addmm(bias, input, weight) = bias + input @ weight
try:
    bias = gpu([100.0, 200.0])
    inp = gpu([[1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0]])
    weight = gpu([[1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0]])
    # inp @ weight = [[4, 5], [10, 11]]
    # + bias       = [[104, 205], [110, 211]]
    c = torch.addmm(bias, inp, weight)
    expected = torch.tensor([[104.0, 205.0],
                              [110.0, 211.0]])
    check_close("addmm (bias + input @ weight)", c, expected)
except Exception as e:
    check(f"addmm (exception: {e})", False)

# addmm with alpha and beta
try:
    bias = gpu([10.0, 20.0])
    a = gpu([[1.0, 2.0],
             [3.0, 4.0]])
    b = gpu([[5.0, 6.0],
             [7.0, 8.0]])
    # beta=2, alpha=3: 2*bias + 3*(a@b)
    # a@b = [[19,22],[43,50]]
    # 2*[10,20] + 3*[[19,22],[43,50]] = [[20,40],[20,40]] + [[57,66],[129,150]]
    #                                  = [[77, 106], [149, 190]]
    c = torch.addmm(bias, a, b, beta=2.0, alpha=3.0)
    expected = torch.tensor([[77.0, 106.0],
                              [149.0, 190.0]])
    check_close("addmm with alpha=3, beta=2", c, expected)
except Exception as e:
    check(f"addmm alpha/beta (exception: {e})", False)

# ====================================================================
# Combined: sum + matmul patterns (like in neural networks)
# ====================================================================
print("── COMBINED ──")

try:
    # Simulate: loss = mean((prediction - target)^2)
    pred = gpu([2.0, 4.0, 6.0])
    target = gpu([1.0, 3.0, 5.0])
    diff = pred - target                     # [1, 1, 1]
    sq = diff * diff                          # [1, 1, 1]
    loss = sq.mean()                          # 1.0
    check_close("MSE loss pattern", loss, torch.tensor(1.0))
except Exception as e:
    check(f"MSE loss (exception: {e})", False)

try:
    # Simulate: output = relu(input @ weight + bias)
    inp = gpu([[1.0, -1.0],
               [2.0, 0.0]])
    weight = gpu([[1.0, 0.5],
                  [0.5, 1.0]])
    bias = gpu([-1.0, 0.0])

    mm_result = torch.mm(inp, weight)         # [[0.5, -0.5], [2.0, 1.0]]
    with_bias = mm_result + bias              # [[-0.5, -0.5], [1.0, 1.0]]
    output = torch.relu(with_bias)            # [[0.0, 0.0], [1.0, 1.0]]

    expected = torch.tensor([[0.0, 0.0],
                              [1.0, 1.0]])
    check_close("linear layer: relu(input @ weight + bias)", output, expected)
except Exception as e:
    check(f"linear layer (exception: {e})", False)

# ====================================================================
# REGRESSION
# ====================================================================
print("── REGRESSION ──")

check_close("CP-4: add", gpu([1.0,2.0]) + gpu([3.0,4.0]), torch.tensor([4.0,6.0]))
check_close("CP-5: mul", gpu([2.0,3.0]) * gpu([4.0,5.0]), torch.tensor([8.0,15.0]))
check_close("CP-5: relu", torch.relu(gpu([-1.0, 2.0])), torch.tensor([0.0, 2.0]))

try:
    back = torch.tensor([1.0, 2.0]).to("verigpu").cpu()
    check("CP-3: round-trip", torch.equal(back, torch.tensor([1.0, 2.0])))
except Exception as e:
    check(f"CP-3 regression (exception: {e})", False)

# ── Summary ────────────────────────────────────────────────────────────

print()
if failed == 0:
    print(f"========================================")
    print(f"  ALL {passed} TESTS PASSED")
    print(f"========================================")
else:
    print(f"  {passed} passed, {failed} FAILED")
    sys.exit(1)
