#!/usr/bin/env python3
"""
test_hw_autograd.py — CP-11 (HW Roadmap): Autograd end-to-end on hardware.

Verifies that loss.backward() produces correct gradients when the
forward pass runs on RISC-V cores. No new kernels or C++ changes —
this is pure verification.

The chain: Python autograd → forward on HW → backward on HW → gradients match CPU.

Run from repository root:
    VERIGPU_USE_HW=1 python3 pytorch_backend/test_hw_autograd.py
"""

import torch
import torch.nn as nn
import subprocess
import sys
import os
import types

# ── Backend setup ──────────────────────────────────────────────────────

print("=== CP-11 (HW Roadmap): Autograd on Hardware ===")
print()

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

print("Assembling and loading kernels...")
os.makedirs("build", exist_ok=True)
for name, asm_path in kernels.items():
    hex_path = f"build/{os.path.basename(asm_path).replace('.asm', '.hex')}"
    subprocess.run(
        ["python3", "verigpu/assembler.py", "--in-asm", asm_path, "--out-hex", hex_path],
        capture_output=True, text=True, check=True)

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

for name, asm_path in kernels.items():
    hex_path = f"build/{os.path.basename(asm_path).replace('.asm', '.hex')}"
    with open(hex_path) as f:
        words = [int(line.strip(), 16) for line in f if line.strip()]
    _verigpu_C.load_kernel(name, words)
print("  Done")
print()

# ── Test infrastructure ───────────────────────────────────────────────

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

def gpu_leaf(data):
    """Create a leaf tensor on VeriGPU with requires_grad=True."""
    return torch.tensor(data, dtype=torch.float32).to("verigpu").detach().requires_grad_(True)

# ── 1. ELEMENT-WISE GRADIENTS ────────────────────────────────────────

print("── ELEMENT-WISE GRADIENTS ──")

# add backward: d(a+b)/da = 1, d(a+b)/db = 1
try:
    a = gpu_leaf([1.0, 2.0, 3.0, 4.0])
    b = gpu_leaf([10.0, 20.0, 30.0, 40.0])
    c = a + b
    c.sum().backward()
    check("add grad_a = ones", torch.allclose(a.grad.cpu(), torch.ones(4), atol=1e-4))
    check("add grad_b = ones", torch.allclose(b.grad.cpu(), torch.ones(4), atol=1e-4))
except Exception as e:
    check(f"add backward ({e})", False)

# sub backward: d(a-b)/da = 1, d(a-b)/db = -1
try:
    a = gpu_leaf([1.0, 2.0, 3.0, 4.0])
    b = gpu_leaf([10.0, 20.0, 30.0, 40.0])
    c = a - b
    c.sum().backward()
    check("sub grad_a = ones", torch.allclose(a.grad.cpu(), torch.ones(4), atol=1e-4))
    check("sub grad_b = -ones", torch.allclose(b.grad.cpu(), -torch.ones(4), atol=1e-4))
except Exception as e:
    check(f"sub backward ({e})", False)

# mul backward: d(a*b)/da = b, d(a*b)/db = a
try:
    a = gpu_leaf([1.0, 2.0, 3.0, 4.0])
    b = gpu_leaf([10.0, 20.0, 30.0, 40.0])
    c = a * b
    c.sum().backward()
    check("mul grad_a = b", torch.allclose(a.grad.cpu(), torch.tensor([10., 20., 30., 40.]), atol=1e-3))
    check("mul grad_b = a", torch.allclose(b.grad.cpu(), torch.tensor([1., 2., 3., 4.]), atol=1e-3))
except Exception as e:
    check(f"mul backward ({e})", False)

# relu backward: d(relu(a))/da = (a > 0) ? 1 : 0
try:
    a = gpu_leaf([1.0, -2.0, 3.0, -4.0])
    c = torch.relu(a)
    c.sum().backward()
    check("relu grad = [1,0,1,0]", torch.allclose(a.grad.cpu(), torch.tensor([1., 0., 1., 0.]), atol=1e-4))
except Exception as e:
    check(f"relu backward ({e})", False)

# ── 2. MATMUL GRADIENTS ──────────────────────────────────────────────

print("── MATMUL GRADIENTS ──")

# mm backward: d(A@B)/dA = grad @ B.T, d(A@B)/dB = A.T @ grad
try:
    torch.manual_seed(42)
    a_data = torch.randn(3, 2)
    b_data = torch.randn(2, 4)

    # GPU
    a_gpu = a_data.to("verigpu").detach().requires_grad_(True)
    b_gpu = b_data.to("verigpu").detach().requires_grad_(True)
    c_gpu = torch.mm(a_gpu, b_gpu)
    c_gpu.sum().backward()
    grad_a_gpu = a_gpu.grad.cpu()
    grad_b_gpu = b_gpu.grad.cpu()

    # CPU reference
    a_cpu = a_data.clone().requires_grad_(True)
    b_cpu = b_data.clone().requires_grad_(True)
    c_cpu = torch.mm(a_cpu, b_cpu)
    c_cpu.sum().backward()

    check("mm grad_A matches CPU",
          torch.allclose(grad_a_gpu, a_cpu.grad, atol=0.01))
    check("mm grad_B matches CPU",
          torch.allclose(grad_b_gpu, b_cpu.grad, atol=0.01))
except Exception as e:
    import traceback; traceback.print_exc()
    check(f"mm backward ({e})", False)

# ── 3. MSE LOSS ──────────────────────────────────────────────────────

print("── MSE LOSS BACKWARD ──")

# MSE = mean((pred - target)^2)
# d(MSE)/d(pred) = 2*(pred - target) / n
try:
    pred   = gpu_leaf([1.0, 2.0, 3.0, 4.0])
    target = torch.tensor([0.0, 1.0, 2.0, 3.0]).to("verigpu")
    diff = pred - target
    loss = (diff * diff).mean()
    loss.backward()

    # Expected: 2 * (pred - target) / n = 2 * [1,1,1,1] / 4 = [0.5,0.5,0.5,0.5]
    expected_grad = torch.tensor([0.5, 0.5, 0.5, 0.5])
    ok = torch.allclose(pred.grad.cpu(), expected_grad, atol=0.01)
    if not ok:
        print(f"  got:      {pred.grad.cpu()}")
        print(f"  expected: {expected_grad}")
    check("MSE grad = 2*(pred-target)/n", ok)
except Exception as e:
    import traceback; traceback.print_exc()
    check(f"MSE backward ({e})", False)

# ── 4. nn.Linear BACKWARD ────────────────────────────────────────────

print("── nn.Linear BACKWARD ──")

try:
    torch.manual_seed(42)

    # GPU
    model_gpu = nn.Linear(3, 2).to("verigpu")
    x_gpu = torch.randn(4, 3).to("verigpu")
    out_gpu = model_gpu(x_gpu)
    out_gpu.sum().backward()
    w_grad_gpu = model_gpu.weight.grad.cpu()
    b_grad_gpu = model_gpu.bias.grad.cpu()

    # CPU reference
    torch.manual_seed(42)
    model_cpu = nn.Linear(3, 2)
    x_cpu = x_gpu.detach().cpu()
    out_cpu = model_cpu(x_cpu)
    out_cpu.sum().backward()

    check("Linear weight grad matches CPU",
          torch.allclose(w_grad_gpu, model_cpu.weight.grad, atol=0.01))
    check("Linear bias grad matches CPU",
          torch.allclose(b_grad_gpu, model_cpu.bias.grad, atol=0.01))
except Exception as e:
    import traceback; traceback.print_exc()
    check(f"Linear backward ({e})", False)

# ── 5. TWO-LAYER NETWORK BACKWARD ────────────────────────────────────

print("── TWO-LAYER NETWORK BACKWARD ──")

try:
    torch.manual_seed(42)

    # GPU
    model_gpu = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1)
    ).to("verigpu")

    x_gpu = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32).to("verigpu")
    y_gpu = torch.tensor([[1.0], [1.0], [0.0]], dtype=torch.float32).to("verigpu")

    pred_gpu = model_gpu(x_gpu)
    diff_gpu = pred_gpu - y_gpu
    loss_gpu = (diff_gpu * diff_gpu).mean()
    loss_gpu.backward()

    grads_gpu = [p.grad.cpu().clone() for p in model_gpu.parameters()]

    # CPU reference (same init)
    torch.manual_seed(42)
    model_cpu = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
    x_cpu = x_gpu.detach().cpu()
    y_cpu = y_gpu.detach().cpu()
    pred_cpu = model_cpu(x_cpu)
    diff_cpu = pred_cpu - y_cpu
    loss_cpu = (diff_cpu * diff_cpu).mean()
    loss_cpu.backward()

    all_match = True
    for i, (g_gpu, p_cpu) in enumerate(zip(grads_gpu, model_cpu.parameters())):
        if not torch.allclose(g_gpu, p_cpu.grad, atol=0.05):
            print(f"  param {i} grad mismatch: max diff = {(g_gpu - p_cpu.grad).abs().max():.6f}")
            all_match = False

    check("2-layer network: all gradients match CPU", all_match)
    check("2-layer network: loss matches CPU",
          abs(loss_gpu.item() - loss_cpu.item()) < 0.01)
except Exception as e:
    import traceback; traceback.print_exc()
    check(f"2-layer backward ({e})", False)

# ── 6. SGD UPDATE STEP ───────────────────────────────────────────────

print("── SGD UPDATE STEP ──")

try:
    torch.manual_seed(42)

    # One full train step on GPU
    model_gpu = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1)).to("verigpu")
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32).to("verigpu")
    y = torch.tensor([[1.0], [0.0]], dtype=torch.float32).to("verigpu")

    pred = model_gpu(x)
    diff = pred - y
    loss = (diff * diff).mean()
    loss.backward()
    with torch.no_grad():
        for p in model_gpu.parameters():
            p.data = p.data - 0.1 * p.grad
    model_gpu.zero_grad()

    # Same on CPU
    torch.manual_seed(42)
    model_cpu = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
    x_cpu = x.detach().cpu()
    y_cpu = y.detach().cpu()
    pred = model_cpu(x_cpu)
    diff = pred - y_cpu
    loss = (diff * diff).mean()
    loss.backward()
    with torch.no_grad():
        for p in model_cpu.parameters():
            p.data -= 0.1 * p.grad
    model_cpu.zero_grad()

    # Compare weights after update
    all_match = True
    for p_gpu, p_cpu in zip(model_gpu.parameters(), model_cpu.parameters()):
        if not torch.allclose(p_gpu.cpu().detach(), p_cpu.detach(), atol=0.05):
            all_match = False
    check("weights match CPU after 1 SGD step", all_match)
except Exception as e:
    import traceback; traceback.print_exc()
    check(f"SGD step ({e})", False)

# ── Summary ───────────────────────────────────────────────────────────

print()
if failed == 0:
    print(f"========================================")
    print(f"  ALL {passed} TESTS PASSED")
    print(f"  Autograd works on RISC-V hardware!")
    print(f"========================================")
else:
    print(f"  {passed} passed, {failed} FAILED")
    sys.exit(1)
