#!/usr/bin/env python3
"""
train_hw_perceptron.py — CP-12 (HW Roadmap): Train perceptron on RISC-V.

Run from repository root:
    VERIGPU_USE_HW=1 python3 pytorch_backend/train_hw_perceptron.py
"""

import torch
import subprocess
import sys
import os
import time
import types

# ── Load kernels ──────────────────────────────────────────────────────

print("=== CP-12 (HW Roadmap): Perceptron on RISC-V Hardware ===")
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

os.makedirs("build", exist_ok=True)
for name, asm_path in kernels.items():
    hex_path = f"build/{os.path.basename(asm_path).replace('.asm', '.hex')}"
    subprocess.run(
        ["python3", "verigpu/assembler.py", "--in-asm", asm_path, "--out-hex", hex_path],
        capture_output=True, text=True, check=True)

import _verigpu_C

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

print("Kernels loaded. Training on RISC-V cores.")
print()

# ── Dataset: OR gate ──────────────────────────────────────────────────

X_data = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
Y_data = [[0.0], [1.0], [1.0], [1.0]]

X = torch.tensor(X_data, dtype=torch.float32).to("verigpu")
Y = torch.tensor(Y_data, dtype=torch.float32).to("verigpu")

print(f"Dataset: OR gate (4 samples)")

# ── Model ─────────────────────────────────────────────────────────────

torch.manual_seed(42)
w = torch.randn(2, 1, dtype=torch.float32).to("verigpu").detach().requires_grad_(True)
b = torch.zeros(1, dtype=torch.float32).to("verigpu").detach().requires_grad_(True)

print(f"Model: y = X @ w + b")
print(f"  w init: {[round(v, 4) for v in w.cpu().flatten().tolist()]}")
print()

# ── Training ──────────────────────────────────────────────────────────

lr = 0.1
num_epochs = 100
print_every = 10

print(f"Training: {num_epochs} epochs, lr={lr}")
print()
print(f"{'Epoch':>6s}  {'Loss':>10s}  {'Pred':>35s}")
print(f"{'─'*6}  {'─'*10}  {'─'*35}")

losses = []
total_start = time.time()

for epoch in range(num_epochs):
    # Forward: pred = X @ w + b  (mm + add kernels on RISC-V)
    pred = torch.mm(X, w) + b

    # Loss: MSE
    diff = pred - Y
    loss = (diff * diff).mean()

    loss_val = loss.item()
    losses.append(loss_val)

    if epoch % print_every == 0 or epoch == num_epochs - 1:
        with torch.no_grad():
            pred_list = [f"{v:.3f}" for v in pred.cpu().flatten().tolist()]
        print(f"{epoch:>6d}  {loss_val:>10.6f}  {str(pred_list):>35s}")

    # Backward
    loss.backward()

    # SGD update on CPU (avoids scalar*tensor sync issues on HW)
    with torch.no_grad():
        w_new = w.data.cpu() - lr * w.grad.cpu()
        b_new = b.data.cpu() - lr * b.grad.cpu()
        w.data.copy_(w_new.to("verigpu"))
        b.data.copy_(b_new.to("verigpu"))

    w.grad = None
    b.grad = None

total_time = time.time() - total_start
print()
print(f"Training complete in {total_time:.1f}s ({total_time/num_epochs:.2f}s/epoch)")
print()

# ── Results ───────────────────────────────────────────────────────────

w_final = w.cpu().detach().flatten().tolist()
b_final = b.cpu().detach().item()

print(f"Final weights: w={[round(v,4) for v in w_final]}, b={round(b_final,4)}")
print()

with torch.no_grad():
    final_pred = (torch.mm(X, w) + b).cpu().flatten()

print(f"Predictions:")
for x, y, p in zip(X_data, Y_data, final_pred.tolist()):
    binary = 1 if p > 0.5 else 0
    correct = "ok" if binary == int(y[0]) else "WRONG"
    print(f"  {x} -> {p:.4f} (>0.5? {binary})  target={int(y[0])}  {correct}")
print()

# ── Verification ──────────────────────────────────────────────────────

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

check("loss decreased", losses[-1] < losses[0])
check(f"loss reduced >80% ({losses[0]:.4f} -> {losses[-1]:.6f})",
      losses[-1] < 0.2 * losses[0])
check(f"final loss < 0.07 (got {losses[-1]:.6f})", losses[-1] < 0.07)

binary_preds = [1 if p > 0.5 else 0 for p in final_pred.tolist()]
check(f"all OR predictions correct {binary_preds}", binary_preds == [0, 1, 1, 1])

check("w on verigpu", w.device.type == "verigpu")
check("b on verigpu", b.device.type == "verigpu")

# CPU comparison (same lr and epochs)
torch.manual_seed(42)
w_cpu = torch.randn(2, 1, dtype=torch.float32).requires_grad_(True)
b_cpu = torch.zeros(1, dtype=torch.float32).requires_grad_(True)
X_cpu = torch.tensor(X_data, dtype=torch.float32)
Y_cpu = torch.tensor(Y_data, dtype=torch.float32)

for _ in range(num_epochs):
    p = torch.mm(X_cpu, w_cpu) + b_cpu
    d = p - Y_cpu
    l = (d * d).mean()
    l.backward()
    with torch.no_grad():
        w_cpu.data -= lr * w_cpu.grad
        b_cpu.data -= lr * b_cpu.grad
    w_cpu.grad = None
    b_cpu.grad = None

w_match = torch.allclose(torch.tensor(w_final), w_cpu.detach().flatten(), atol=0.1)
b_match = abs(b_final - b_cpu.detach().item()) < 0.1
check("weights close to CPU training", w_match)
check("bias close to CPU training", b_match)

print()
if failed == 0:
    print(f"========================================")
    print(f"  ALL {passed} TESTS PASSED")
    print(f"  Perceptron trained on RISC-V hardware!")
    print(f"  {total_time:.1f}s total ({num_epochs} epochs)")
    print(f"========================================")
else:
    print(f"  {passed} passed, {failed} FAILED")
    sys.exit(1)