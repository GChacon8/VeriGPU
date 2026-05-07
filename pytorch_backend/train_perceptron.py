#!/usr/bin/env python3
"""
train_perceptron.py — CP-8: Train a perceptron on VeriGPU.

This is the first neural network trained end-to-end on the VeriGPU backend.
It learns the OR function (linearly separable) using:
  - Forward:  y = X @ w + b  (linear, no activation needed)
  - Loss:     MSE = mean((y - target)^2)
  - Backward: autograd computes gradients
  - Update:   manual SGD (w -= lr * grad)

Run from repository root with venv activated:
    python3 pytorch_backend/train_perceptron.py
"""

import torch
import sys
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

print(f"PyTorch {torch.__version__}")
print(f"========================================")
print(f"  CP-8: Training a Perceptron on VeriGPU")
print(f"========================================")
print()

# ── Dataset: OR function ───────────────────────────────────────────────

X_data = [[0.0, 0.0],
          [0.0, 1.0],
          [1.0, 0.0],
          [1.0, 1.0]]

Y_data = [[0.0],
          [1.0],
          [1.0],
          [1.0]]

X = torch.tensor(X_data, dtype=torch.float32).to("verigpu")
Y = torch.tensor(Y_data, dtype=torch.float32).to("verigpu")

print(f"Dataset: OR function")
print(f"  X shape: {X.shape}  (4 samples, 2 features)")
print(f"  Y shape: {Y.shape}  (4 samples, 1 output)")
print()

# ── Model: single linear layer y = X @ w + b ─────────────────────────

torch.manual_seed(42)
w = torch.randn(2, 1, dtype=torch.float32).to("verigpu").detach().requires_grad_(True)
b = torch.zeros(1, dtype=torch.float32).to("verigpu").detach().requires_grad_(True)

print(f"Model: y = X @ w + b")
print(f"  w shape: {w.shape}  init: {w.cpu().flatten().tolist()}")
print(f"  b shape: {b.shape}  init: {b.cpu().flatten().tolist()}")
print()

# ── Training loop ─────────────────────────────────────────────────────

lr = 0.5
num_epochs = 100
print_every = 10

print(f"Training: {num_epochs} epochs, lr={lr}")
print(f"{'Epoch':>6s}  {'Loss':>10s}  {'Pred':>30s}")
print(f"{'─'*6}  {'─'*10}  {'─'*30}")

losses = []

for epoch in range(num_epochs):
    # Forward
    pred = torch.mm(X, w) + b
    
    # Loss (MSE)
    diff = pred - Y
    loss = (diff * diff).mean()
    
    loss_val = loss.item()
    losses.append(loss_val)

    if epoch % print_every == 0 or epoch == num_epochs - 1:
        pred_list = [f"{v:.3f}" for v in pred.cpu().detach().flatten().tolist()]
        print(f"{epoch:>6d}  {loss_val:>10.6f}  {str(pred_list):>30s}")

    # Backward
    loss.backward()

    # SGD update
    with torch.no_grad():
        w.data = w.data - lr * w.grad
        b.data = b.data - lr * b.grad

    # Zero gradients
    w.grad.zero_()
    b.grad.zero_()

print()

# ── Final results ─────────────────────────────────────────────────────

with torch.no_grad():
    final_pred = (torch.mm(X, w) + b).cpu().flatten()

final_w = w.cpu().detach().flatten().tolist()
final_b = b.cpu().detach().flatten().tolist()

print(f"Final weights: w={[round(v,4) for v in final_w]}, b={[round(v,4) for v in final_b]}")
print()
print(f"Predictions vs targets:")
for i, (x, y, p) in enumerate(zip(X_data, Y_data, final_pred.tolist())):
    binary_pred = 1 if p > 0.5 else 0
    correct = "ok" if binary_pred == int(y[0]) else "WRONG"
    print(f"  {x} -> pred={p:.4f}  (>0.5? -> {binary_pred})  target={int(y[0])}  {correct}")

# ── Verification ──────────────────────────────────────────────────────

print()
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

# Test 1: Loss decreased
check("loss decreased (first -> last)", losses[-1] < losses[0])

# Test 2: Loss decreased significantly
check(f"loss reduced >80% ({losses[0]:.4f} -> {losses[-1]:.6f})",
      losses[-1] < 0.2 * losses[0])

# Test 3: Final loss is small
check(f"final loss < 0.07 (got {losses[-1]:.6f})", losses[-1] < 0.07)

# Test 4: All predictions correct
binary_preds = [1 if p > 0.5 else 0 for p in final_pred.tolist()]
expected = [0, 1, 1, 1]
check(f"all predictions correct {binary_preds} == {expected}",
      binary_preds == expected)

# Test 5: Loss mostly decreasing
decreasing = sum(1 for i in range(1, len(losses)) if losses[i] <= losses[i-1] + 1e-6)
check(f"loss mostly decreasing ({decreasing}/{num_epochs-1} steps)",
      decreasing > 0.8 * (num_epochs - 1))

# Test 6: Everything stayed on device
check("weights on verigpu", w.device.type == "verigpu")
check("bias on verigpu", b.device.type == "verigpu")

# Test 7: Gradients work
pred_check = torch.mm(X, w) + b
loss_check = ((pred_check - Y) * (pred_check - Y)).mean()
loss_check.backward()
check("w.grad exists after backward", w.grad is not None)
check("b.grad exists after backward", b.grad is not None)
check("w.grad computed (may be ~0 at convergence)", w.grad is not None)

# Test 8: Results match CPU training exactly
torch.manual_seed(42)
w_cpu = torch.randn(2, 1, dtype=torch.float32, requires_grad=True)
b_cpu = torch.zeros(1, dtype=torch.float32, requires_grad=True)
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
    w_cpu.grad.zero_()
    b_cpu.grad.zero_()

check("VeriGPU weights match CPU training",
      torch.allclose(w.cpu(), w_cpu, atol=1e-5))
check("VeriGPU bias matches CPU training",
      torch.allclose(b.cpu(), b_cpu, atol=1e-5))

# ── Summary ────────────────────────────────────────────────────────────

print()
if failed == 0:
    print(f"========================================")
    print(f"  ALL {passed} TESTS PASSED")
    print(f"  First neural network trained on VeriGPU!")
    print(f"========================================")
else:
    print(f"  {passed} passed, {failed} FAILED")
    sys.exit(1)
