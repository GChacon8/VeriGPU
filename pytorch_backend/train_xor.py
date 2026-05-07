#!/usr/bin/env python3
"""
train_xor.py — CP-9: Train a 2-layer network on XOR using nn.Linear + nn.ReLU.

XOR is NOT linearly separable — it requires a hidden layer with non-linear
activation. This validates that nn.Linear, nn.ReLU, and autograd through
multiple layers all work correctly on VeriGPU.

Architecture:
    nn.Linear(2, 8) → nn.ReLU() → nn.Linear(8, 1)

    Input [4,2] → hidden [4,8] → relu [4,8] → output [4,1]

This uses standard PyTorch modules (not manual mm/add like CP-8).
nn.Linear internally calls F.linear → addmm, and nn.ReLU calls relu.

Run from repository root with venv activated:
    python3 pytorch_backend/train_xor.py
"""

import torch
import torch.nn as nn
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
print(f"  CP-9: Training XOR on VeriGPU")
print(f"  Model: nn.Linear(2,8) + ReLU + nn.Linear(8,1)")
print(f"========================================")
print()

# ── Dataset: XOR function ─────────────────────────────────────────────

X_data = [[0.0, 0.0],
          [0.0, 1.0],
          [1.0, 0.0],
          [1.0, 1.0]]

Y_data = [[0.0],
          [1.0],
          [1.0],
          [0.0]]

X = torch.tensor(X_data, dtype=torch.float32).to("verigpu")
Y = torch.tensor(Y_data, dtype=torch.float32).to("verigpu")

print(f"Dataset: XOR function")
print(f"  [0,0]→0  [0,1]→1  [1,0]→1  [1,1]→0")
print()

# ── Model ─────────────────────────────────────────────────────────────

torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

# Move to VeriGPU
model = model.to("verigpu")

# Verify model is on device
param_devices = [p.device.type for p in model.parameters()]
print(f"Model on device: {param_devices}")
num_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {num_params}")
print()

# ── Training loop ─────────────────────────────────────────────────────

lr = 0.5
num_epochs = 500
print_every = 50

print(f"Training: {num_epochs} epochs, lr={lr}")
print(f"{'Epoch':>6s}  {'Loss':>10s}  {'Pred':>35s}")
print(f"{'─'*6}  {'─'*10}  {'─'*35}")

losses = []

for epoch in range(num_epochs):
    # Forward
    pred = model(X)

    # Loss (MSE)
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

    # Manual SGD
    with torch.no_grad():
        for param in model.parameters():
            param.data = param.data - lr * param.grad

    # Zero gradients
    model.zero_grad()

print()

# ── Final results ─────────────────────────────────────────────────────

with torch.no_grad():
    final_pred = model(X).cpu().flatten()

print(f"Predictions vs targets:")
for x, y, p in zip(X_data, Y_data, final_pred.tolist()):
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
check("loss decreased", losses[-1] < losses[0])

# Test 2: Loss decreased significantly
check(f"loss reduced >90% ({losses[0]:.4f} -> {losses[-1]:.6f})",
      losses[-1] < 0.1 * losses[0])

# Test 3: Final loss is small
check(f"final loss < 0.05 (got {losses[-1]:.6f})",
      losses[-1] < 0.05)

# Test 4: All XOR predictions correct
binary_preds = [1 if p > 0.5 else 0 for p in final_pred.tolist()]
expected = [0, 1, 1, 0]
check(f"all XOR predictions correct {binary_preds} == {expected}",
      binary_preds == expected)

# Test 5: Model stayed on device
all_on_device = all(p.device.type == "verigpu" for p in model.parameters())
check("all parameters on verigpu", all_on_device)

# Test 6: Gradients flow through both layers
pred_check = model(X)
diff_check = pred_check - Y
loss_check = (diff_check * diff_check).mean()
loss_check.backward()

layer1_grad = list(model.parameters())[0].grad
layer2_grad = list(model.parameters())[2].grad
check("layer 1 grad exists",
      layer1_grad is not None)
check("layer 2 grad exists",
      layer2_grad is not None)

# Test 7: nn.Linear used correctly (check shapes)
check("layer 1 weight shape [8,2]", list(model.parameters())[0].shape == (8, 2))
check("layer 1 bias shape [8]", list(model.parameters())[1].shape == (8,))
check("layer 2 weight shape [1,8]", list(model.parameters())[2].shape == (1, 8))
check("layer 2 bias shape [1]", list(model.parameters())[3].shape == (1,))

# Test 8: Compare with CPU training
torch.manual_seed(42)
model_cpu = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
X_cpu = torch.tensor(X_data, dtype=torch.float32)
Y_cpu = torch.tensor(Y_data, dtype=torch.float32)

for _ in range(num_epochs):
    p = model_cpu(X_cpu)
    d = p - Y_cpu
    l = (d * d).mean()
    l.backward()
    with torch.no_grad():
        for param in model_cpu.parameters():
            param.data -= lr * param.grad
    model_cpu.zero_grad()

cpu_pred = model_cpu(X_cpu).detach().flatten()
gpu_pred = final_pred

check("VeriGPU predictions match CPU",
      torch.allclose(gpu_pred, cpu_pred, atol=1e-4))

# Compare all parameters
all_params_match = True
for p_gpu, p_cpu in zip(model.parameters(), model_cpu.parameters()):
    if not torch.allclose(p_gpu.cpu().detach(), p_cpu.detach(), atol=1e-4):
        all_params_match = False
        break
check("all parameters match CPU training", all_params_match)

# ── Summary ────────────────────────────────────────────────────────────

print()
if failed == 0:
    print(f"========================================")
    print(f"  ALL {passed} TESTS PASSED")
    print(f"  XOR learned with nn.Linear + ReLU!")
    print(f"========================================")
else:
    print(f"  {passed} passed, {failed} FAILED")
    sys.exit(1)
