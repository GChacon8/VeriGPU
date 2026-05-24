#!/usr/bin/env python3
"""
train_hw_mnist.py — CP-13 (HW Roadmap): MNIST on RISC-V Hardware.

The final benchmark: train a neural network on real handwritten digit
images with every arithmetic operation executing on RISC-V cores.

Architecture: Linear(784, 16) → ReLU → Linear(16, 10)
Loss: MSE with one-hot targets
Dataset: 10 training samples (1 per digit), 10 test samples
Optimizer: SGD (lr=0.1), update on CPU

Each epoch runs the full pipeline on hardware:
  Forward:  2x mm kernel + relu kernel + add kernels
  Loss:     sub + mul + sum kernels
  Backward: 2x mm + threshold_backward + mul + sub kernels

Expected timing: ~2-10 minutes per epoch (Verilator simulation).

Run from repository root:
    VERIGPU_USE_HW=1 python3 pytorch_backend/train_hw_mnist.py

Prerequisites:
    pip install torchvision --index-url https://download.pytorch.org/whl/cpu
"""

import torch
import torch.nn as nn
import subprocess
import sys
import os
import time
import types

# ── Load kernels ──────────────────────────────────────────────────────

print("=== CP-13 (HW Roadmap): MNIST on RISC-V Hardware ===")
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

print("Kernels loaded.")
print()

# ── Load MNIST ────────────────────────────────────────────────────────

try:
    from torchvision import datasets, transforms
except ImportError:
    print("ERROR: torchvision not found. Install:")
    print("  pip install torchvision --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

print("Loading MNIST dataset...")
mnist_train = datasets.MNIST(root='./data', train=True, download=True,
                              transform=transforms.ToTensor())
mnist_test = datasets.MNIST(root='./data', train=False, download=True,
                             transform=transforms.ToTensor())

TRAIN_PER_CLASS = 15   # 1 sample per digit = 10 total
TEST_PER_CLASS = 5    # 1 sample per digit = 10 total
NUM_CLASSES = 10

def extract_subset(dataset, per_class):
    counts = [0] * NUM_CLASSES
    images, labels = [], []
    for img, label in dataset:
        if counts[label] < per_class:
            images.append(img.view(-1))  # Flatten 28x28 → 784
            labels.append(label)
            counts[label] += 1
        if all(c >= per_class for c in counts):
            break
    return torch.stack(images), torch.tensor(labels)

X_train, Y_train_idx = extract_subset(mnist_train, TRAIN_PER_CLASS)
X_test, Y_test_idx = extract_subset(mnist_test, TEST_PER_CLASS)

# One-hot encode targets (MSE loss, no cross-entropy)
def one_hot(labels, n_classes):
    oh = torch.zeros(len(labels), n_classes)
    for i, l in enumerate(labels):
        oh[i, l] = 1.0
    return oh

Y_train = one_hot(Y_train_idx, NUM_CLASSES)
Y_test = one_hot(Y_test_idx, NUM_CLASSES)

n_train = X_train.shape[0]
n_test = X_test.shape[0]
print(f"  Train: {n_train} samples ({TRAIN_PER_CLASS} per class)")
print(f"  Test:  {n_test} samples ({TEST_PER_CLASS} per class)")
print(f"  Input: {X_train.shape[1]} features (28x28 flattened)")
print()

# Move to device
X_train_gpu = X_train.to("verigpu")
Y_train_gpu = Y_train.to("verigpu")
X_test_gpu = X_test.to("verigpu")

# ── Model ─────────────────────────────────────────────────────────────

HIDDEN = 16

torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(784, HIDDEN),
    nn.ReLU(),
    nn.Linear(HIDDEN, NUM_CLASSES)
).to("verigpu")

n_params = sum(p.numel() for p in model.parameters())
print(f"Model: Linear(784,{HIDDEN}) → ReLU → Linear({HIDDEN},{NUM_CLASSES})")
print(f"  Parameters: {n_params}")
print()

# ── Training ──────────────────────────────────────────────────────────

lr = 0.2
num_epochs = 8
print(f"Training: {num_epochs} epochs, lr={lr}")
print(f"  NOTE: Each epoch may take several minutes (Verilator simulation)")
print()
print(f"{'Epoch':>6s}  {'Loss':>10s}  {'Train%':>8s}  {'Time':>10s}")
print(f"{'─'*6}  {'─'*10}  {'─'*8}  {'─'*10}")

losses = []
total_start = time.time()

for epoch in range(num_epochs):
    t0 = time.time()

    # Forward
    pred = model(X_train_gpu)

    # Loss: MSE
    diff = pred - Y_train_gpu
    loss = (diff * diff).mean()

    loss_val = loss.item()
    losses.append(loss_val)

    # Accuracy
    with torch.no_grad():
        pred_class = pred.cpu().argmax(dim=1)
        train_acc = (pred_class == Y_train_idx).float().mean().item() * 100

    elapsed = time.time() - t0
    print(f"{epoch:>6d}  {loss_val:>10.6f}  {train_acc:>6.1f}%  {elapsed:>8.1f}s")
    sys.stdout.flush()

    # Backward
    loss.backward()

    # SGD update on CPU
    with torch.no_grad():
        for p in model.parameters():
            p_new = p.data.cpu() - lr * p.grad.cpu()
            p.data.copy_(p_new.to("verigpu"))

    model.zero_grad()

total_time = time.time() - total_start
print()
print(f"Training complete in {total_time:.1f}s ({total_time/num_epochs:.1f}s/epoch)")
print()

# ── Results ───────────────────────────────────────────────────────────

with torch.no_grad():
    train_pred = model(X_train_gpu).cpu()
    test_pred = model(X_test_gpu).cpu()

train_class = train_pred.argmax(dim=1)
test_class = test_pred.argmax(dim=1)
train_acc = (train_class == Y_train_idx).float().mean().item() * 100
test_acc = (test_class == Y_test_idx).float().mean().item() * 100

print(f"Final accuracy:")
print(f"  Train: {train_acc:.1f}% ({int(train_acc*n_train/100)}/{n_train})")
print(f"  Test:  {test_acc:.1f}% ({int(test_acc*n_test/100)}/{n_test})")
print()

print("Test predictions:")
for i in range(n_test):
    correct = "ok" if test_class[i] == Y_test_idx[i] else "WRONG"
    print(f"  Digit {Y_test_idx[i].item()}: predicted {test_class[i].item()} {correct}")
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

# Training metrics
check("loss decreased", losses[-1] < losses[0])
check(f"loss reduced >30% ({losses[0]:.4f} -> {losses[-1]:.4f})",
      losses[-1] < 0.7 * losses[0])

# Architecture
check("layer 1 weight shape [16,784]",
      list(model[0].weight.shape) == [HIDDEN, 784])
check("layer 1 bias shape [16]",
      list(model[0].bias.shape) == [HIDDEN])
check("layer 2 weight shape [10,16]",
      list(model[2].weight.shape) == [NUM_CLASSES, HIDDEN])
check("layer 2 bias shape [10]",
      list(model[2].bias.shape) == [NUM_CLASSES])

# Device
all_on_device = all(p.device.type == "verigpu" for p in model.parameters())
check("all parameters on verigpu", all_on_device)

# CPU comparison
torch.manual_seed(42)
model_cpu = nn.Sequential(
    nn.Linear(784, HIDDEN),
    nn.ReLU(),
    nn.Linear(HIDDEN, NUM_CLASSES)
)

for epoch in range(num_epochs):
    p = model_cpu(X_train)
    d = p - Y_train
    l = (d * d).mean()
    l.backward()
    with torch.no_grad():
        for param in model_cpu.parameters():
            param.data -= lr * param.grad
    model_cpu.zero_grad()

cpu_pred = model_cpu(X_train).detach()
gpu_pred = train_pred

pred_match = torch.allclose(gpu_pred, cpu_pred, atol=0.5)
check("predictions close to CPU", pred_match)
if not pred_match:
    diff = (gpu_pred - cpu_pred).abs().max().item()
    print(f"  max prediction diff: {diff:.6f}")

# ── Summary ───────────────────────────────────────────────────────────

print()
if failed == 0:
    print(f"========================================")
    print(f"  ALL {passed} TESTS PASSED")
    print(f"  MNIST on RISC-V hardware!")
    print(f"  {total_time:.1f}s total ({num_epochs} epochs)")
    print(f"========================================")
else:
    print(f"  {passed} passed, {failed} FAILED")
    sys.exit(1)
