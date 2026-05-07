#!/usr/bin/env python3
"""
train_mnist.py — CP-10: Train on MNIST subset using VeriGPU.

This is the final demo: a real ML task (handwritten digit classification)
running entirely on our custom GPU backend.

Architecture: Linear(784, 32) → ReLU → Linear(32, 10)
Loss: MSE with one-hot targets (we don't have cross-entropy registered)
Dataset: 100 training images (10 per class), 40 test images (4 per class)

Prerequisites:
    pip install torchvision

Run from repository root with venv activated:
    python3 pytorch_backend/train_mnist.py
"""

import torch
import torch.nn as nn
import sys
import types
import time

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

try:
    from torchvision import datasets, transforms
except ImportError:
    print("ERROR: torchvision not found. Install it:")
    print("  pip install torchvision --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

print(f"PyTorch {torch.__version__}")
print(f"========================================")
print(f"  CP-10: MNIST on VeriGPU")
print(f"  Final demo — real ML task")
print(f"========================================")
print()

# ── Load MNIST and create subset ──────────────────────────────────────

print("Loading MNIST dataset...")
mnist_train = datasets.MNIST(root='./data', train=True, download=True,
                              transform=transforms.ToTensor())
mnist_test = datasets.MNIST(root='./data', train=False, download=True,
                             transform=transforms.ToTensor())

# Extract subset: N samples per class
TRAIN_PER_CLASS = 100
TEST_PER_CLASS = 20
NUM_CLASSES = 10

def extract_subset(dataset, per_class):
    """Extract exactly per_class samples for each digit 0-9."""
    images, labels = [], []
    counts = [0] * NUM_CLASSES
    for img, label in dataset:
        if counts[label] < per_class:
            # Flatten 1x28x28 → 784 and normalize
            images.append(img.view(-1))
            labels.append(label)
            counts[label] += 1
        if all(c >= per_class for c in counts):
            break
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)

X_train_cpu, Y_train_labels = extract_subset(mnist_train, TRAIN_PER_CLASS)
X_test_cpu, Y_test_labels = extract_subset(mnist_test, TEST_PER_CLASS)

# One-hot encode targets (MSE loss needs same shape as output)
def one_hot(labels, num_classes):
    n = labels.shape[0]
    result = torch.zeros(n, num_classes, dtype=torch.float32)
    for i in range(n):
        result[i, labels[i]] = 1.0
    return result

Y_train_cpu = one_hot(Y_train_labels, NUM_CLASSES)
Y_test_cpu = one_hot(Y_test_labels, NUM_CLASSES)

print(f"  Train: {X_train_cpu.shape[0]} samples ({TRAIN_PER_CLASS} per class)")
print(f"  Test:  {X_test_cpu.shape[0]} samples ({TEST_PER_CLASS} per class)")
print(f"  Input:  {X_train_cpu.shape[1]} features (28x28 flattened)")
print(f"  Output: {NUM_CLASSES} classes")
print()

# Move to VeriGPU
X_train = X_train_cpu.to("verigpu")
Y_train = Y_train_cpu.to("verigpu")
X_test = X_test_cpu.to("verigpu")
Y_test = Y_test_cpu.to("verigpu")

# ── Model ─────────────────────────────────────────────────────────────

torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, NUM_CLASSES)
)
model = model.to("verigpu")

num_params = sum(p.numel() for p in model.parameters())
print(f"Model: Linear(784,64) → ReLU → Linear(64,10)")
print(f"  Parameters: {num_params}")
print()

# ── Helper: compute accuracy ──────────────────────────────────────────

def compute_accuracy(model, X, labels):
    """Compute classification accuracy (argmax on CPU)."""
    with torch.no_grad():
        pred = model(X).cpu()
    predicted_classes = pred.argmax(dim=1)
    correct = (predicted_classes == labels).sum().item()
    return correct / len(labels) * 100.0

# ── Training loop ─────────────────────────────────────────────────────

lr = 0.1
num_epochs = 200
print_every = 10

print(f"Training: {num_epochs} epochs, lr={lr}")
print(f"{'Epoch':>6s}  {'Loss':>10s}  {'Train%':>8s}  {'Test%':>8s}  {'Time':>8s}")
print(f"{'─'*6}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}")

losses = []
train_accs = []
test_accs = []
epoch_times = []

total_start = time.time()

for epoch in range(num_epochs):
    t0 = time.time()

    # Forward
    pred = model(X_train)

    # MSE Loss
    diff = pred - Y_train
    loss = (diff * diff).mean()

    loss_val = loss.item()
    losses.append(loss_val)

    # Backward
    loss.backward()

    # SGD update
    with torch.no_grad():
        for param in model.parameters():
            param.data = param.data - lr * param.grad

    model.zero_grad()

    t1 = time.time()
    epoch_times.append(t1 - t0)

    # Compute accuracy periodically
    if epoch % print_every == 0 or epoch == num_epochs - 1:
        train_acc = compute_accuracy(model, X_train, Y_train_labels)
        test_acc = compute_accuracy(model, X_test, Y_test_labels)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(f"{epoch:>6d}  {loss_val:>10.6f}  {train_acc:>7.1f}%  {test_acc:>7.1f}%  {(t1-t0)*1000:>6.1f}ms")

total_time = time.time() - total_start
avg_epoch_ms = sum(epoch_times) / len(epoch_times) * 1000

print()
print(f"Training complete in {total_time:.2f}s ({avg_epoch_ms:.1f}ms/epoch)")
print()

# ── Final evaluation ──────────────────────────────────────────────────

final_train_acc = compute_accuracy(model, X_train, Y_train_labels)
final_test_acc = compute_accuracy(model, X_test, Y_test_labels)

print(f"Final accuracy:")
print(f"  Train: {final_train_acc:.1f}%")
print(f"  Test:  {final_test_acc:.1f}%")
print()

# Per-class breakdown
with torch.no_grad():
    test_pred = model(X_test).cpu().argmax(dim=1)

print(f"Per-class test results:")
for digit in range(NUM_CLASSES):
    mask = Y_test_labels == digit
    if mask.sum() > 0:
        digit_correct = (test_pred[mask] == digit).sum().item()
        digit_total = mask.sum().item()
        print(f"  Digit {digit}: {digit_correct}/{digit_total} correct")

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

# Test 2: Significant reduction
check(f"loss reduced >50% ({losses[0]:.4f} -> {losses[-1]:.6f})",
      losses[-1] < 0.5 * losses[0])

# Test 3: Training accuracy > 80%
check(f"train accuracy > 80% (got {final_train_acc:.1f}%)",
      final_train_acc > 80.0)

# Test 4: Test accuracy > 50% (reasonable for 100 training samples)
check(f"test accuracy > 50% (got {final_test_acc:.1f}%)",
      final_test_acc > 50.0)

# Test 5: All parameters stayed on device
all_on_device = all(p.device.type == "verigpu" for p in model.parameters())
check("all parameters on verigpu", all_on_device)

# Test 6: Model has correct architecture
params = list(model.parameters())
check("layer 1 weight shape [64,784]", params[0].shape == (64, 784))
check("layer 1 bias shape [64]", params[1].shape == (64,))
check("layer 2 weight shape [10,64]", params[2].shape == (10, 64))
check("layer 2 bias shape [10]", params[3].shape == (10,))

# Test 7: Gradients work through the full model
pred_chk = model(X_train)
diff_chk = pred_chk - Y_train
loss_chk = (diff_chk * diff_chk).mean()
loss_chk.backward()
check("gradients exist after backward",
      all(p.grad is not None for p in model.parameters()))
model.zero_grad()

# Test 8: Compare with CPU training
torch.manual_seed(42)
model_cpu = nn.Sequential(nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, NUM_CLASSES))
for ep in range(num_epochs):
    p = model_cpu(X_train_cpu)
    d = p - Y_train_cpu
    l = (d * d).mean()
    l.backward()
    with torch.no_grad():
        for param in model_cpu.parameters():
            param.data -= lr * param.grad
    model_cpu.zero_grad()

cpu_pred = model_cpu(X_test_cpu).detach().argmax(dim=1)
gpu_pred_final = model(X_test).cpu().detach().argmax(dim=1)
check("VeriGPU predictions match CPU",
      torch.equal(gpu_pred_final, cpu_pred))

all_match = all(
    torch.allclose(pg.cpu().detach(), pc.detach(), atol=1e-4)
    for pg, pc in zip(model.parameters(), model_cpu.parameters())
)
check("all parameters match CPU training", all_match)

# ── Summary ────────────────────────────────────────────────────────────

print()
print(f"{'='*50}")
print(f"  Performance summary:")
print(f"    Epochs:          {num_epochs}")
print(f"    Avg epoch time:  {avg_epoch_ms:.1f} ms")
print(f"    Total time:      {total_time:.2f} s")
print(f"    Train accuracy:  {final_train_acc:.1f}%")
print(f"    Test accuracy:   {final_test_acc:.1f}%")
print(f"    Parameters:      {num_params}")
print(f"{'='*50}")

print()
if failed == 0:
    print(f"========================================")
    print(f"  ALL {passed} TESTS PASSED")
    print(f"  MNIST trained on VeriGPU!")
    print(f"========================================")
else:
    print(f"  {passed} passed, {failed} FAILED")
    sys.exit(1)
