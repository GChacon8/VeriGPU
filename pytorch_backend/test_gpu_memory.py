#!/usr/bin/env python3
"""
test_gpu_memory.py — CP-5 (HW Roadmap): Verify dual memory allocation.

Tests that in HW mode, tensor data exists in BOTH:
  - Host RAM (for PyTorch operations)
  - GPU simulated memory (for future kernel execution)

Run from repository root:
    # Host mode (regression):
    python3 pytorch_backend/test_gpu_memory.py

    # HW mode (new tests):
    VERIGPU_USE_HW=1 python3 pytorch_backend/test_gpu_memory.py
"""

import torch
import struct
import sys
import os
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

hw_mode = _verigpu_C.is_hw_mode()

print(f"PyTorch {torch.__version__}")
print(f"========================================")
print(f"  CP-5 (HW Roadmap): GPU Memory Test")
print(f"  Mode: {'HARDWARE' if hw_mode else 'HOST CPU'}")
print(f"========================================")
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

# ── Host-mode regression (always runs) ────────────────────────────────

print("── REGRESSION (host ops) ──")

try:
    t = torch.tensor([1.0, 2.0, 3.0]).to("verigpu")
    check("round-trip", torch.equal(t.cpu(), torch.tensor([1.0, 2.0, 3.0])))
except Exception as e:
    check(f"round-trip (exception: {e})", False)

try:
    a = torch.tensor([1.0, 2.0]).to("verigpu")
    b = torch.tensor([3.0, 4.0]).to("verigpu")
    c = (a + b).cpu()
    check("add", torch.equal(c, torch.tensor([4.0, 6.0])))
except Exception as e:
    check(f"add (exception: {e})", False)

try:
    check("zeros", torch.equal(torch.zeros(3, device="verigpu").cpu(), torch.zeros(3)))
    check("ones", torch.equal(torch.ones(3, device="verigpu").cpu(), torch.ones(3)))
except Exception as e:
    check(f"zeros/ones (exception: {e})", False)

try:
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to("verigpu")
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]]).to("verigpu")
    c = torch.mm(a, b).cpu()
    check("mm", torch.equal(c, torch.tensor([[19.0, 22.0], [43.0, 50.0]])))
except Exception as e:
    check(f"mm (exception: {e})", False)

# ── HW-mode specific tests ───────────────────────────────────────────

if hw_mode:
    print()
    print("── GPU MEMORY ALLOCATION ──")

    # Test: tensors have GPU addresses
    try:
        t = torch.tensor([1.0, 2.0, 3.0, 4.0]).to("verigpu")
        addr = _verigpu_C.gpu_addr_of(t)
        check(f"tensor has GPU address (addr={addr})", addr > 0)
    except Exception as e:
        check(f"GPU address (exception: {e})", False)

    # Test: multiple tensors have different GPU addresses
    try:
        a = torch.tensor([1.0, 2.0]).to("verigpu")
        b = torch.tensor([3.0, 4.0]).to("verigpu")
        addr_a = _verigpu_C.gpu_addr_of(a)
        addr_b = _verigpu_C.gpu_addr_of(b)
        check(f"different GPU addrs (a={addr_a}, b={addr_b})",
              addr_a > 0 and addr_b > 0 and addr_a != addr_b)
    except Exception as e:
        check(f"different addrs (exception: {e})", False)

    # Test: zeros have GPU address
    try:
        z = torch.zeros(4, device="verigpu")
        addr = _verigpu_C.gpu_addr_of(z)
        check(f"zeros has GPU address (addr={addr})", addr > 0)
    except Exception as e:
        check(f"zeros GPU addr (exception: {e})", False)

    print()
    print("── GPU MEMORY CONTENT ──")

    # Test: data actually made it to GPU memory
    try:
        t = torch.tensor([1.0, 2.0, 3.0, 4.0]).to("verigpu")
        addr = _verigpu_C.gpu_addr_of(t)
        gpu_floats = _verigpu_C.gpu_readback_floats(addr, 4)
        check("GPU memory matches host [1,2,3,4]",
              all(abs(a - b) < 1e-6 for a, b in
                  zip(gpu_floats, [1.0, 2.0, 3.0, 4.0])))
    except Exception as e:
        check(f"GPU readback (exception: {e})", False)

    # Test: ones in GPU memory
    try:
        t = torch.ones(3, device="verigpu")
        addr = _verigpu_C.gpu_addr_of(t)
        gpu_floats = _verigpu_C.gpu_readback_floats(addr, 3)
        check("GPU memory has ones [1,1,1]",
              all(abs(v - 1.0) < 1e-6 for v in gpu_floats))
    except Exception as e:
        check(f"ones GPU readback (exception: {e})", False)

    # Test: zeros in GPU memory
    try:
        t = torch.zeros(4, device="verigpu")
        addr = _verigpu_C.gpu_addr_of(t)
        gpu_floats = _verigpu_C.gpu_readback_floats(addr, 4)
        check("GPU memory has zeros [0,0,0,0]",
              all(abs(v) < 1e-6 for v in gpu_floats))
    except Exception as e:
        check(f"zeros GPU readback (exception: {e})", False)

    # Test: larger tensor
    try:
        data = list(range(1, 17))  # [1.0 .. 16.0]
        t = torch.tensor(data, dtype=torch.float32).to("verigpu")
        addr = _verigpu_C.gpu_addr_of(t)
        gpu_floats = _verigpu_C.gpu_readback_floats(addr, 16)
        match = all(abs(a - b) < 1e-6 for a, b in zip(gpu_floats, data))
        check("16-element tensor in GPU memory", match)
    except Exception as e:
        check(f"large tensor GPU readback (exception: {e})", False)

    # Test: fill_ syncs to GPU
    try:
        t = torch.empty(4, device="verigpu")
        t.fill_(42.0)
        addr = _verigpu_C.gpu_addr_of(t)
        gpu_floats = _verigpu_C.gpu_readback_floats(addr, 4)
        check("fill_(42) synced to GPU",
              all(abs(v - 42.0) < 1e-6 for v in gpu_floats))
    except Exception as e:
        check(f"fill_ sync (exception: {e})", False)

else:
    print()
    print("── SKIPPING HW TESTS (host mode) ──")
    print("  Run with VERIGPU_USE_HW=1 to test GPU memory")

# ── Summary ────────────────────────────────────────────────────────────

print()
mode_str = "HARDWARE (Verilator)" if hw_mode else "HOST CPU"
if failed == 0:
    print(f"========================================")
    print(f"  ALL {passed} TESTS PASSED")
    print(f"  Mode: {mode_str}")
    print(f"========================================")
else:
    print(f"  {passed} passed, {failed} FAILED")
    print(f"  Mode: {mode_str}")
    sys.exit(1)
