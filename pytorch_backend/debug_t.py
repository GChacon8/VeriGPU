#!/usr/bin/env python3
import torch, sys, types
try:
    import _verigpu_C
except ImportError:
    sys.exit(1)
_mod = types.ModuleType("torch.verigpu")
_mod.__path__ = []
sys.modules["torch.verigpu"] = _mod
torch.utils.rename_privateuse1_backend("verigpu")

print("1. Creating tensor...")
a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to("verigpu")
print(f"   a device={a.device}, shape={a.shape}, strides={a.stride()}, contig={a.is_contiguous()}")

print("2. Calling t()...")
at = a.t()
print(f"   at device={at.device}, shape={at.shape}, strides={at.stride()}, contig={at.is_contiguous()}")

print("3. Checking nbytes and data_ptr...")
print(f"   at.nbytes={at.nbytes()}, at.storage_offset()={at.storage_offset()}")
print(f"   at.data_ptr={at.data_ptr()}, a.data_ptr={a.data_ptr()}")
print(f"   same storage: {at.data_ptr() == a.data_ptr()}")

print("4. Calling at.cpu() ...")
sys.stdout.flush()
result = at.cpu()
print(f"   result={result}")
print("DONE")