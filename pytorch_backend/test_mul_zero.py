# guardar como pytorch_backend/test_mul_zero.py
import sys, os, types, torch, subprocess

os.makedirs('build', exist_ok=True)
kernels = {
    'vadd_f32':'kernels/vector_add_f32_param.asm',
    'vsub_f32':'kernels/vector_sub_f32.asm',
    'vmul_f32':'kernels/vector_mul_f32.asm',
    'vneg_f32':'kernels/vector_neg_f32.asm',
    'vabs_f32':'kernels/vector_abs_f32.asm',
    'vdiv_f32':'kernels/vector_div_f32.asm',
    'vrelu_f32':'kernels/vector_relu_f32.asm',
    'vthreshold_bwd_f32':'kernels/vector_threshold_bwd_f32.asm',
    'vsum_f32':'kernels/vector_sum_f32.asm',
    'vmm_f32':'kernels/matmul_f32.asm'
}
for n, a in kernels.items():
    h = f'build/{os.path.basename(a).replace(".asm",".hex")}'
    subprocess.run(['python3','verigpu/assembler.py','--in-asm',a,'--out-hex',h],
                   check=True, capture_output=True)

import _verigpu_C
_m = types.ModuleType('torch.verigpu'); _m.__path__ = []
sys.modules['torch.verigpu'] = _m
torch.utils.rename_privateuse1_backend('verigpu')

for n, a in kernels.items():
    h = f'build/{os.path.basename(a).replace(".asm",".hex")}'
    with open(h) as f:
        w = [int(l.strip(), 16) for l in f if l.strip()]
    _verigpu_C.load_kernel(n, w)

print("\n=== Test 1: mul con dos tensores distintos, sin ceros ===")
a = torch.tensor([1.0, 2.0, 3.0, 4.0]).to('verigpu')
b = torch.tensor([5.0, 6.0, 7.0, 8.0]).to('verigpu')
r = (a * b).cpu().tolist()
print(f"  [1,2,3,4] * [5,6,7,8] = {r}   (esperado [5, 12, 21, 32])")

print("\n=== Test 2: mul con dos tensores distintos, con cero en pos 0 ===")
a = torch.tensor([0.0, 2.0, 3.0, 4.0]).to('verigpu')
b = torch.tensor([0.0, 6.0, 7.0, 8.0]).to('verigpu')
r = (a * b).cpu().tolist()
print(f"  [0,2,3,4] * [0,6,7,8] = {r}   (esperado [0, 12, 21, 32])")

print("\n=== Test 3: mul con el mismo tensor (a * a), sin ceros ===")
a = torch.tensor([1.0, 2.0, 3.0, 4.0]).to('verigpu')
r = (a * a).cpu().tolist()
print(f"  [1,2,3,4] * [1,2,3,4] = {r}   (esperado [1, 4, 9, 16])")

print("\n=== Test 4: mul con el mismo tensor (a * a), con cero ===")
a = torch.tensor([0.0, 2.0, 3.0, 4.0]).to('verigpu')
r = (a * a).cpu().tolist()
print(f"  [0,2,3,4] * [0,2,3,4] = {r}   (esperado [0, 4, 9, 16])")

print("\n=== Test 5: replica del caso del perceptrón ===")
pred = torch.tensor([0.0, 0.128809, 0.33669, 0.4655]).to('verigpu')
Y = torch.tensor([0.0, 1.0, 1.0, 1.0]).to('verigpu')
diff = pred - Y
print(f"  diff = {diff.cpu().tolist()}")
sq = diff * diff
print(f"  sq   = {sq.cpu().tolist()}   (esperado [0, ~0.759, ~0.440, ~0.286])")
