# guardar como pytorch_backend/test_chain.py
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

def gpu_addr(t):
    return _verigpu_C.gpu_addr_of(t)

print("\n=== Test C: cadena sub -> mul SIN leer diff entremedio ===")
pred = torch.tensor([0.0, 0.128809, 0.33669, 0.4655]).to('verigpu')
Y = torch.tensor([0.0, 1.0, 1.0, 1.0]).to('verigpu')
diff = pred - Y                 # sub kernel
# NO llamar diff.cpu() aca
sq = diff * diff                # mul kernel, lee diff directo de GPU
print(f"  sq = {sq.cpu().tolist()}   (esperado [0, ~0.759, ~0.440, ~0.286])")

print("\n=== Test D: cadena sub -> mul LEYENDO diff entremedio ===")
pred = torch.tensor([0.0, 0.128809, 0.33669, 0.4655]).to('verigpu')
Y = torch.tensor([0.0, 1.0, 1.0, 1.0]).to('verigpu')
diff = pred - Y
d = diff.cpu().tolist()         # forzar sync_from_gpu de diff
print(f"  diff (leido) = {[round(v,4) for v in d]}")
sq = diff * diff
print(f"  sq = {sq.cpu().tolist()}   (esperado [0, ~0.759, ~0.440, ~0.286])")

print("\n=== Test E: replicar mm -> add -> sub -> mul completo ===")
X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]).to('verigpu')
w = torch.tensor([[0.33669],[0.128809]]).to('verigpu')
b = torch.tensor([0.0]).to('verigpu')
Y = torch.tensor([[0.],[1.],[1.],[1.]]).to('verigpu')

mm_out = torch.mm(X, w)
print(f"  mm_out = {mm_out.cpu().flatten().tolist()}")
pred = mm_out + b
print(f"  pred   = {pred.cpu().flatten().tolist()}")
diff = pred - Y
print(f"  diff   = {[round(v,4) for v in diff.cpu().flatten().tolist()]}")
print(f"  diff gpu_addr = {gpu_addr(diff)}")
sq = diff * diff
print(f"  sq     = {sq.cpu().flatten().tolist()}   (esperado [0, ~0.759, ~0.440, ~0.286])")

print("\n=== Test F: mismo que E pero SIN imprimir nada entremedio ===")
X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]).to('verigpu')
w = torch.tensor([[0.33669],[0.128809]]).to('verigpu')
b = torch.tensor([0.0]).to('verigpu')
Y = torch.tensor([[0.],[1.],[1.],[1.]]).to('verigpu')
mm_out = torch.mm(X, w)
pred = mm_out + b
diff = pred - Y
sq = diff * diff
print(f"  sq = {sq.cpu().flatten().tolist()}   (esperado [0, ~0.759, ~0.440, ~0.286])")
