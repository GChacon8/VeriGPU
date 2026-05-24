# guardar como pytorch_backend/test_realloc.py
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

print("\n=== Test A: ensuciar memoria, liberar, y reusar ===")
# 1. Crear un tensor con valores grandes (ensucia un chunk)
dirty = torch.tensor([6.0, 6.0, 6.0, 6.0]).to('verigpu')
_ = (dirty * dirty).cpu()   # forzar uso en GPU; resultado [36,36,36,36]
del dirty                    # liberar -> chunk vuelve al pool con 6.0/36.0 en GPU

# 2. Ahora reutilizar ese chunk con un tensor que tiene cero
pred = torch.tensor([0.0, 0.128809, 0.33669, 0.4655]).to('verigpu')
Y = torch.tensor([0.0, 1.0, 1.0, 1.0]).to('verigpu')
diff = pred - Y
print(f"  diff = {diff.cpu().tolist()}")
sq = diff * diff
print(f"  sq   = {sq.cpu().tolist()}")
print(f"  (esperado [0, ~0.759, ~0.440, ~0.286])")
print(f"  Si sq[0] = 36.0 -> confirmado: chunk reutilizado con basura GPU")

print("\n=== Test B: misma secuencia muchas veces seguidas ===")
for i in range(5):
    pred = torch.tensor([0.0, 0.5, 0.5, 0.5]).to('verigpu')
    Y = torch.tensor([0.0, 1.0, 1.0, 1.0]).to('verigpu')
    diff = pred - Y
    sq = diff * diff
    s = sq.cpu().tolist()
    bad = "  <<< MAL" if abs(s[0]) > 0.001 else ""
    print(f"  iter {i}: sq = {[round(v,4) for v in s]}{bad}")