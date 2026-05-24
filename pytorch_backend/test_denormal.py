# guardar como pytorch_backend/test_denormal.py
import sys, os, types, torch, subprocess
os.makedirs('build', exist_ok=True)
kernels = {'vmul_f32':'kernels/vector_mul_f32.asm'}
# (cargá todos los kernels igual que en los otros tests; abreviado aca)
for n,a in {'vadd_f32':'kernels/vector_add_f32_param.asm','vsub_f32':'kernels/vector_sub_f32.asm','vmul_f32':'kernels/vector_mul_f32.asm','vneg_f32':'kernels/vector_neg_f32.asm','vabs_f32':'kernels/vector_abs_f32.asm','vdiv_f32':'kernels/vector_div_f32.asm','vrelu_f32':'kernels/vector_relu_f32.asm','vthreshold_bwd_f32':'kernels/vector_threshold_bwd_f32.asm','vsum_f32':'kernels/vector_sum_f32.asm','vmm_f32':'kernels/matmul_f32.asm'}.items():
    h=f'build/{os.path.basename(a).replace(".asm",".hex")}'
    subprocess.run(['python3','verigpu/assembler.py','--in-asm',a,'--out-hex',h],check=True,capture_output=True)
import _verigpu_C
_m=types.ModuleType('torch.verigpu');_m.__path__=[];sys.modules['torch.verigpu']=_m
torch.utils.rename_privateuse1_backend('verigpu')
for n,a in {'vadd_f32':'kernels/vector_add_f32_param.asm','vsub_f32':'kernels/vector_sub_f32.asm','vmul_f32':'kernels/vector_mul_f32.asm','vneg_f32':'kernels/vector_neg_f32.asm','vabs_f32':'kernels/vector_abs_f32.asm','vdiv_f32':'kernels/vector_div_f32.asm','vrelu_f32':'kernels/vector_relu_f32.asm','vthreshold_bwd_f32':'kernels/vector_threshold_bwd_f32.asm','vsum_f32':'kernels/vector_sum_f32.asm','vmm_f32':'kernels/matmul_f32.asm'}.items():
    h=f'build/{os.path.basename(a).replace(".asm",".hex")}'
    with open(h) as f: w=[int(l.strip(),16) for l in f if l.strip()]
    _verigpu_C.load_kernel(n,w)

import struct
# Crear un tensor con un subnormal en pos 0
subnormal = struct.unpack('f', struct.pack('I', 0x00C00000))[0]  # ~1.76e-38
print(f"subnormal usado: {subnormal}")
a = torch.tensor([subnormal, 2.0, 3.0, 4.0]).to('verigpu')
r = (a * a).cpu().tolist()
print(f"  subnormal^2 = {r[0]}  (matematicamente ~0, si da 36 o algo grande -> bug FMUL denormal)")

# Confirmar que 0*algo da cero limpio o subnormal
z = torch.tensor([0.0, 0.0]).to('verigpu')
k = torch.tensor([0.3367, 0.1288]).to('verigpu')
import struct as st
prod = (z * k).cpu()
bits = [hex(st.unpack('I', st.pack('f', v))[0]) for v in prod.tolist()]
print(f"  0.0 * 0.3367 = {prod.tolist()[0]}  bits={bits[0]}  (si no es 0x0 -> el FMUL deja residuo)")