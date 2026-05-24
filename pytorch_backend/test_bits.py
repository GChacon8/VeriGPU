# guardar como pytorch_backend/test_bits.py
import sys, os, types, torch, subprocess, struct
os.makedirs('build', exist_ok=True)
K = {'vadd_f32':'kernels/vector_add_f32_param.asm','vsub_f32':'kernels/vector_sub_f32.asm','vmul_f32':'kernels/vector_mul_f32.asm','vneg_f32':'kernels/vector_neg_f32.asm','vabs_f32':'kernels/vector_abs_f32.asm','vdiv_f32':'kernels/vector_div_f32.asm','vrelu_f32':'kernels/vector_relu_f32.asm','vthreshold_bwd_f32':'kernels/vector_threshold_bwd_f32.asm','vsum_f32':'kernels/vector_sum_f32.asm','vmm_f32':'kernels/matmul_f32.asm'}
for n,a in K.items():
    h=f'build/{os.path.basename(a).replace(".asm",".hex")}'
    subprocess.run(['python3','verigpu/assembler.py','--in-asm',a,'--out-hex',h],check=True,capture_output=True)
import _verigpu_C
_m=types.ModuleType('torch.verigpu');_m.__path__=[];sys.modules['torch.verigpu']=_m
torch.utils.rename_privateuse1_backend('verigpu')
for n,a in K.items():
    h=f'build/{os.path.basename(a).replace(".asm",".hex")}'
    with open(h) as f: w=[int(l.strip(),16) for l in f if l.strip()]
    _verigpu_C.load_kernel(n,w)

def bits_host(t):
    vals = t.detach().cpu().flatten().tolist()
    return [hex(struct.unpack('I', struct.pack('f', v))[0]) for v in vals]

def bits_gpu(t):
    addr = _verigpu_C.gpu_addr_of(t)
    n = t.numel()
    floats = _verigpu_C.gpu_readback_floats(addr, n)
    return [hex(struct.unpack('I', struct.pack('f', v))[0]) for v in floats], addr

print("\n=== Replicar mm->add->sub y mirar bits de diff ===")
X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]).to('verigpu')
w = torch.tensor([[0.33669],[0.128809]]).to('verigpu')
b = torch.tensor([0.0]).to('verigpu')
Y = torch.tensor([[0.],[1.],[1.],[1.]]).to('verigpu')

mm_out = torch.mm(X, w)
print(f"mm_out host bits = {bits_host(mm_out)}")
g, addr = bits_gpu(mm_out)
print(f"mm_out GPU  bits = {g}  (addr={addr})")

pred = mm_out + b
print(f"pred host bits   = {bits_host(pred)}")
g, addr = bits_gpu(pred)
print(f"pred GPU  bits   = {g}  (addr={addr})")

diff = pred - Y
print(f"diff host bits   = {bits_host(diff)}")
g, addr = bits_gpu(diff)
print(f"diff GPU  bits   = {g}  (addr={addr})")
print(f"   ^ si host=0x0 pero GPU!=0x0 -> desincronizacion")
print(f"   ^ si ambos son subnormales (ej 0x00C00000) -> flush no los agarro")

sq = diff * diff
print(f"sq host bits     = {bits_host(sq)}")
print(f"sq values        = {sq.cpu().flatten().tolist()}")
