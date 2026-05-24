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

def show(name, t):
    """Print tensor name + its CPU values, flagging any NaN."""
    vals = t.detach().cpu().flatten().tolist()
    has_nan = any(v != v for v in vals)
    marker = "  <<< NaN!!!" if has_nan else ""
    print(f"  {name:20s} = {[round(v, 6) for v in vals]}{marker}")

X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]).to('verigpu')
Y = torch.tensor([[0.],[1.],[1.],[1.]]).to('verigpu')
torch.manual_seed(42)
w = torch.randn(2,1).to('verigpu').detach().requires_grad_(True)
b = torch.zeros(1).to('verigpu').detach().requires_grad_(True)

print("=== INITIAL STATE ===")
show("X", X)
show("Y", Y)
show("w", w)
show("b", b)

for e in range(3):
    print(f"\n=== EPOCH {e}: FORWARD ===")

    # Forward, breaking it up so we can inspect each step
    mm_out = torch.mm(X, w)
    show("mm(X,w)", mm_out)

    pred = mm_out + b
    show("pred = mm + b", pred)

    diff = pred - Y
    show("diff = pred - Y", diff)

    sq = diff * diff
    show("sq = diff * diff", sq)

    loss = sq.mean()
    print(f"  loss (item)          = {loss.item():.6f}")

    print(f"\n=== EPOCH {e}: BACKWARD ===")
    loss.backward()
    show("w.grad", w.grad)
    show("b.grad", b.grad)

    print(f"\n=== EPOCH {e}: SGD ===")
    show("w (before)", w)
    show("b (before)", b)

    with torch.no_grad():
        # Compute on CPU
        w_grad_cpu = w.grad.cpu()
        b_grad_cpu = b.grad.cpu()
        print(f"  w.grad.cpu()         = {w_grad_cpu.flatten().tolist()}")
        print(f"  b.grad.cpu()         = {b_grad_cpu.flatten().tolist()}")

        w_new_cpu = w.data.cpu() - 0.1 * w_grad_cpu
        b_new_cpu = b.data.cpu() - 0.1 * b_grad_cpu
        print(f"  w_new (on CPU)       = {w_new_cpu.flatten().tolist()}")
        print(f"  b_new (on CPU)       = {b_new_cpu.flatten().tolist()}")

        w_new_gpu = w_new_cpu.to('verigpu')
        b_new_gpu = b_new_cpu.to('verigpu')
        show("w_new (after .to vgpu)", w_new_gpu)
        show("b_new (after .to vgpu)", b_new_gpu)

        w.data.copy_(w_new_gpu)
        b.data.copy_(b_new_gpu)

    show("w (after SGD)", w)
    show("b (after SGD)", b)

    w.grad = None
    b.grad = None