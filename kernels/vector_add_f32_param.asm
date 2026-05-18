# vector_add_f32_param.asm — Parameterized float vector addition
#
# Parameters stored in GPU memory at address 0:
#   addr 0:  base address of array a
#   addr 4:  base address of array b
#   addr 8:  base address of array out
#   addr 12: n (number of elements)
#
# x5 = thread_id (loaded by compute_unit during clr)
# Since x0 is hardwired to 0, "lw rd, offset(x0)" loads from
# absolute GPU address = offset.

# Load parameters from GPU memory
lw a0, 0(x0)
lw a1, 4(x0)
lw a2, 8(x0)
lw a3, 12(x0)

# Bounds check
bge x5, a3, done

# Compute: out[tid] = a[tid] + b[tid]
slli x6, x5, 2
add x7, a0, x6
lw x8, 0(x7)
add x7, a1, x6
lw x9, 0(x7)
fadd.s x10, x8, x9
add x7, a2, x6
sw x10, 0(x7)

done:
halt
