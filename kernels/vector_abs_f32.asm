# vector_abs_f32.asm — out[tid] = |a[tid]|
# UNARY: Params at addr 0: [addr_a, addr_out, n]
# fsgnjx.s rd, rs, rs  =  abs (sign XOR sign = 0 = positive)

lw a0, 0(x0)
lw a1, 4(x0)
lw a2, 8(x0)
bge x5, a2, done
slli x6, x5, 2
add x7, a0, x6
lw x8, 0(x7)
fsgnjx.s x10, x8, x8
add x7, a1, x6
sw x10, 0(x7)
done:
halt
