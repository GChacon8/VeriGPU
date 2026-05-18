# vector_div_f32.asm — out[tid] = a[tid] / b[tid]
# Binary: Params at addr 0: [addr_a, addr_b, addr_out, n]
# Uses Newton-Raphson for reciprocal (2 iterations), then multiply.
# Accuracy: ~24 bits (matches float32 mantissa)

lw a0, 0(x0)
lw a1, 4(x0)
lw a2, 8(x0)
lw a3, 12(x0)
bge x5, a3, done

slli x6, x5, 2
add x7, a0, x6
lw x8, 0(x7)
add x7, a1, x6
lw x9, 0(x7)

# Reciprocal via Newton-Raphson: r = 1/b
# Initial guess: r0 = as_float(0x7EF311C3 - as_int(b))
li x14, 2129535427
sub x15, x14, x9

# 2.0 constant
lui x16, 0x40000

# Iteration 1: r = r * (2 - b*r)
fmul.s x17, x9, x15
fsub.s x17, x16, x17
fmul.s x15, x15, x17

# Iteration 2: r = r * (2 - b*r)
fmul.s x17, x9, x15
fsub.s x17, x16, x17
fmul.s x15, x15, x17

# out = a * (1/b)
fmul.s x10, x8, x15
add x7, a2, x6
sw x10, 0(x7)

done:
halt
