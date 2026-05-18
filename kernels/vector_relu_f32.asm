# vector_relu_f32.asm — out[tid] = max(0, a[tid])
# UNARY: Params at addr 0: [addr_a, addr_out, n]
# Uses FLT.S to compare with zero, branch to store 0 or a.

lw a0, 0(x0)
lw a1, 4(x0)
lw a2, 8(x0)
bge x5, a2, done

slli x6, x5, 2
add x7, a0, x6
lw x8, 0(x7)

add x7, a1, x6
flt.s x17, x8, x0
bne x17, x0, neg
sw x8, 0(x7)
beq x0, x0, done
neg:
sw x0, 0(x7)

done:
halt
