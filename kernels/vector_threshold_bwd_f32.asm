# vector_threshold_bwd_f32.asm — relu backward
# out[tid] = (input[tid] > 0) ? grad[tid] : 0
# Binary: Params at addr 0: [addr_grad, addr_input, addr_out, n]
# Threshold is hardcoded to 0 (always the case for relu backward)

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

add x7, a2, x6
flt.s x17, x0, x9
bne x17, x0, pos
sw x0, 0(x7)
beq x0, x0, done
pos:
sw x8, 0(x7)

done:
halt
