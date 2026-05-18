# vector_sum_f32.asm — sum = Σ a[i] for i=0..n-1
# REDUCTION: Params at addr 0: [addr_a, addr_out, n]
# Only thread 0 works. Uses x20-x24 to avoid clobbering a0-a2.

lw a0, 0(x0)
lw a1, 4(x0)
lw a2, 8(x0)

bne x5, x0, done

add x20, x0, x0

add x21, x0, x0

loop:
bge x21, a2, end_loop
slli x22, x21, 2
add x23, a0, x22
lw x24, 0(x23)
fadd.s x20, x20, x24
addi x21, x21, 1
beq x0, x0, loop

end_loop:
sw x20, 0(a1)

done:
halt