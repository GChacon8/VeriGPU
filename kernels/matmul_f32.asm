# matmul_f32.asm — C = A × B  (matrix multiplication)
#
# Thread tid computes ROW tid of output C.
# C[tid][j] = Σ_k A[tid][k] * B[k][j]   for all j in 0..N-1
#
# With 4 cores and batching, rows are processed 4 at a time.
# For an MxN output, we need ceil(M/4) kernel launches.
#
# Params at addr 0: [addr_A, addr_B, addr_C, M, K, N]
#   addr_A: base address of A [M×K]
#   addr_B: base address of B [K×N]
#   addr_C: base address of C [M×N]
#   M: rows of A (and C)
#   K: cols of A = rows of B
#   N: cols of B (and C)
#
# Register allocation (using x20+ to avoid clobbering params):
#   a0-a5 (x10-x15) = parameters (read-only after load)
#   x5  = thread_id (tid = which row to compute)
#   x20 = j (column loop counter)
#   x21 = k (inner loop counter)
#   x22 = accumulator (float)
#   x23-x27 = temporaries for address computation and loads
#   x28 = &A[tid][0] (precomputed, constant per thread)
#   x29 = &C[tid][0] (precomputed, constant per thread)

# Load 6 parameters
lw a0, 0(x0)
lw a1, 4(x0)
lw a2, 8(x0)
lw a3, 12(x0)
lw a4, 16(x0)
lw a5, 20(x0)

# Bounds check
bge x5, a3, done

# Precompute base pointers for this row
mul x28, x5, a4
slli x28, x28, 2
add x28, a0, x28

mul x29, x5, a5
slli x29, x29, 2
add x29, a2, x29

# Outer loop: for j = 0..N-1
add x20, x0, x0

col_loop:
bge x20, a5, done

# acc = 0.0
add x22, x0, x0

# Inner loop: for k = 0..K-1
add x21, x0, x0

k_loop:
bge x21, a4, end_k

# A[tid][k]: &A[tid][0] + k*4
slli x23, x21, 2
add x23, x28, x23
lw x24, 0(x23)

# B[k][j]: addr_B + (k*N + j)*4
mul x26, x21, a5
add x26, x26, x20
slli x26, x26, 2
add x26, a1, x26
lw x25, 0(x26)

# acc += A[tid][k] * B[k][j]
fmul.s x27, x24, x25
fadd.s x22, x22, x27

addi x21, x21, 1
beq x0, x0, k_loop

end_k:
# C[tid][j] = acc
slli x23, x20, 2
add x23, x29, x23
sw x22, 0(x23)

addi x20, x20, 1
beq x0, x0, col_loop

done:
halt
