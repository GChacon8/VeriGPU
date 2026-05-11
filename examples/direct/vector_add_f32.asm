# vector_add_f32.asm — Float vector addition kernel for VeriGPU
#
# Computes: out[tid] = a[tid] + b[tid]  (IEEE 754 float32)
#
# This is the FIRST kernel that uses FADD.S in a multicore context.
# Each of the 4 cores processes one element in parallel.
# With batching, arrays of any size can be processed.
#
# Memory layout (hardcoded for standalone testbench):
#   a[]:   address 256  (input array 1)
#   b[]:   address 320  (input array 2)
#   out[]: address 384  (output array)
#
# Register usage:
#   x5  = thread_id (loaded by compute_unit during clr)
#   x6  = tid * 4 (byte offset)
#   x7  = address pointer (reused)
#   x8  = a[tid] value (float bits in integer register, zfinx)
#   x9  = b[tid] value
#   x10 = result of FADD.S
#
# Note: VeriGPU uses the Zfinx extension — float operations use
# integer registers. LW/SW load/store the IEEE 754 bits directly,
# and FADD.S operates on them as floats.

slli x6, x5, 2          # x6 = tid * 4 (each float is 4 bytes)
addi x7, x6, 256        # x7 = &a[tid]
lw   x8, 0(x7)          # x8 = a[tid]
addi x7, x6, 320        # x7 = &b[tid]
lw   x9, 0(x7)          # x9 = b[tid]
fadd.s x10, x8, x9      # x10 = a[tid] + b[tid]  ← THE KEY INSTRUCTION
addi x7, x6, 384        # x7 = &out[tid]
sw   x10, 0(x7)         # out[tid] = result
halt
