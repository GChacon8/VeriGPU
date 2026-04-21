; ============================================================
; math_lib.asm — Software math library for VeriGPU
; ============================================================
;
; v3 FIX: __flog bypasses polynomial when mantissa bits are zero
; (avoids float_add_pipeline bug where equal-value subtraction
; produces 0x00800000 instead of 0x00000000).

__fdiv:
    fsgnjx.s t3, a0, a1
    fsgnjx.s a0, a0, a0
    fsgnjx.s a1, a1, a1
    li t0, 0x7EF311C3
    sub t0, t0, a1
    li t1, 2.0
    fmul.s t2, a1, t0
    fsub.s t2, t1, t2
    fmul.s t0, t0, t2
    fmul.s t2, a1, t0
    fsub.s t2, t1, t2
    fmul.s t0, t0, t2
    fmul.s t2, a1, t0
    fsub.s t2, t1, t2
    fmul.s t0, t0, t2
    fmul.s a0, a0, t0
    fsgnj.s a0, a0, t3
    jalr zero, 0(ra)

__fsqrt:
    srli t0, a0, 1
    li t1, 0x5F3759DF
    sub t0, t1, t0
    li t1, 1.5
    li t2, 0.5
    fmul.s t3, t2, a0
    fmul.s t4, t0, t0
    fmul.s t4, t3, t4
    fsub.s t4, t1, t4
    fmul.s t0, t0, t4
    fmul.s t4, t0, t0
    fmul.s t4, t3, t4
    fsub.s t4, t1, t4
    fmul.s t0, t0, t4
    fmul.s t4, t0, t0
    fmul.s t4, t3, t4
    fsub.s t4, t1, t4
    fmul.s t0, t0, t4
    fmul.s a0, a0, t0
    jalr zero, 0(ra)

__fexp:
    li t0, 0.03125
    fmul.s t1, a0, t0
    li t3, 0.04166667
    fmul.s t3, t1, t3
    li t4, 0.16666667
    fadd.s t3, t3, t4
    fmul.s t3, t1, t3
    li t4, 0.5
    fadd.s t3, t3, t4
    fmul.s t3, t1, t3
    li t4, 1.0
    fadd.s t3, t3, t4
    fmul.s t3, t1, t3
    fadd.s a0, t3, t4
    fmul.s a0, a0, a0
    fmul.s a0, a0, a0
    fmul.s a0, a0, a0
    fmul.s a0, a0, a0
    fmul.s a0, a0, a0
    jalr zero, 0(ra)

__flog:
    srli t0, a0, 23
    andi t0, t0, 255
    li t1, 0x007FFFFF
    and t2, a0, t1
    li t4, 0.0
    li t5, 1.0
    li t3, 127
    beq t0, t3, __flog_exp_done
    bltu t0, t3, __flog_exp_neg
    sub t0, t0, t3
__flog_exp_pos_loop:
    beq t0, zero, __flog_exp_done
    fadd.s t4, t4, t5
    addi t0, t0, -1
    jal zero, __flog_exp_pos_loop
__flog_exp_neg:
    sub t0, t3, t0
__flog_exp_neg_loop:
    beq t0, zero, __flog_exp_done
    fsub.s t4, t4, t5
    addi t0, t0, -1
    jal zero, __flog_exp_neg_loop
__flog_exp_done:
    li t5, 0.6931472
    fmul.s t4, t4, t5
    ; Skip polynomial if mantissa is zero (power of 2)
    beq t2, zero, __flog_skip_poly
    ; Reconstruct 1.mantissa, compute m
    li t1, 0x3F800000
    or t2, t2, t1
    li t5, 1.0
    fsub.s t2, t2, t5
    ; Horner polynomial
    li t5, 0.2
    fmul.s t6, t2, t5
    li t5, 0.25
    fsub.s t6, t5, t6
    fmul.s t6, t2, t6
    li t5, 0.33333333
    fsub.s t6, t5, t6
    fmul.s t6, t2, t6
    li t5, 0.5
    fsub.s t6, t5, t6
    fmul.s t6, t2, t6
    li t5, 1.0
    fsub.s t6, t5, t6
    fmul.s t6, t2, t6
    fadd.s a0, t4, t6
    jalr zero, 0(ra)
__flog_skip_poly:
    mv a0, t4
    jalr zero, 0(ra)

__ftanh:
    addi sp, sp, -8
    sw ra, 0(sp)
    sw s1, 4(sp)
    li t0, 2.0
    fmul.s a0, a0, t0
    jal ra, __fexp
    mv s1, a0
    li t0, 1.0
    fsub.s a0, s1, t0
    fadd.s a1, s1, t0
    jal ra, __fdiv
    lw ra, 0(sp)
    lw s1, 4(sp)
    addi sp, sp, 8
    jalr zero, 0(ra)
    