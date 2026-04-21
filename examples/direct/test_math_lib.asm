; ============================================================
; test_math_lib.asm — Software math library + test driver (v3)
; FDIV, FSQRT, FEXP, FLOG, FTANH
; ============================================================
;
; FIX LOG (v3):
;   Root cause: VeriGPU's float_add_pipeline produces 0x00800000
;   (2^-126) when subtracting two equal floats (e.g. 1.0 - 1.0)
;   instead of exact 0x00000000. The normalization loop shifts
;   until exp reaches 1 (minimum normalized) without detecting
;   that the mantissa is all zeros.
;
;   Fix: When input mantissa bits are zero (x is exact power of 2),
;   skip the fsub and polynomial entirely. ln(2^n) = n * ln(2).
;
; FIX STACK (v2):
;   Stack pointer set to 2000 (within mem_large's 512 words = 2048 bytes).

; ============ ENTRY: skip over library ============
jal zero, __test_main

; ============================================================
; __fdiv: a0 = a0 / a1
; ============================================================
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

; ============================================================
; __fsqrt: a0 = sqrt(a0)
; ============================================================
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

; ============================================================
; __fexp: a0 = exp(a0)
; ============================================================
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

; ============================================================
; __flog: a0 = ln(a0)   (natural log, a0 > 0)
; ============================================================
; Algorithm: IEEE 754 decomposition.
;   x = 2^(e-127) * (1+m)  →  ln(x) = (e-127)*ln(2) + ln(1+m)
;
; FIX: When mantissa bits are zero (x is exact power of 2),
; skip the polynomial entirely. This avoids the fsub(1.0, 1.0)
; bug in float_add_pipeline that produces 0x00800000 instead of 0.
;
__flog:
    ; Step 1: Extract biased exponent
    srli t0, a0, 23
    andi t0, t0, 255

    ; Step 2: Extract raw mantissa bits (without implicit 1)
    li t1, 0x007FFFFF
    and t2, a0, t1

    ; Step 3: Convert (biased_exp - 127) to float via loop
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
    ; t4 = float(biased_exp - 127)
    ; Multiply by ln(2)
    li t5, 0.6931472
    fmul.s t4, t4, t5

    ; Step 4: If mantissa bits == 0, x is exact power of 2.
    ; ln(2^n) = n * ln(2) = t4. Skip polynomial (avoids fsub bug).
    beq t2, zero, __flog_skip_poly

    ; Step 5: Reconstruct 1.mantissa and compute m = 1.mantissa - 1.0
    li t1, 0x3F800000
    or t2, t2, t1
    li t5, 1.0
    fsub.s t2, t2, t5

    ; Step 6: Horner polynomial for ln(1+m), 5 terms
    ;   ln(1+m) ≈ m*(1 - m*(1/2 - m*(1/3 - m*(1/4 - m*1/5))))
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

    ; Step 7: result = exponent_part + polynomial_part
    fadd.s a0, t4, t6
    jalr zero, 0(ra)

__flog_skip_poly:
    ; x is exact power of 2 → result = (exp-127) * ln(2) = t4
    mv a0, t4
    jalr zero, 0(ra)

; ============================================================
; __ftanh: a0 = tanh(a0)
; ============================================================
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

; ============================================================
; TEST MAIN
; ============================================================
__test_main:
    li sp, 2000
    li s3, 1000008

    ; Test 1: fdiv(6.0, 3.0)
    li a0, 6.0
    li a1, 3.0
    jal ra, __fdiv
    sw a0, 0(s3)

    ; Test 2: fdiv(1.0, 4.0)
    li a0, 1.0
    li a1, 4.0
    jal ra, __fdiv
    sw a0, 0(s3)

    ; Test 3: fdiv(-6.0, 3.0)
    li a0, -6.0
    li a1, 3.0
    jal ra, __fdiv
    sw a0, 0(s3)

    ; Test 4: fsqrt(4.0)
    li a0, 4.0
    jal ra, __fsqrt
    sw a0, 0(s3)

    ; Test 5: fsqrt(9.0)
    li a0, 9.0
    jal ra, __fsqrt
    sw a0, 0(s3)

    ; Test 6: fexp(0.0)
    li a0, 0.0
    jal ra, __fexp
    sw a0, 0(s3)

    ; Test 7: fexp(1.0)
    li a0, 1.0
    jal ra, __fexp
    sw a0, 0(s3)

    ; Test 8: flog(1.0) → expect 0.0
    li a0, 1.0
    jal ra, __flog
    sw a0, 0(s3)

    ; Test 9: flog(2.0) → expect ~0.6931472
    li a0, 2.0
    jal ra, __flog
    sw a0, 0(s3)

    ; Test 10: ftanh(0.0) → expect 0.0
    li a0, 0.0
    jal ra, __ftanh
    sw a0, 0(s3)

    halt
    