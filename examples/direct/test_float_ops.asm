; ============================================================
; test_float_ops.asm — Prueba de operaciones float
; ============================================================
;
; Este programa ejercita todas las instrucciones float nuevas:
;   FSUB.S, FEQ.S, FLT.S, FLE.S, FSGNJN.S (FNEG), FSGNJX.S (FABS)
;
; Cada test carga operandos, ejecuta la operación, y envía el
; resultado a stdout. El testbench verifica la secuencia de salidas.
;
; Convenciones de registros:
;   x6 = dirección stdout entero (0xF4240 = 1000000)
;   x7 = dirección stdout float  (0xF4248 = 1000008)
;   x8 = dirección halt          (0xF4244 = 1000004)
;   x1, x2 = operandos
;   x3 = resultado
;
; Para ejecutar:
;   python verigpu/assembler.py \
;       --in-asm examples/direct/test_float_ops.asm \
;       --out-hex build/test_float_ops.hex

; --- Setup: cargar direcciones de I/O ---
li x6, 1000000       ; stdout int
li x7, 1000008       ; stdout float
li x8, 1000004       ; halt

; --- Test 1: FSUB.S  3.0 - 1.0 = 2.0 ---
li x1, 3.0
li x2, 1.0
fsub.s x3, x1, x2
sw x3, 0(x7)          ; output float: expect 2.0

; --- Test 2: FSUB.S  1.0 - 3.0 = -2.0 ---
li x1, 1.0
li x2, 3.0
fsub.s x3, x1, x2
sw x3, 0(x7)          ; output float: expect -2.0

; --- Test 3: FLT.S  1.0 < 3.0 → 1 ---
li x1, 1.0
li x2, 3.0
flt.s x3, x1, x2
sw x3, 0(x6)          ; output int: expect 1

; --- Test 4: FLT.S  3.0 < 1.0 → 0 ---
li x1, 3.0
li x2, 1.0
flt.s x3, x1, x2
sw x3, 0(x6)          ; output int: expect 0

; --- Test 5: FEQ.S  3.0 == 3.0 → 1 ---
li x1, 3.0
li x2, 3.0
feq.s x3, x1, x2
sw x3, 0(x6)          ; output int: expect 1

; --- Test 6: FEQ.S  1.0 == 3.0 → 0 ---
li x1, 1.0
li x2, 3.0
feq.s x3, x1, x2
sw x3, 0(x6)          ; output int: expect 0

; --- Test 7: FLE.S  1.0 <= 1.0 → 1 ---
li x1, 1.0
li x2, 1.0
fle.s x3, x1, x2
sw x3, 0(x6)          ; output int: expect 1

; --- Test 8: FLE.S  3.0 <= 1.0 → 0 ---
li x1, 3.0
li x2, 1.0
fle.s x3, x1, x2
sw x3, 0(x6)          ; output int: expect 0

; --- Test 9: FNEG.S  neg(3.0) = -3.0 ---
li x1, 3.0
fneg.s x3, x1
sw x3, 0(x7)          ; output float: expect -3.0

; --- Test 10: FABS.S  abs(-3.0) = 3.0 ---
li x1, -3.0
fabs.s x3, x1
sw x3, 0(x7)          ; output float: expect 3.0

; --- Halt ---
halt