/*
test_float_ops.sv

Behavioral test: verifies FSUB.S, FEQ.S, FLT.S, FLE.S,
FSGNJN.S (FNEG), and FSGNJX.S (FABS) instructions.

The test program is hardcoded as RISC-V machine code (63 instructions).
It outputs 10 results through the stdout I/O ports:
  - 4 float results (via outflen): FSUB, FNEG, FABS
  - 6 integer results (via outen): FLT, FEQ, FLE comparisons

Convention notes:
  - Uses tick() for clock advancement (neg edge then pos edge)
  - Uses <= for testbench assignments (simulates flip-flop delay)
  - Program loaded via contr_mem_wr_en (controller port, no delay)
  - Core starts at PC=128 after clr
*/

module test_float_ops();
    reg clk;
    reg rst;

    // Memory write interface
    reg contr_mem_wr_en;
    reg [31:0] contr_mem_wr_addr;
    reg [31:0] contr_mem_wr_data;

    // Core output signals
    wire [31:0] out;
    wire outen;
    wire outflen;

    // Core control signals
    reg contr_core1_ena;
    reg contr_core1_clr;
    reg contr_core1_set_pc_req;
    reg [31:0] contr_core1_set_pc_addr;
    wire contr_core1_halt;

    // Thread ID (from CP-1 multicore, set to 0 for this test)
    reg [31:0] contr_core1_thread_id;

    // Instantiate core_and_mem
    core_and_mem dut(
        .clk(clk),
        .rst(rst),
        .contr_mem_wr_en(contr_mem_wr_en),
        .contr_mem_wr_addr(contr_mem_wr_addr),
        .contr_mem_wr_data(contr_mem_wr_data),
        .out(out),
        .outen(outen),
        .outflen(outflen),
        .contr_core1_ena(contr_core1_ena),
        .contr_core1_clr(contr_core1_clr),
        .contr_core1_set_pc_req(contr_core1_set_pc_req),
        .contr_core1_set_pc_addr(contr_core1_set_pc_addr),
        .contr_core1_thread_id(contr_core1_thread_id),
        .contr_core1_halt(contr_core1_halt)
    );

    // -------------------------------------------------------
    // Clock helper
    // -------------------------------------------------------
    task tick();
        #5 clk = 0;
        #5 clk = 1;
    endtask

    // -------------------------------------------------------
    // Write one word to memory via controller port
    // -------------------------------------------------------
    task write_mem(input [31:0] addr, input [31:0] data);
        contr_mem_wr_en <= 1;
        contr_mem_wr_addr <= addr;
        contr_mem_wr_data <= data;
        tick();
        contr_mem_wr_en <= 0;
        tick();
    endtask

    // -------------------------------------------------------
    // Load the test program into memory starting at address 128
    // -------------------------------------------------------
    // Program structure:
    //   addr 128-150: Setup (LI x6=stdout_int, x7=stdout_float, x8=halt_addr)
    //   addr 152-175: Test 1 (FSUB 3.0-1.0) + Test 2 (FSUB 1.0-3.0)
    //   addr 200-243: Tests 3-4 (FLT)
    //   addr 248-291: Tests 5-6 (FEQ)
    //   addr 296-339: Tests 7-8 (FLE)
    //   addr 344-355: Test 9 (FNEG)
    //   addr 360-375: Test 10 (FABS)
    //   addr 376: HALT

    task load_program();
        // --- Setup: LI x6, 0xF4240 (int stdout) ---
        write_mem(128, 32'h000F4337);  // LUI   x6, 0xF4
        write_mem(132, 32'h24030313);  // ADDI  x6, x6, 0x240

        // --- Setup: LI x7, 0xF4248 (float stdout) ---
        write_mem(136, 32'h000F43B7);  // LUI   x7, 0xF4
        write_mem(140, 32'h24838393);  // ADDI  x7, x7, 0x248

        // --- Setup: LI x8, 0xF4244 (halt addr) ---
        write_mem(144, 32'h000F4437);  // LUI   x8, 0xF4
        write_mem(148, 32'h24440413);  // ADDI  x8, x8, 0x244

        // === Test 1: FSUB.S  3.0 - 1.0 = 2.0 ===
        write_mem(152, 32'h404000B7);  // LUI   x1, 0x40400  (3.0 upper)
        write_mem(156, 32'h00008093);  // ADDI  x1, x1, 0
        write_mem(160, 32'h3F800137);  // LUI   x2, 0x3F800  (1.0 upper)
        write_mem(164, 32'h00010113);  // ADDI  x2, x2, 0
        write_mem(168, 32'h082081D3);  // FSUB.S x3, x1, x2
        write_mem(172, 32'h0033A023);  // SW    x3, 0(x7)     -> float out

        // === Test 2: FSUB.S  1.0 - 3.0 = -2.0 ===
        write_mem(176, 32'h3F8000B7);  // LUI   x1, 0x3F800  (1.0)
        write_mem(180, 32'h00008093);  // ADDI  x1, x1, 0
        write_mem(184, 32'h40400137);  // LUI   x2, 0x40400  (3.0)
        write_mem(188, 32'h00010113);  // ADDI  x2, x2, 0
        write_mem(192, 32'h082081D3);  // FSUB.S x3, x1, x2
        write_mem(196, 32'h0033A023);  // SW    x3, 0(x7)     -> float out

        // === Test 3: FLT.S  1.0 < 3.0 → 1 ===
        write_mem(200, 32'h3F8000B7);  // LUI   x1, 0x3F800
        write_mem(204, 32'h00008093);  // ADDI  x1, x1, 0
        write_mem(208, 32'h40400137);  // LUI   x2, 0x40400
        write_mem(212, 32'h00010113);  // ADDI  x2, x2, 0
        write_mem(216, 32'hA02091D3);  // FLT.S x3, x1, x2
        write_mem(220, 32'h00332023);  // SW    x3, 0(x6)     -> int out

        // === Test 4: FLT.S  3.0 < 1.0 → 0 ===
        write_mem(224, 32'h404000B7);  // LUI   x1, 0x40400
        write_mem(228, 32'h00008093);  // ADDI  x1, x1, 0
        write_mem(232, 32'h3F800137);  // LUI   x2, 0x3F800
        write_mem(236, 32'h00010113);  // ADDI  x2, x2, 0
        write_mem(240, 32'hA02091D3);  // FLT.S x3, x1, x2
        write_mem(244, 32'h00332023);  // SW    x3, 0(x6)     -> int out

        // === Test 5: FEQ.S  3.0 == 3.0 → 1 ===
        write_mem(248, 32'h404000B7);  // LUI   x1, 0x40400
        write_mem(252, 32'h00008093);  // ADDI  x1, x1, 0
        write_mem(256, 32'h40400137);  // LUI   x2, 0x40400
        write_mem(260, 32'h00010113);  // ADDI  x2, x2, 0
        write_mem(264, 32'hA020A1D3);  // FEQ.S x3, x1, x2
        write_mem(268, 32'h00332023);  // SW    x3, 0(x6)     -> int out

        // === Test 6: FEQ.S  1.0 == 3.0 → 0 ===
        write_mem(272, 32'h3F8000B7);  // LUI   x1, 0x3F800
        write_mem(276, 32'h00008093);  // ADDI  x1, x1, 0
        write_mem(280, 32'h40400137);  // LUI   x2, 0x40400
        write_mem(284, 32'h00010113);  // ADDI  x2, x2, 0
        write_mem(288, 32'hA020A1D3);  // FEQ.S x3, x1, x2
        write_mem(292, 32'h00332023);  // SW    x3, 0(x6)     -> int out

        // === Test 7: FLE.S  1.0 <= 1.0 → 1 ===
        write_mem(296, 32'h3F8000B7);  // LUI   x1, 0x3F800
        write_mem(300, 32'h00008093);  // ADDI  x1, x1, 0
        write_mem(304, 32'h3F800137);  // LUI   x2, 0x3F800
        write_mem(308, 32'h00010113);  // ADDI  x2, x2, 0
        write_mem(312, 32'hA02081D3);  // FLE.S x3, x1, x2
        write_mem(316, 32'h00332023);  // SW    x3, 0(x6)     -> int out

        // === Test 8: FLE.S  3.0 <= 1.0 → 0 ===
        write_mem(320, 32'h404000B7);  // LUI   x1, 0x40400
        write_mem(324, 32'h00008093);  // ADDI  x1, x1, 0
        write_mem(328, 32'h3F800137);  // LUI   x2, 0x3F800
        write_mem(332, 32'h00010113);  // ADDI  x2, x2, 0
        write_mem(336, 32'hA02081D3);  // FLE.S x3, x1, x2
        write_mem(340, 32'h00332023);  // SW    x3, 0(x6)     -> int out

        // === Test 9: FNEG.S  neg(3.0) = -3.0 ===
        // FSGNJN.S x3, x1, x1 (funct5=00100, funct3=001)
        write_mem(344, 32'h404000B7);  // LUI   x1, 0x40400  (3.0)
        write_mem(348, 32'h00008093);  // ADDI  x1, x1, 0
        write_mem(352, 32'h201091D3);  // FSGNJN.S x3, x1, x1
        write_mem(356, 32'h0033A023);  // SW    x3, 0(x7)     -> float out

        // === Test 10: FABS.S  abs(-3.0) = 3.0 ===
        // FSGNJX.S x3, x1, x1 (funct5=00100, funct3=010)
        write_mem(360, 32'hC04000B7);  // LUI   x1, 0xC0400  (-3.0)
        write_mem(364, 32'h00008093);  // ADDI  x1, x1, 0
        write_mem(368, 32'h2010A1D3);  // FSGNJX.S x3, x1, x1
        write_mem(372, 32'h0033A023);  // SW    x3, 0(x7)     -> float out

        // === HALT ===
        write_mem(376, 32'h00042023);  // SW    x0, 0(x8)     -> halt
    endtask

    // -------------------------------------------------------
    // Expected output table
    // -------------------------------------------------------
    // Each entry: {is_float, expected_value}
    //   is_float=1 → check outflen, is_float=0 → check outen

    reg [31:0] expected_values [0:9];
    reg        expected_is_float [0:9];
    reg [63:0] expected_desc [0:9]; // not used in sim, just for docs

    initial begin
        //  #   type     value         description
        expected_is_float[0] = 1; expected_values[0] = 32'h40000000; //  2.0  = 3.0 - 1.0
        expected_is_float[1] = 1; expected_values[1] = 32'hC0000000; // -2.0  = 1.0 - 3.0
        expected_is_float[2] = 0; expected_values[2] = 32'd1;        //  1    = (1.0 < 3.0)
        expected_is_float[3] = 0; expected_values[3] = 32'd0;        //  0    = (3.0 < 1.0)
        expected_is_float[4] = 0; expected_values[4] = 32'd1;        //  1    = (3.0 == 3.0)
        expected_is_float[5] = 0; expected_values[5] = 32'd0;        //  0    = (1.0 == 3.0)
        expected_is_float[6] = 0; expected_values[6] = 32'd1;        //  1    = (1.0 <= 1.0)
        expected_is_float[7] = 0; expected_values[7] = 32'd0;        //  0    = (3.0 <= 1.0)
        expected_is_float[8] = 1; expected_values[8] = 32'hC0400000; // -3.0  = neg(3.0)
        expected_is_float[9] = 1; expected_values[9] = 32'h40400000; //  3.0  = abs(-3.0)
    end

    // -------------------------------------------------------
    // Output collector
    // -------------------------------------------------------
    integer output_index;
    integer num_tests;
    integer all_passed;
    reg [31:0] received_value;
    reg        received_is_float;

    // -------------------------------------------------------
    // Main test sequence
    // -------------------------------------------------------
    integer i;
    integer done;

    initial begin
        num_tests = 10;
        all_passed = 1;
        output_index = 0;

        clk = 0;
        rst = 0;
        contr_mem_wr_en = 0;
        contr_core1_ena = 0;
        contr_core1_clr = 0;
        contr_core1_set_pc_req = 0;
        contr_core1_set_pc_addr = 0;
        contr_core1_thread_id = 0;

        // Reset
        tick();
        tick();
        rst = 1;
        tick();

        // Load program
        load_program();

        // Clear + enable core
        contr_core1_clr <= 1;
        tick();
        contr_core1_clr <= 0;
        tick();
        contr_core1_ena <= 1;
        tick();

        // Run until halt (with timeout)
        done = 0;
        for (i = 0; i < 5000 && !done; i = i + 1) begin
            // Check for output
            if (outen || outflen) begin
                received_value = out;
                received_is_float = outflen;

                if (output_index < num_tests) begin
                    // Verify output type
                    if (received_is_float !== expected_is_float[output_index]) begin
                        $display("FAIL test %0d: wrong output type (got %s, expected %s)",
                            output_index + 1,
                            received_is_float ? "float" : "int",
                            expected_is_float[output_index] ? "float" : "int");
                        all_passed = 0;
                    end
                    // Verify output value
                    else if (received_value !== expected_values[output_index]) begin
                        $display("FAIL test %0d: expected 0x%08H, got 0x%08H",
                            output_index + 1,
                            expected_values[output_index],
                            received_value);
                        all_passed = 0;
                    end
                    else begin
                        if (received_is_float)
                            $display("PASS test %2d: float output = 0x%08H", output_index + 1, received_value);
                        else
                            $display("PASS test %2d: int   output = %0d", output_index + 1, received_value);
                    end
                    output_index = output_index + 1;
                end
            end

            // Check for halt
            if (contr_core1_halt) begin
                contr_core1_ena <= 0;
                tick();
                done = 1;
            end

            if (!done) tick();
        end

        // Final report
        $display("");
        if (!done) begin
            $display("FAIL: timeout — core did not halt within 5000 cycles");
            $finish;
        end

        if (output_index !== num_tests) begin
            $display("FAIL: expected %0d outputs, got %0d", num_tests, output_index);
            $finish;
        end

        if (all_passed) begin
            $display("========================================");
            $display("  ALL %0d TESTS PASSED", num_tests);
            $display("========================================");
        end else begin
            $display("SOME TESTS FAILED");
            $finish;
        end

        $finish;
    end
endmodule
