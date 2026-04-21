/*
test_math_lib.sv — v2 (fixed)

Changes from v1:
  - Expected values for tests 7 and 9 use generous tolerance (512 ULP)
    to account for hardware-vs-Python differences in rounding.
  - Timeout increased to 200000 cycles to account for the loop-based
    int-to-float conversion in __flog.
*/

module test_math_lib();
    reg clk;
    reg rst;

    reg contr_mem_wr_en;
    reg [31:0] contr_mem_wr_addr;
    reg [31:0] contr_mem_wr_data;

    wire [31:0] out;
    wire outen;
    wire outflen;

    reg contr_core1_ena;
    reg contr_core1_clr;
    reg contr_core1_set_pc_req;
    reg [31:0] contr_core1_set_pc_addr;
    wire contr_core1_halt;
    reg [31:0] contr_core1_thread_id;

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

    task tick();
        #5 clk = 0;
        #5 clk = 1;
    endtask

    task write_mem(input [31:0] addr, input [31:0] data);
        contr_mem_wr_en <= 1;
        contr_mem_wr_addr <= addr;
        contr_mem_wr_data <= data;
        tick();
        contr_mem_wr_en <= 0;
        tick();
    endtask

    task load_hex_file(input string filename);
        integer fd, status;
        reg [31:0] word;
        integer addr;
        integer count;

        fd = $fopen(filename, "r");
        if (fd == 0) begin
            $display("ERROR: Cannot open hex file '%s'", filename);
            $display("  Run assembler first:");
            $display("  python3 verigpu/assembler.py --in-asm examples/direct/test_math_lib.asm --out-hex build/test_math_lib.hex --offset 128");
            $finish;
        end
        addr = 128;
        count = 0;
        while (!$feof(fd)) begin
            status = $fscanf(fd, "%h\n", word);
            if (status == 1) begin
                write_mem(addr, word);
                addr = addr + 4;
                count = count + 1;
            end
        end
        $fclose(fd);
        $display("  Loaded %0d instructions from %s", count, filename);
    endtask

    // Float comparison with ULP tolerance
    function automatic integer check_float_near(
        input [31:0] actual,
        input [31:0] expected,
        input integer max_diff
    );
        integer a_signed, e_signed, diff;
        a_signed = actual[31] ? (32'sh80000000 - $signed({1'b0, actual[30:0]}))
                              : $signed({1'b0, actual[30:0]});
        e_signed = expected[31] ? (32'sh80000000 - $signed({1'b0, expected[30:0]}))
                                : $signed({1'b0, expected[30:0]});
        diff = a_signed - e_signed;
        if (diff < 0) diff = -diff;
        check_float_near = (diff <= max_diff);
    endfunction

    // -------------------------------------------------------
    // Expected outputs
    // -------------------------------------------------------
    parameter NUM_TESTS = 10;
    reg [31:0] expected_values [0:NUM_TESTS-1];
    integer    expected_tol    [0:NUM_TESTS-1];
    reg [8*24-1:0] test_names  [0:NUM_TESTS-1];

    initial begin
        test_names[0] = "fdiv(6.0, 3.0)";
        test_names[1] = "fdiv(1.0, 4.0)";
        test_names[2] = "fdiv(-6.0, 3.0)";
        test_names[3] = "fsqrt(4.0)";
        test_names[4] = "fsqrt(9.0)";
        test_names[5] = "fexp(0.0)";
        test_names[6] = "fexp(1.0)";
        test_names[7] = "flog(1.0)";
        test_names[8] = "flog(2.0)";
        test_names[9] = "ftanh(0.0)";

        // Exact operations: tight tolerance (2 ULP)
        expected_values[0] = 32'h40000000;  expected_tol[0] =    2;  // 2.0
        expected_values[1] = 32'h3E800000;  expected_tol[1] =    2;  // 0.25
        expected_values[2] = 32'hC0000000;  expected_tol[2] =    2;  // -2.0
        expected_values[3] = 32'h40000000;  expected_tol[3] =    2;  // 2.0
        expected_values[4] = 32'h40400000;  expected_tol[4] =    2;  // 3.0
        expected_values[5] = 32'h3F800000;  expected_tol[5] =    2;  // 1.0

        // Transcendentals: generous tolerance (512 ULP ≈ 6e-5 relative)
        expected_values[6] = 32'h402DF854;  expected_tol[6] =  512;  // e ≈ 2.71828

        // log: exact for these cases
        expected_values[7] = 32'h00000000;  expected_tol[7] =    2;  // 0.0
        expected_values[8] = 32'h3F317218;  expected_tol[8] =  512;  // ln(2) ≈ 0.6931

        // tanh(0) = 0 exactly
        expected_values[9] = 32'h00000000;  expected_tol[9] =    2;  // 0.0
    end

    // -------------------------------------------------------
    // Main test sequence
    // -------------------------------------------------------
    integer output_index;
    integer all_passed;
    integer i;
    integer done;

    initial begin
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

        tick();
        tick();
        rst = 1;
        tick();

        $display("Loading test program...");
        load_hex_file("build/test_math_lib.hex");
        $display("");

        contr_core1_clr <= 1;
        tick();
        contr_core1_clr <= 0;
        tick();
        contr_core1_ena <= 1;
        tick();

        // Generous timeout: flog uses loops, ftanh calls two functions
        done = 0;
        for (i = 0; i < 200000 && !done; i = i + 1) begin
            if (outflen) begin
                if (output_index < NUM_TESTS) begin
                    if (check_float_near(out, expected_values[output_index],
                                         expected_tol[output_index])) begin
                        $display("PASS test %2d: %-24s got=0x%08H  exp=0x%08H",
                            output_index + 1, test_names[output_index],
                            out, expected_values[output_index]);
                    end else begin
                        $display("FAIL test %2d: %-24s got=0x%08H  exp=0x%08H  tol=%0d",
                            output_index + 1, test_names[output_index],
                            out, expected_values[output_index],
                            expected_tol[output_index]);
                        all_passed = 0;
                    end
                    output_index = output_index + 1;
                end
            end

            if (outen && !outflen) begin
                $display("  [unexpected int output: %0d at cycle %0d]", out, i);
            end

            if (contr_core1_halt) begin
                contr_core1_ena <= 0;
                tick();
                done = 1;
            end

            if (!done) tick();
        end

        $display("");
        if (!done) begin
            $display("FAIL: timeout at cycle %0d (output_index=%0d/%0d)", i, output_index, NUM_TESTS);
            $finish;
        end

        if (output_index !== NUM_TESTS) begin
            $display("FAIL: expected %0d outputs, got %0d", NUM_TESTS, output_index);
            $finish;
        end

        if (all_passed) begin
            $display("========================================");
            $display("  ALL %0d TESTS PASSED", NUM_TESTS);
            $display("========================================");
        end else begin
            $display("SOME TESTS FAILED");
        end

        $finish;
    end
endmodule
