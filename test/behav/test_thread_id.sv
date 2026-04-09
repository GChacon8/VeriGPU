/*
test_thread_id.sv

Behavioral test: verifies that the core receives a thread_id
via its new input port, loads it into register x5 (t0) during clr,
and that a program can read and output that value.

The test program (equivalent to "outr x5; halt") is hardcoded as 5
RISC-V instructions:

  Address 128: LUI   x6, 0xF4       -> x6 = 0xF4000
  Address 132: ADDI  x6, x6, 0x240  -> x6 = 0xF4240 (stdout addr = 1000000)
  Address 136: SW    x5, 0(x6)      -> write thread_id to stdout
  Address 140: ADDI  x6, x6, 4      -> x6 = 0xF4244 (halt addr = 1000004)
  Address 144: SW    x0, 0(x6)      -> halt

Convention notes:
  - Uses tick() task for clock advancement (neg edge then pos edge)
  - Uses <= for testbench assignments (simulates flip-flop delay per coding guidelines)
  - Program loads via contr_mem_wr_en (controller memory port, no simulated delay)
  - Core starts at PC=128 after clr (hardcoded in core.sv)
*/

module test_thread_id();
    reg clk;
    reg rst;

    // Memory write interface (to load program before enabling core)
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

    // The new thread_id input
    reg [31:0] contr_core1_thread_id;

    // Instantiate core_and_mem (modified with thread_id port)
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
    // Clock helpers (following project conventions)
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
    task load_program();
        //  "outr x5"  expands to:
        write_mem(128, 32'h000F4337);  // LUI   x6, 0xF4
        write_mem(132, 32'h24030313);  // ADDI  x6, x6, 0x240   -> x6 = 0xF4240
        write_mem(136, 32'h00532023);  // SW    x5, 0(x6)       -> stdout <- x5

        //  "halt"  expands to:
        write_mem(140, 32'h00430313);  // ADDI  x6, x6, 4       -> x6 = 0xF4244
        write_mem(144, 32'h00032023);  // SW    x0, 0(x6)       -> halt
    endtask

    // -------------------------------------------------------
    // Run a single test: set thread_id, reset core, run, check output
    // -------------------------------------------------------
    integer output_received;
    integer output_value;

    task run_test(input [31:0] tid);
        integer i;
        integer done;
        output_received = 0;
        output_value = 0;
        done = 0;

        $display("--- Test: thread_id = %0d ---", tid);

        // Set the thread_id wire
        contr_core1_thread_id = tid;

        // Reset the entire system
        rst = 0;
        contr_core1_ena = 0;
        contr_core1_clr = 0;
        contr_core1_set_pc_req = 0;
        contr_core1_set_pc_addr = 0;
        contr_mem_wr_en = 0;
        tick();
        tick();

        // Release reset
        rst = 1;
        tick();

        // Load the program into memory (core is disabled)
        load_program();

        // Assert clr for one cycle:
        //   - sets PC = 128
        //   - loads regs[5] = thread_id
        //   - resets core state to C0
        contr_core1_clr <= 1;
        tick();
        contr_core1_clr <= 0;
        tick();

        // Enable the core — it starts executing from PC=128
        contr_core1_ena <= 1;
        tick();

        // Wait for output or halt (with timeout)
        for (i = 0; i < 500 && !done; i = i + 1) begin
            if (outen) begin
                output_received = 1;
                output_value = out;
                $display("  output received: %0d", out);
            end
            if (contr_core1_halt) begin
                // Done
                contr_core1_ena <= 0;
                tick();
                // Verify
                if (!output_received) begin
                    $display("FAIL: core halted but no output was received");
                    $finish;
                end
                if (output_value !== tid) begin
                    $display("FAIL: expected %0d, got %0d", tid, output_value);
                    $finish;
                end
                $display("PASS: thread_id=%0d, output=%0d", tid, output_value);
                done = 1;
            end
            if (!done) tick();
        end

        // If we get here without done, we timed out
        if (!done) begin
            $display("FAIL: timeout — core did not halt within 500 cycles");
            $finish;
        end
    endtask

    // -------------------------------------------------------
    // Main test sequence
    // -------------------------------------------------------
    initial begin
        clk = 0;
        rst = 0;
        contr_mem_wr_en = 0;
        contr_core1_ena = 0;
        contr_core1_clr = 0;
        contr_core1_set_pc_req = 0;
        contr_core1_set_pc_addr = 0;
        contr_core1_thread_id = 0;

        // Test with four different thread_id values
        run_test(42);
        run_test(7);
        run_test(0);
        run_test(255);

        $display("");
        $display("ALL 4 TESTS PASSED");
        $finish;
    end
endmodule