/*
test_gpu_die_multicore.sv

CP-4+5 test: exercises the full gpu_die stack with compute_unit.
Replicates the exact protocol used by gpu_runtime.cpp:
  - gpuCopyToDevice  → INSTR_COPY_TO_GPU
  - gpuLaunchKernel  → INSTR_KERNEL_LAUNCH
  - gpuCopyFromDevice → INSTR_COPY_FROM_GPU

This is the first test that proves the complete chain works:
  host protocol → gpu_controller → compute_unit(4 cores) → memory

The test program (same as CP-3):
  slli x6, x5, 2      ; x6 = tid * 4
  sw   x5, 0(x6)       ; mem[tid*4] = tid
  lui  x7, 0xF4        ; x7 = 0xF4000
  addi x7, x7, 0x244   ; x7 = 0xF4244
  sw   x0, 0(x7)        ; halt
*/

module test_gpu_die_multicore();

    reg         clk, rst;
    reg  [31:0] cpu_recv_instr;
    reg  [31:0] cpu_in_data;
    wire [31:0] cpu_out_data;
    wire        cpu_out_ack;
    wire        halt;
    wire [31:0] out;
    wire        outen, outflen;

    gpu_die gpu_die_(
        .clk(clk), .rst(rst),
        .cpu_recv_instr(cpu_recv_instr),
        .cpu_in_data(cpu_in_data),
        .cpu_out_data(cpu_out_data),
        .cpu_out_ack(cpu_out_ack),
        .halt(halt),
        .out(out), .outen(outen), .outflen(outflen)
    );

    // -------------------------------------------------------
    // Helpers
    // -------------------------------------------------------
    task tick();
        #5 clk = 0;
        #5 clk = 1;
    endtask

    // Protocol constants (matching gpu_runtime.cpp)
    localparam INSTR_NOP            = 0;
    localparam INSTR_COPY_TO_GPU    = 1;
    localparam INSTR_COPY_FROM_GPU  = 2;
    localparam INSTR_KERNEL_LAUNCH  = 3;

    // -------------------------------------------------------
    // gpu_copy_to_device: replicates gpuCopyToDevice()
    //   Sends COPY_TO_GPU instruction, dest addr, size, then data words.
    //   Uses blocking assignments (=) for protocol signals, matching
    //   the C++ runtime's direct assignment before eval().
    // -------------------------------------------------------
    task gpu_copy_to_device(
        input [31:0] dest_addr,
        input integer num_words,
        input [31:0] data0, input [31:0] data1,
        input [31:0] data2, input [31:0] data3,
        input [31:0] data4
    );
        integer i;
        reg [31:0] words [0:4];

        words[0] = data0; words[1] = data1; words[2] = data2;
        words[3] = data3; words[4] = data4;

        cpu_recv_instr = INSTR_COPY_TO_GPU;
        tick();

        cpu_in_data = dest_addr;
        tick();

        cpu_in_data = num_words * 4; // size in bytes
        tick();

        cpu_recv_instr = INSTR_NOP;
        for (i = 0; i < num_words; i = i + 1) begin
            cpu_in_data = words[i];
            tick();
        end
    endtask

    // -------------------------------------------------------
    // gpu_launch_kernel: replicates gpuLaunchKernel()
    //   Sends KERNEL_LAUNCH with kernel address and 0 params.
    //   Waits for cpu_out_ack (kernel finished).
    // -------------------------------------------------------
    task gpu_launch_kernel(input [31:0] kernel_addr);
        integer done;
        integer cycle_count;

        cpu_recv_instr = INSTR_KERNEL_LAUNCH;
        tick();

        cpu_in_data = kernel_addr;
        tick();

        cpu_in_data = 0; // num kernel params = 0
        tick();

        cpu_recv_instr = INSTR_NOP;

        // Wait for ack (kernel finished)
        done = 0;
        for (cycle_count = 0; cycle_count < 10000 && !done; cycle_count = cycle_count + 1) begin
            tick();
            if (cpu_out_ack) begin
                done = 1;
                $display("  kernel finished after ~%0d cycles", cycle_count);
            end
        end

        if (!done) begin
            $display("FAIL: kernel launch timeout");
            $finish;
        end
    endtask

    // -------------------------------------------------------
    // gpu_copy_from_device: replicates gpuCopyFromDevice()
    //   Sends COPY_FROM_GPU, src addr, size.
    //   Reads words back one at a time via cpu_out_ack/cpu_out_data.
    // -------------------------------------------------------
    reg [31:0] readback_buf [0:15];

    task gpu_copy_from_device(input [31:0] src_addr, input integer num_words);
        integer i;
        integer done;
        integer cycle_count;

        cpu_recv_instr = INSTR_COPY_FROM_GPU;
        tick();

        cpu_in_data = src_addr;
        tick();

        cpu_in_data = num_words * 4; // size in bytes
        tick();

        cpu_recv_instr = INSTR_NOP;

        i = 0;
        for (cycle_count = 0; cycle_count < 5000 && i < num_words; cycle_count = cycle_count + 1) begin
            if (cpu_out_ack) begin
                readback_buf[i] = cpu_out_data;
                i = i + 1;
            end
            tick();
        end

        if (i < num_words) begin
            $display("FAIL: copy_from_device timeout (got %0d of %0d words)", i, num_words);
            $finish;
        end
    endtask

    // -------------------------------------------------------
    // Main test sequence
    // -------------------------------------------------------
    integer test_num;

    initial begin
        clk = 0;
        rst = 0;
        cpu_recv_instr = INSTR_NOP;
        cpu_in_data = 0;

        // Reset
        tick(); tick();
        rst = 1;
        tick(); tick();

        // =======================================================
        // Test 1: Full cycle — copy program, launch, read results
        //   4 cores with base_thread_id = 0
        //   Expect: mem[0]=0, mem[4]=1, mem[8]=2, mem[12]=3
        // =======================================================
        test_num = 1;
        $display("--- Test %0d: Full cycle with 4 cores ---", test_num);

        // Step 1: Copy program to GPU address 128
        $display("  Copying program to address 128...");
        gpu_copy_to_device(
            128,  // dest_addr
            5,    // num_words (5 instructions)
            32'h00229313,  // slli x6, x5, 2
            32'h00532023,  // sw   x5, 0(x6)
            32'h000F43B7,  // lui  x7, 0xF4
            32'h24438393,  // addi x7, x7, 0x244
            32'h0003A023   // sw   x0, 0(x7) — halt
        );
        $display("  Program copied.");

        // Step 2: Launch kernel at address 128
        $display("  Launching kernel...");
        gpu_launch_kernel(128);

        // Step 3: Read back 4 words from address 0
        $display("  Reading results...");
        gpu_copy_from_device(0, 4);

        // Step 4: Verify
        $display("  Results: [%0d, %0d, %0d, %0d] (expect [0, 1, 2, 3])",
            readback_buf[0], readback_buf[1], readback_buf[2], readback_buf[3]);

        if (readback_buf[0] !== 0 || readback_buf[1] !== 1 ||
            readback_buf[2] !== 2 || readback_buf[3] !== 3) begin
            $display("FAIL test %0d: unexpected values", test_num);
            $finish;
        end
        $display("PASS test %0d", test_num);
        tick(); tick();

        // =======================================================
        // Test 2: Second launch to verify controller returns to IDLE
        //   Copy a simple "write 42 to addr 200, halt" program
        //   Single-thread behavior (all 4 cores write same thing)
        // =======================================================
        test_num = 2;
        $display("--- Test %0d: Second kernel launch (controller reuse) ---", test_num);

        // Program: li x6, 200; li x7, 42; sw x7, 0(x6); halt
        // Encoded as:
        //   addi x6, x0, 200     = 0x0C800313
        //   addi x7, x0, 42      = 0x02A00393
        //   sw   x7, 0(x6)       = 0x00732023
        //   lui  x8, 0xF4        = 0x000F4437
        //   addi x8, x8, 0x244   = 0x24440413
        //   sw   x0, 0(x8)       = 0x00042023
        // That's 6 instructions = 24 bytes. But our copy task only handles 5.
        // Simpler: reuse the CP-3 program at addr 256, read from different addrs.

        // Actually, let's just re-launch the SAME program at addr 128.
        // The results at 0-12 get overwritten with the same values (idempotent).
        $display("  Re-launching same kernel...");
        gpu_launch_kernel(128);

        $display("  Reading results...");
        gpu_copy_from_device(0, 4);

        $display("  Results: [%0d, %0d, %0d, %0d] (expect [0, 1, 2, 3])",
            readback_buf[0], readback_buf[1], readback_buf[2], readback_buf[3]);

        if (readback_buf[0] !== 0 || readback_buf[1] !== 1 ||
            readback_buf[2] !== 2 || readback_buf[3] !== 3) begin
            $display("FAIL test %0d: unexpected values", test_num);
            $finish;
        end
        $display("PASS test %0d", test_num);

        // =======================================================
        // Test 3: Verify test 1 data at addrs 0-12 survived
        //   (redundant with test 2, but confirms COPY_FROM_GPU works)
        // =======================================================
        test_num = 3;
        $display("--- Test %0d: Verify data persists after 2 launches ---", test_num);
        gpu_copy_from_device(0, 4);

        if (readback_buf[0] !== 0 || readback_buf[1] !== 1 ||
            readback_buf[2] !== 2 || readback_buf[3] !== 3) begin
            $display("FAIL test %0d", test_num);
            $finish;
        end
        $display("PASS test %0d", test_num);

        // =======================================================
        $display("");
        $display("ALL 3 TESTS PASSED");
        $finish;
    end

    // Timeout watchdog
    initial begin
        #1000000;
        $display("FAIL: global timeout");
        $finish;
    end
endmodule
