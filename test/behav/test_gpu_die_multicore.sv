/*
test_gpu_die_multicore.sv

CP-4+5+6 test: exercises the full gpu_die stack.

Tests 1-3: from CP-4+5 (single batch)
Test 4: CP-6 batching — 8 threads in 2 batches of 4
Test 5: CP-6 batching — 12 threads in 3 batches of 4
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
    localparam INSTR_NOP              = 0;
    localparam INSTR_COPY_TO_GPU      = 1;
    localparam INSTR_COPY_FROM_GPU    = 2;
    localparam INSTR_KERNEL_LAUNCH    = 3;
    localparam INSTR_SET_THREAD_BASE  = 4;  // NEW for CP-6

    // -------------------------------------------------------
    // gpu_set_base_thread_id: NEW for CP-6
    // -------------------------------------------------------
    task gpu_set_base_thread_id(input [31:0] base);
        cpu_recv_instr = INSTR_SET_THREAD_BASE;
        tick();
        cpu_in_data = base;
        tick();
        cpu_recv_instr = INSTR_NOP;
        tick();
    endtask

    // -------------------------------------------------------
    // gpu_copy_to_device
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
        cpu_in_data = num_words * 4;
        tick();
        cpu_recv_instr = INSTR_NOP;
        for (i = 0; i < num_words; i = i + 1) begin
            cpu_in_data = words[i];
            tick();
        end
    endtask

    // -------------------------------------------------------
    // gpu_launch_kernel
    // -------------------------------------------------------
    task gpu_launch_kernel(input [31:0] kernel_addr);
        integer done, cycle_count;

        cpu_recv_instr = INSTR_KERNEL_LAUNCH;
        tick();
        cpu_in_data = kernel_addr;
        tick();
        cpu_in_data = 0;
        tick();
        cpu_recv_instr = INSTR_NOP;

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
    // gpu_copy_from_device
    // -------------------------------------------------------
    reg [31:0] readback_buf [0:15];

    task gpu_copy_from_device(input [31:0] src_addr, input integer num_words);
        integer i, cycle_count;

        cpu_recv_instr = INSTR_COPY_FROM_GPU;
        tick();
        cpu_in_data = src_addr;
        tick();
        cpu_in_data = num_words * 4;
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
    integer j;

    initial begin
        clk = 0; rst = 0;
        cpu_recv_instr = INSTR_NOP;
        cpu_in_data = 0;

        tick(); tick();
        rst = 1;
        tick(); tick();

        // =======================================================
        // Load the test program at addr 128 (same as CP-3/4/5)
        //   slli x6, x5, 2    ; x6 = tid * 4
        //   sw   x5, 0(x6)     ; mem[tid*4] = tid
        //   lui  x7, 0xF4      ; x7 = 0xF4000
        //   addi x7, x7, 0x244 ; x7 = 0xF4244
        //   sw   x0, 0(x7)      ; halt
        // =======================================================
        $display("Loading program to address 128...");
        gpu_copy_to_device(128, 5,
            32'h00229313, 32'h00532023, 32'h000F43B7,
            32'h24438393, 32'h0003A023);
        $display("Program loaded.");
        $display("");

        // =======================================================
        // Test 1: Single batch (base=0) — from CP-4+5
        // =======================================================
        test_num = 1;
        $display("--- Test %0d: Single batch, base=0 ---", test_num);
        gpu_set_base_thread_id(0);
        gpu_launch_kernel(128);
        gpu_copy_from_device(0, 4);
        $display("  Results: [%0d, %0d, %0d, %0d] (expect [0,1,2,3])",
            readback_buf[0], readback_buf[1], readback_buf[2], readback_buf[3]);
        if (readback_buf[0]!==0 || readback_buf[1]!==1 ||
            readback_buf[2]!==2 || readback_buf[3]!==3) begin
            $display("FAIL test %0d", test_num); $finish;
        end
        $display("PASS test %0d", test_num);
        tick(); tick();

        // =======================================================
        // Test 2: Single batch (base=10) — verify base_thread_id works
        // =======================================================
        test_num = 2;
        $display("--- Test %0d: Single batch, base=10 ---", test_num);
        gpu_set_base_thread_id(10);
        gpu_launch_kernel(128);
        gpu_copy_from_device(40, 4);  // tid*4 = 40,44,48,52
        $display("  Results: [%0d, %0d, %0d, %0d] (expect [10,11,12,13])",
            readback_buf[0], readback_buf[1], readback_buf[2], readback_buf[3]);
        if (readback_buf[0]!==10 || readback_buf[1]!==11 ||
            readback_buf[2]!==12 || readback_buf[3]!==13) begin
            $display("FAIL test %0d", test_num); $finish;
        end
        $display("PASS test %0d", test_num);
        tick(); tick();

        // =======================================================
        // Test 3: Controller reuse — re-launch works
        // =======================================================
        test_num = 3;
        $display("--- Test %0d: Controller reuse (re-launch base=0) ---", test_num);
        gpu_set_base_thread_id(0);
        gpu_launch_kernel(128);
        gpu_copy_from_device(0, 4);
        if (readback_buf[0]!==0 || readback_buf[1]!==1 ||
            readback_buf[2]!==2 || readback_buf[3]!==3) begin
            $display("FAIL test %0d", test_num); $finish;
        end
        $display("PASS test %0d", test_num);
        tick(); tick();

        // =======================================================
        // Test 4: BATCHING — 8 threads in 2 batches of 4
        //   Batch 1: base=0 → threads 0,1,2,3 → write to addrs 0,4,8,12
        //   Batch 2: base=4 → threads 4,5,6,7 → write to addrs 16,20,24,28
        //   Read all 8 words and verify
        // =======================================================
        test_num = 4;
        $display("--- Test %0d: BATCHING — 8 threads in 2 batches ---", test_num);

        $display("  Batch 1 (base=0)...");
        gpu_set_base_thread_id(0);
        gpu_launch_kernel(128);

        $display("  Batch 2 (base=4)...");
        gpu_set_base_thread_id(4);
        gpu_launch_kernel(128);

        $display("  Reading 8 words from addr 0...");
        gpu_copy_from_device(0, 8);
        $display("  Results: [%0d, %0d, %0d, %0d, %0d, %0d, %0d, %0d]",
            readback_buf[0], readback_buf[1], readback_buf[2], readback_buf[3],
            readback_buf[4], readback_buf[5], readback_buf[6], readback_buf[7]);
        $display("  Expect:  [0, 1, 2, 3, 4, 5, 6, 7]");

        for (j = 0; j < 8; j = j + 1) begin
            if (readback_buf[j] !== j) begin
                $display("FAIL test %0d: readback_buf[%0d] = %0d, expected %0d",
                    test_num, j, readback_buf[j], j);
                $finish;
            end
        end
        $display("PASS test %0d", test_num);
        tick(); tick();

        // =======================================================
        // Test 5: BATCHING — 12 threads in 3 batches of 4
        // =======================================================
        test_num = 5;
        $display("--- Test %0d: BATCHING — 12 threads in 3 batches ---", test_num);

        gpu_set_base_thread_id(0);
        gpu_launch_kernel(128);

        gpu_set_base_thread_id(4);
        gpu_launch_kernel(128);

        gpu_set_base_thread_id(8);
        gpu_launch_kernel(128);

        gpu_copy_from_device(0, 12);
        $display("  Results: [%0d,%0d,%0d,%0d, %0d,%0d,%0d,%0d, %0d,%0d,%0d,%0d]",
            readback_buf[0], readback_buf[1], readback_buf[2], readback_buf[3],
            readback_buf[4], readback_buf[5], readback_buf[6], readback_buf[7],
            readback_buf[8], readback_buf[9], readback_buf[10], readback_buf[11]);

        for (j = 0; j < 12; j = j + 1) begin
            if (readback_buf[j] !== j) begin
                $display("FAIL test %0d: readback_buf[%0d] = %0d, expected %0d",
                    test_num, j, readback_buf[j], j);
                $finish;
            end
        end
        $display("PASS test %0d", test_num);

        // =======================================================
        $display("");
        $display("ALL 5 TESTS PASSED");
        $finish;
    end

    initial begin
        #2000000;
        $display("FAIL: global timeout");
        $finish;
    end
endmodule
