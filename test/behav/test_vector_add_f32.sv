/*
 * test_vector_add_f32.sv — Test float vector addition on VeriGPU hardware.
 *
 * This is the float equivalent of test_vector_add.sv (which tested integers).
 * The kernel uses FADD.S (added in CP-1 of the original roadmap) to add
 * IEEE 754 float32 values in the RISC-V cores.
 *
 * Tests:
 *   1. 4 float additions (1 batch, 4 cores)
 *   2. 8 float additions (2 batches, 4 cores each)
 *   3. Mixed positive/negative floats
 *   4. Idempotency (re-run without reload)
 *
 * Memory layout:
 *   Kernel:  address 128
 *   a[]:     address 256
 *   b[]:     address 320
 *   out[]:   address 384
 */

module test_vector_add_f32();

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

    localparam INSTR_NOP              = 0;
    localparam INSTR_COPY_TO_GPU      = 1;
    localparam INSTR_COPY_FROM_GPU    = 2;
    localparam INSTR_KERNEL_LAUNCH    = 3;
    localparam INSTR_SET_THREAD_BASE  = 4;

    // -------------------------------------------------------
    // gpu_copy_to_device (up to 16 words)
    // -------------------------------------------------------
    reg [31:0] upload_buf [0:15];

    task gpu_copy_to_device_buf(input [31:0] dest_addr, input integer num_words);
        integer i;
        cpu_recv_instr = INSTR_COPY_TO_GPU;
        tick();
        cpu_in_data = dest_addr;
        tick();
        cpu_in_data = num_words * 4;
        tick();
        cpu_recv_instr = INSTR_NOP;
        for (i = 0; i < num_words; i = i + 1) begin
            cpu_in_data = upload_buf[i];
            tick();
        end
    endtask

    // -------------------------------------------------------
    // gpu_set_base_thread_id
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
            $display("FAIL: copy_from_device timeout (got %0d of %0d)", i, num_words);
            $finish;
        end
    endtask

    // -------------------------------------------------------
    // Batched launch
    // -------------------------------------------------------
    task batched_launch(input [31:0] kernel_addr, input integer total_threads);
        integer batch_start;
        for (batch_start = 0; batch_start < total_threads; batch_start = batch_start + 4) begin
            gpu_set_base_thread_id(batch_start);
            gpu_launch_kernel(kernel_addr);
        end
    endtask

    // -------------------------------------------------------
    // Upload array of up to 8 words
    // -------------------------------------------------------
    task upload_array(input [31:0] addr, input integer n,
                      input [31:0] v0, input [31:0] v1,
                      input [31:0] v2, input [31:0] v3,
                      input [31:0] v4, input [31:0] v5,
                      input [31:0] v6, input [31:0] v7);
        upload_buf[0] = v0; upload_buf[1] = v1;
        upload_buf[2] = v2; upload_buf[3] = v3;
        upload_buf[4] = v4; upload_buf[5] = v5;
        upload_buf[6] = v6; upload_buf[7] = v7;
        gpu_copy_to_device_buf(addr, n);
    endtask

    task clear_output(input integer n);
        integer ci;
        for (ci = 0; ci < n; ci = ci + 1) upload_buf[ci] = 0;
        gpu_copy_to_device_buf(384, n);
    endtask

    // -------------------------------------------------------
    // Load vector_add_f32 kernel at address 128
    //
    // This is identical to vector_add EXCEPT instruction [5]
    // uses FADD.S (opcode 1010011) instead of ADD (opcode 0110011).
    //
    // slli    x6, x5, 2         ; x6 = tid * 4
    // addi    x7, x6, 256       ; x7 = &a[tid]
    // lw      x8, 0(x7)         ; x8 = a[tid] (float bits)
    // addi    x7, x6, 320       ; x7 = &b[tid]
    // lw      x9, 0(x7)         ; x9 = b[tid] (float bits)
    // fadd.s  x10, x8, x9       ; x10 = a[tid] + b[tid]  ← FLOAT ADD
    // addi    x7, x6, 384       ; x7 = &out[tid]
    // sw      x10, 0(x7)        ; out[tid] = result
    // lui     x11, 0xF4         ; \
    // addi    x11, x11, 0x244   ;  > halt sequence
    // sw      x0, 0(x11)        ; /
    // -------------------------------------------------------
    task load_vector_add_f32_kernel();
        upload_buf[0]  = 32'h00229313;  // slli  x6, x5, 2
        upload_buf[1]  = 32'h10030393;  // addi  x7, x6, 256
        upload_buf[2]  = 32'h0003A403;  // lw    x8, 0(x7)
        upload_buf[3]  = 32'h14030393;  // addi  x7, x6, 320
        upload_buf[4]  = 32'h0003A483;  // lw    x9, 0(x7)
        upload_buf[5]  = 32'h00940553;  // fadd.s x10, x8, x9  ← 0x553 not 0x533
        upload_buf[6]  = 32'h18030393;  // addi  x7, x6, 384
        upload_buf[7]  = 32'h00A3A023;  // sw    x10, 0(x7)
        upload_buf[8]  = 32'h000F45B7;  // lui   x11, 0xF4
        upload_buf[9]  = 32'h24458593;  // addi  x11, x11, 0x244
        upload_buf[10] = 32'h0005A023;  // sw    x0, 0(x11)
        gpu_copy_to_device_buf(128, 11);
    endtask

    // -------------------------------------------------------
    // IEEE 754 float constants
    // -------------------------------------------------------
    // Positive values
    localparam F_1_0  = 32'h3F800000;  //  1.0
    localparam F_2_0  = 32'h40000000;  //  2.0
    localparam F_3_0  = 32'h40400000;  //  3.0
    localparam F_4_0  = 32'h40800000;  //  4.0
    localparam F_5_0  = 32'h40A00000;  //  5.0
    localparam F_6_0  = 32'h40C00000;  //  6.0
    localparam F_7_0  = 32'h40E00000;  //  7.0
    localparam F_8_0  = 32'h41000000;  //  8.0
    localparam F_10_0 = 32'h41200000;  // 10.0
    localparam F_20_0 = 32'h41A00000;  // 20.0
    localparam F_30_0 = 32'h41F00000;  // 30.0
    localparam F_40_0 = 32'h42200000;  // 40.0
    localparam F_50_0 = 32'h42480000;  // 50.0
    localparam F_60_0 = 32'h42700000;  // 60.0
    localparam F_70_0 = 32'h428C0000;  // 70.0
    localparam F_80_0 = 32'h42A00000;  // 80.0

    // Expected results
    localparam F_11_0 = 32'h41300000;  // 11.0
    localparam F_22_0 = 32'h41B00000;  // 22.0
    localparam F_33_0 = 32'h42040000;  // 33.0
    localparam F_44_0 = 32'h42300000;  // 44.0
    localparam F_55_0 = 32'h425C0000;  // 55.0
    localparam F_66_0 = 32'h42840000;  // 66.0
    localparam F_77_0 = 32'h429A0000;  // 77.0
    localparam F_88_0 = 32'h42B00000;  // 88.0

    // Negative values (for test 3)
    localparam F_N2_0  = 32'hC0000000;  // -2.0
    localparam F_N4_0  = 32'hC0800000;  // -4.0
    localparam F_N10_0 = 32'hC1200000;  // -10.0
    localparam F_N30_0 = 32'hC1F00000;  // -30.0

    // Expected results for test 3
    localparam F_N9_0  = 32'hC1100000;  // -9.0  (1.0 + -10.0)
    localparam F_18_0  = 32'h41900000;  // 18.0  (-2.0 + 20.0)
    localparam F_N27_0 = 32'hC1D80000;  // -27.0 (3.0 + -30.0)
    localparam F_36_0  = 32'h42100000;  // 36.0  (-4.0 + 40.0)

    // -------------------------------------------------------
    // Main test sequence
    // -------------------------------------------------------
    integer test_num, j, pass;
    reg [31:0] expected [0:7];

    initial begin
        clk = 0; rst = 0;
        cpu_recv_instr = INSTR_NOP;
        cpu_in_data = 0;

        tick(); tick();
        rst = 1;
        tick(); tick();

        $display("");
        $display("=== CP-2 (HW Roadmap): Float Vector Addition on RISC-V Cores ===");
        $display("");

        // =======================================================
        // Test 1: 4 float additions (1 batch)
        //   a = [1.0, 2.0, 3.0, 4.0]
        //   b = [10.0, 20.0, 30.0, 40.0]
        //   expected = [11.0, 22.0, 33.0, 44.0]
        // =======================================================
        test_num = 1;
        $display("--- Test %0d: 4 float adds (1 batch) ---", test_num);

        load_vector_add_f32_kernel();
        upload_array(256, 4, F_1_0, F_2_0, F_3_0, F_4_0,     0, 0, 0, 0);
        upload_array(320, 4, F_10_0, F_20_0, F_30_0, F_40_0,  0, 0, 0, 0);
        clear_output(4);

        batched_launch(128, 4);

        gpu_copy_from_device(384, 4);

        expected[0] = F_11_0; expected[1] = F_22_0;
        expected[2] = F_33_0; expected[3] = F_44_0;

        $display("  out = [%h, %h, %h, %h]",
            readback_buf[0], readback_buf[1], readback_buf[2], readback_buf[3]);
        $display("  exp = [%h, %h, %h, %h]  (11.0, 22.0, 33.0, 44.0)",
            expected[0], expected[1], expected[2], expected[3]);

        pass = 1;
        for (j = 0; j < 4; j = j + 1) begin
            if (readback_buf[j] !== expected[j]) begin
                $display("  MISMATCH at [%0d]: got %h, expected %h", j, readback_buf[j], expected[j]);
                pass = 0;
            end
        end
        if (!pass) begin $display("FAIL test %0d", test_num); $finish; end
        $display("PASS test %0d", test_num);
        $display("");
        tick(); tick();

        // =======================================================
        // Test 2: 8 float additions (2 batches)
        //   a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        //   b = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
        //   expected = [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0]
        // =======================================================
        test_num = 2;
        $display("--- Test %0d: 8 float adds (2 batches) ---", test_num);

        load_vector_add_f32_kernel();
        upload_array(256, 8, F_1_0,  F_2_0,  F_3_0,  F_4_0,  F_5_0,  F_6_0,  F_7_0,  F_8_0);
        upload_array(320, 8, F_10_0, F_20_0, F_30_0, F_40_0, F_50_0, F_60_0, F_70_0, F_80_0);
        clear_output(8);

        batched_launch(128, 8);

        gpu_copy_from_device(384, 8);

        expected[0] = F_11_0; expected[1] = F_22_0;
        expected[2] = F_33_0; expected[3] = F_44_0;
        expected[4] = F_55_0; expected[5] = F_66_0;
        expected[6] = F_77_0; expected[7] = F_88_0;

        $display("  out = [%h, %h, %h, %h, %h, %h, %h, %h]",
            readback_buf[0], readback_buf[1], readback_buf[2], readback_buf[3],
            readback_buf[4], readback_buf[5], readback_buf[6], readback_buf[7]);

        pass = 1;
        for (j = 0; j < 8; j = j + 1) begin
            if (readback_buf[j] !== expected[j]) begin
                $display("  MISMATCH at [%0d]: got %h, expected %h", j, readback_buf[j], expected[j]);
                pass = 0;
            end
        end
        if (!pass) begin $display("FAIL test %0d", test_num); $finish; end
        $display("PASS test %0d", test_num);
        $display("");
        tick(); tick();

        // =======================================================
        // Test 3: Mixed positive/negative floats
        //   a = [1.0, -2.0, 3.0, -4.0]
        //   b = [-10.0, 20.0, -30.0, 40.0]
        //   expected = [-9.0, 18.0, -27.0, 36.0]
        // =======================================================
        test_num = 3;
        $display("--- Test %0d: mixed positive/negative floats ---", test_num);

        load_vector_add_f32_kernel();
        upload_array(256, 4, F_1_0,   F_N2_0,  F_3_0,   F_N4_0,   0, 0, 0, 0);
        upload_array(320, 4, F_N10_0, F_20_0,  F_N30_0, F_40_0,   0, 0, 0, 0);
        clear_output(4);

        batched_launch(128, 4);

        gpu_copy_from_device(384, 4);

        expected[0] = F_N9_0;  expected[1] = F_18_0;
        expected[2] = F_N27_0; expected[3] = F_36_0;

        $display("  out = [%h, %h, %h, %h]",
            readback_buf[0], readback_buf[1], readback_buf[2], readback_buf[3]);
        $display("  exp = [%h, %h, %h, %h]  (-9.0, 18.0, -27.0, 36.0)",
            expected[0], expected[1], expected[2], expected[3]);

        pass = 1;
        for (j = 0; j < 4; j = j + 1) begin
            if (readback_buf[j] !== expected[j]) begin
                $display("  MISMATCH at [%0d]: got %h, expected %h", j, readback_buf[j], expected[j]);
                pass = 0;
            end
        end
        if (!pass) begin $display("FAIL test %0d", test_num); $finish; end
        $display("PASS test %0d", test_num);
        $display("");
        tick(); tick();

        // =======================================================
        // Test 4: Idempotency — re-run without reloading kernel
        //   Same data as test 1, just re-launch
        // =======================================================
        test_num = 4;
        $display("--- Test %0d: Idempotency (re-run) ---", test_num);

        upload_array(256, 4, F_1_0, F_2_0, F_3_0, F_4_0,     0, 0, 0, 0);
        upload_array(320, 4, F_10_0, F_20_0, F_30_0, F_40_0,  0, 0, 0, 0);
        clear_output(4);

        batched_launch(128, 4);

        gpu_copy_from_device(384, 4);

        expected[0] = F_11_0; expected[1] = F_22_0;
        expected[2] = F_33_0; expected[3] = F_44_0;

        pass = 1;
        for (j = 0; j < 4; j = j + 1) begin
            if (readback_buf[j] !== expected[j]) begin
                $display("  MISMATCH at [%0d]: got %h, expected %h", j, readback_buf[j], expected[j]);
                pass = 0;
            end
        end
        if (!pass) begin $display("FAIL test %0d", test_num); $finish; end
        $display("PASS test %0d", test_num);
        $display("");

        // =======================================================
        // Summary
        // =======================================================
        $display("========================================");
        $display("  ALL 4 TESTS PASSED");
        $display("  FADD.S works in multicore context!");
        $display("========================================");
        $finish;
    end

endmodule
