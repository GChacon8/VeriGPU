/*
test_vector_add.sv

CP-7 FINAL TEST: end-to-end vector_add with multi-core execution.

This is the ultimate proof that the multicore pipeline works.
It runs a REAL kernel (vector_add) that does useful work across
multiple threads, using the complete hardware stack:
  host protocol → gpu_controller → compute_unit(4 cores) → arbiter → memory

Memory layout:
  128-172:  Program (11 instructions)
  256-287:  Array a (up to 8 words)
  320-351:  Array b (up to 8 words)
  384-415:  Array out (up to 8 words)

Kernel (assembly):
  slli  x6, x5, 2         ; x6 = tid * 4  (byte offset)
  addi  x7, x6, 256       ; x7 = &a[tid]
  lw    x8, 0(x7)          ; x8 = a[tid]
  addi  x7, x6, 320       ; x7 = &b[tid]
  lw    x9, 0(x7)          ; x9 = b[tid]
  add   x10, x8, x9       ; x10 = a[tid] + b[tid]
  addi  x7, x6, 384       ; x7 = &out[tid]
  sw    x10, 0(x7)         ; out[tid] = x10
  lui   x11, 0xF4          ; x11 = 0xF4000
  addi  x11, x11, 0x244   ; x11 = 0xF4244 (halt)
  sw    x0, 0(x11)          ; halt

Tests:
  1. vector_add with 4 threads (1 batch)
  2. vector_add with 8 threads (2 batches)
  3. vector_add with different data (verify not stale)
  4. Idempotency: re-run same kernel, same results
  5. vector_scale: different kernel (out[tid] = a[tid] * 3)
  6. Stress: 12 threads in 3 batches
*/

module test_vector_add();

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
    // Flexible gpu_copy_to_device (up to 16 words)
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
            if (cpu_out_ack) done = 1;
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
    // Batched launch: runs total_threads across batches of 4
    // -------------------------------------------------------
    task batched_launch(input [31:0] kernel_addr, input integer total_threads);
        integer batch_start;
        for (batch_start = 0; batch_start < total_threads; batch_start = batch_start + 4) begin
            gpu_set_base_thread_id(batch_start);
            gpu_launch_kernel(kernel_addr);
        end
    endtask

    // -------------------------------------------------------
    // Load vector_add kernel at address 128 (11 instructions)
    //
    // slli  x6, x5, 2         ; x6 = tid * 4
    // addi  x7, x6, 256       ; x7 = &a[tid]
    // lw    x8, 0(x7)          ; x8 = a[tid]
    // addi  x7, x6, 320       ; x7 = &b[tid]
    // lw    x9, 0(x7)          ; x9 = b[tid]
    // add   x10, x8, x9       ; x10 = a[tid] + b[tid]
    // addi  x7, x6, 384       ; x7 = &out[tid]
    // sw    x10, 0(x7)         ; out[tid] = x10
    // lui   x11, 0xF4
    // addi  x11, x11, 0x244
    // sw    x0, 0(x11)          ; halt
    // -------------------------------------------------------
    task load_vector_add_kernel();
        upload_buf[0]  = 32'h00229313;  // slli  x6, x5, 2
        upload_buf[1]  = 32'h10030393;  // addi  x7, x6, 256
        upload_buf[2]  = 32'h0003A403;  // lw    x8, 0(x7)
        upload_buf[3]  = 32'h14030393;  // addi  x7, x6, 320
        upload_buf[4]  = 32'h0003A483;  // lw    x9, 0(x7)
        upload_buf[5]  = 32'h00940533;  // add   x10, x8, x9
        upload_buf[6]  = 32'h18030393;  // addi  x7, x6, 384
        upload_buf[7]  = 32'h00A3A023;  // sw    x10, 0(x7)
        upload_buf[8]  = 32'h000F45B7;  // lui   x11, 0xF4
        upload_buf[9]  = 32'h24458593;  // addi  x11, x11, 0x244
        upload_buf[10] = 32'h0005A023;  // sw    x0, 0(x11)
        gpu_copy_to_device_buf(128, 11);
    endtask

    // -------------------------------------------------------
    // Load vector_scale kernel at address 192 (12 instructions)
    //
    // out[tid] = a[tid] * 3
    //
    // slli  x6, x5, 2
    // addi  x7, x6, 256       ; &a[tid]
    // lw    x8, 0(x7)          ; x8 = a[tid]
    // addi  x9, x0, 3         ; x9 = 3
    // mul   x10, x8, x9       ; x10 = a[tid] * 3
    // addi  x7, x6, 384       ; &out[tid]
    // sw    x10, 0(x7)         ; out[tid] = x10
    // lui   x11, 0xF4
    // addi  x11, x11, 0x244
    // sw    x0, 0(x11)
    // -------------------------------------------------------
    task load_vector_scale_kernel();
        upload_buf[0]  = 32'h00229313;  // slli  x6, x5, 2
        upload_buf[1]  = 32'h10030393;  // addi  x7, x6, 256
        upload_buf[2]  = 32'h0003A403;  // lw    x8, 0(x7)
        upload_buf[3]  = 32'h00300493;  // addi  x9, x0, 3
        upload_buf[4]  = 32'h02940533;  // mul   x10, x8, x9
        upload_buf[5]  = 32'h18030393;  // addi  x7, x6, 384
        upload_buf[6]  = 32'h00A3A023;  // sw    x10, 0(x7)
        upload_buf[7]  = 32'h000F45B7;  // lui   x11, 0xF4
        upload_buf[8]  = 32'h24458593;  // addi  x11, x11, 0x244
        upload_buf[9]  = 32'h0005A023;  // sw    x0, 0(x11)
        gpu_copy_to_device_buf(192, 10);
    endtask

    // -------------------------------------------------------
    // Upload array data
    // -------------------------------------------------------
    task upload_array(input [31:0] addr, input integer n,
                      input [31:0] v0,  input [31:0] v1,
                      input [31:0] v2,  input [31:0] v3,
                      input [31:0] v4,  input [31:0] v5,
                      input [31:0] v6,  input [31:0] v7);
        upload_buf[0] = v0; upload_buf[1] = v1;
        upload_buf[2] = v2; upload_buf[3] = v3;
        upload_buf[4] = v4; upload_buf[5] = v5;
        upload_buf[6] = v6; upload_buf[7] = v7;
        gpu_copy_to_device_buf(addr, n);
    endtask

    // Clear output region
    task clear_output(input integer n);
        integer ci;
        for (ci = 0; ci < n; ci = ci + 1) upload_buf[ci] = 0;
        gpu_copy_to_device_buf(384, n);
    endtask

    // -------------------------------------------------------
    // Main test sequence
    // -------------------------------------------------------
    integer test_num, j, pass;

    initial begin
        clk = 0; rst = 0;
        cpu_recv_instr = INSTR_NOP;
        cpu_in_data = 0;

        tick(); tick();
        rst = 1;
        tick(); tick();

        // =======================================================
        // Test 1: vector_add, 4 threads (1 batch)
        //   a = {10, 20, 30, 40}
        //   b = {1, 2, 3, 4}
        //   expect out = {11, 22, 33, 44}
        // =======================================================
        test_num = 1;
        $display("=== Test %0d: vector_add, 4 threads (1 batch) ===", test_num);

        load_vector_add_kernel();
        upload_array(256, 4,  10, 20, 30, 40,  0, 0, 0, 0);  // a
        upload_array(320, 4,   1,  2,  3,  4,  0, 0, 0, 0);  // b
        clear_output(4);

        batched_launch(128, 4);

        gpu_copy_from_device(384, 4);
        $display("  out = [%0d, %0d, %0d, %0d]", readback_buf[0], readback_buf[1], readback_buf[2], readback_buf[3]);
        $display("  exp = [11, 22, 33, 44]");

        pass = (readback_buf[0]==11 && readback_buf[1]==22 && readback_buf[2]==33 && readback_buf[3]==44);
        if (!pass) begin $display("FAIL test %0d", test_num); $finish; end
        $display("PASS test %0d", test_num);
        $display("");
        tick(); tick();

        // =======================================================
        // Test 2: vector_add, 8 threads (2 batches)
        //   a = {100, 200, 300, 400, 500, 600, 700, 800}
        //   b = {1, 2, 3, 4, 5, 6, 7, 8}
        //   expect out = {101, 202, 303, 404, 505, 606, 707, 808}
        // =======================================================
        test_num = 2;
        $display("=== Test %0d: vector_add, 8 threads (2 batches) ===", test_num);

        load_vector_add_kernel();
        upload_array(256, 8,  100, 200, 300, 400, 500, 600, 700, 800);
        upload_array(320, 8,    1,   2,   3,   4,   5,   6,   7,   8);
        clear_output(8);

        batched_launch(128, 8);

        gpu_copy_from_device(384, 8);
        $display("  out = [%0d, %0d, %0d, %0d, %0d, %0d, %0d, %0d]",
            readback_buf[0], readback_buf[1], readback_buf[2], readback_buf[3],
            readback_buf[4], readback_buf[5], readback_buf[6], readback_buf[7]);

        pass = 1;
        for (j = 0; j < 8; j = j + 1) begin
            if (readback_buf[j] !== (j + 1) * 100 + (j + 1)) begin
                $display("  MISMATCH at [%0d]: got %0d, expected %0d", j, readback_buf[j], (j+1)*100 + (j+1));
                pass = 0;
            end
        end
        if (!pass) begin $display("FAIL test %0d", test_num); $finish; end
        $display("PASS test %0d", test_num);
        $display("");
        tick(); tick();

        // =======================================================
        // Test 3: vector_add with different data (not stale)
        //   a = {5, 10, 15, 20}
        //   b = {100, 200, 300, 400}
        //   expect out = {105, 210, 315, 420}
        // =======================================================
        test_num = 3;
        $display("=== Test %0d: vector_add, different data ===", test_num);

        load_vector_add_kernel();
        upload_array(256, 4,  5, 10, 15, 20,  0, 0, 0, 0);
        upload_array(320, 4,  100, 200, 300, 400,  0, 0, 0, 0);
        clear_output(4);

        batched_launch(128, 4);

        gpu_copy_from_device(384, 4);
        $display("  out = [%0d, %0d, %0d, %0d]", readback_buf[0], readback_buf[1], readback_buf[2], readback_buf[3]);

        pass = (readback_buf[0]==105 && readback_buf[1]==210 && readback_buf[2]==315 && readback_buf[3]==420);
        if (!pass) begin $display("FAIL test %0d", test_num); $finish; end
        $display("PASS test %0d", test_num);
        $display("");
        tick(); tick();

        // =======================================================
        // Test 4: Idempotency — re-run same kernel, same results
        // =======================================================
        test_num = 4;
        $display("=== Test %0d: Idempotency (re-run) ===", test_num);

        // Don't reload anything — just re-launch
        clear_output(4);
        batched_launch(128, 4);
        gpu_copy_from_device(384, 4);

        pass = (readback_buf[0]==105 && readback_buf[1]==210 && readback_buf[2]==315 && readback_buf[3]==420);
        if (!pass) begin
            $display("  got [%0d, %0d, %0d, %0d]", readback_buf[0], readback_buf[1], readback_buf[2], readback_buf[3]);
            $display("FAIL test %0d", test_num); $finish;
        end
        $display("PASS test %0d", test_num);
        $display("");
        tick(); tick();

        // =======================================================
        // Test 5: Different kernel — vector_scale: out[tid] = a[tid] * 3
        //   a = {7, 11, 13, 17}
        //   expect out = {21, 33, 39, 51}
        // =======================================================
        test_num = 5;
        $display("=== Test %0d: vector_scale (different kernel) ===", test_num);

        load_vector_scale_kernel();
        upload_array(256, 4,  7, 11, 13, 17,  0, 0, 0, 0);
        clear_output(4);

        batched_launch(192, 4);  // kernel at addr 192

        gpu_copy_from_device(384, 4);
        $display("  out = [%0d, %0d, %0d, %0d]", readback_buf[0], readback_buf[1], readback_buf[2], readback_buf[3]);
        $display("  exp = [21, 33, 39, 51]");

        pass = (readback_buf[0]==21 && readback_buf[1]==33 && readback_buf[2]==39 && readback_buf[3]==51);
        if (!pass) begin $display("FAIL test %0d", test_num); $finish; end
        $display("PASS test %0d", test_num);
        $display("");
        tick(); tick();

        // =======================================================
        // Test 6: vector_scale, 8 threads (2 batches)
        //   a = {2, 4, 6, 8, 10, 12, 14, 16}
        //   expect out = {6, 12, 18, 24, 30, 36, 42, 48}
        // =======================================================
        test_num = 6;
        $display("=== Test %0d: vector_scale, 8 threads (2 batches) ===", test_num);

        load_vector_scale_kernel();
        upload_array(256, 8,  2, 4, 6, 8, 10, 12, 14, 16);
        clear_output(8);

        batched_launch(192, 8);

        gpu_copy_from_device(384, 8);
        $display("  out = [%0d, %0d, %0d, %0d, %0d, %0d, %0d, %0d]",
            readback_buf[0], readback_buf[1], readback_buf[2], readback_buf[3],
            readback_buf[4], readback_buf[5], readback_buf[6], readback_buf[7]);

        pass = 1;
        for (j = 0; j < 8; j = j + 1) begin
            if (readback_buf[j] !== (j + 1) * 2 * 3) begin
                $display("  MISMATCH at [%0d]: got %0d, expected %0d", j, readback_buf[j], (j+1)*2*3);
                pass = 0;
            end
        end
        if (!pass) begin $display("FAIL test %0d", test_num); $finish; end
        $display("PASS test %0d", test_num);
        $display("");

        // =======================================================
        $display("============================================");
        $display("  ALL 6 TESTS PASSED");
        $display("  Multi-core vector operations verified:");
        $display("    - vector_add: 4 threads, 8 threads");
        $display("    - vector_scale: 4 threads, 8 threads");
        $display("    - Batching across multiple launches");
        $display("    - Idempotent re-launch");
        $display("    - Multiple kernel support");
        $display("============================================");
        $finish;
    end

    initial begin
        #3000000;
        $display("FAIL: global timeout");
        $finish;
    end
endmodule
