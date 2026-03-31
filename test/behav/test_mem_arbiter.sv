/*
test_mem_arbiter.sv

Behavioral test for CP-2: verifies the round-robin memory arbiter.

Setup:
  - mem_arbiter with NUM_CORES=2
  - global_mem_controller (real memory with simulated delay)
  - Arbiter downstream port wired to global_mem_controller core1_* port
  - Controller port (contr_*) used to pre-load and read-back memory

Tests:
  1. Single read from core 0
  2. Single read from core 1
  3. Single write from core 0, then readback to verify
  4. Single write from core 1, then readback to verify
  5. Simultaneous reads from both cores
  6. Simultaneous writes from both cores, then readback both
  7. Cross-read: core 0 reads what core 1 wrote, and vice versa
*/

module test_mem_arbiter();

    // -------------------------------------------------------
    // Signals
    // -------------------------------------------------------
    reg clk;
    reg rst;

    // Arbiter upstream: driven by testbench (simulating 2 cores)
    reg  [1:0]  core_rd_req;
    reg  [1:0]  core_wr_req;
    reg  [63:0] core_addr;      // 2 cores * 32 bits
    reg  [63:0] core_wr_data;
    wire [63:0] core_rd_data;
    wire [1:0]  core_ack;
    wire [1:0]  core_busy;

    // Arbiter downstream: wired to global_mem_controller
    wire        mem_rd_req;
    wire        mem_wr_req;
    wire [31:0] mem_addr;
    wire [31:0] mem_wr_data_w;
    wire [31:0] mem_rd_data;
    wire        mem_ack;
    wire        mem_busy;

    // Controller port: for pre-loading and reading back memory
    reg         contr_wr_en;
    reg         contr_rd_en;
    reg  [31:0] contr_wr_addr;
    reg  [31:0] contr_wr_data;
    reg  [31:0] contr_rd_addr;
    wire [31:0] contr_rd_data;
    wire        contr_rd_ack;

    // -------------------------------------------------------
    // Instantiate arbiter (NUM_CORES=2)
    // -------------------------------------------------------
    mem_arbiter #(.NUM_CORES(2)) arb(
        .clk(clk),
        .rst(rst),

        .core_rd_req(core_rd_req),
        .core_wr_req(core_wr_req),
        .core_addr(core_addr),
        .core_wr_data(core_wr_data),
        .core_rd_data(core_rd_data),
        .core_ack(core_ack),
        .core_busy(core_busy),

        .mem_rd_req(mem_rd_req),
        .mem_wr_req(mem_wr_req),
        .mem_addr(mem_addr),
        .mem_wr_data(mem_wr_data_w),
        .mem_rd_data(mem_rd_data),
        .mem_ack(mem_ack),
        .mem_busy(mem_busy)
    );

    // -------------------------------------------------------
    // Instantiate global_mem_controller
    // -------------------------------------------------------
    global_mem_controller gmem(
        .clk(clk),
        .rst(rst),

        .core1_rd_req(mem_rd_req),
        .core1_wr_req(mem_wr_req),
        .core1_addr(mem_addr),
        .core1_wr_data(mem_wr_data_w),
        .core1_rd_data(mem_rd_data),
        .core1_busy(mem_busy),
        .core1_ack(mem_ack),

        .contr_wr_en(contr_wr_en),
        .contr_rd_en(contr_rd_en),
        .contr_wr_addr(contr_wr_addr),
        .contr_wr_data(contr_wr_data),
        .contr_rd_addr(contr_rd_addr),
        .contr_rd_data(contr_rd_data),
        .contr_rd_ack(contr_rd_ack)
    );

    // -------------------------------------------------------
    // Helpers
    // -------------------------------------------------------
    task tick();
        #5 clk = 0;
        #5 clk = 1;
    endtask

    // Write a word to memory via controller port (instant, no simulated delay)
    task preload_mem(input [31:0] addr, input [31:0] data);
        contr_wr_en   <= 1;
        contr_wr_addr <= addr;
        contr_wr_data <= data;
        tick();
        contr_wr_en <= 0;
        tick();
    endtask

    // Read a word from memory via controller port.
    // FIX: we need TWO ticks after asserting contr_rd_en.
    //   tick 1: memory controller's posedge block runs, schedules NBA for contr_rd_data
    //   tick 2: guarantees the NBA has propagated, so blocking read sees the new value
    task readback_mem(input [31:0] addr, output [31:0] data);
        contr_rd_en   <= 1;
        contr_rd_addr <= addr;
        tick();                // memory controller processes rd_en, schedules NBA
        contr_rd_en   <= 0;
        tick();                // NBA has propagated by now
        data = contr_rd_data;  // safe to read the updated value
    endtask

    // Simulate core i sending a read request (1-cycle pulse)
    task core_send_rd(input integer ci, input [31:0] addr);
        core_rd_req[ci]              <= 1;
        core_addr[ci * 32 +: 32]     <= addr;
        tick();
        core_rd_req[ci]              <= 0;
    endtask

    // Simulate core i sending a write request (1-cycle pulse)
    task core_send_wr(input integer ci, input [31:0] addr, input [31:0] data);
        core_wr_req[ci]              <= 1;
        core_addr[ci * 32 +: 32]     <= addr;
        core_wr_data[ci * 32 +: 32]  <= data;
        tick();
        core_wr_req[ci]              <= 0;
    endtask

    // Wait for core i to receive ack (with timeout)
    task wait_core_ack(input integer ci, output [31:0] rd_data, output integer ok);
        integer cycle_count;
        ok = 0;
        rd_data = '0;
        for (cycle_count = 0; cycle_count < 200; cycle_count = cycle_count + 1) begin
            if (core_ack[ci] && !ok) begin
                rd_data = core_rd_data[ci * 32 +: 32];
                ok = 1;
            end
            if (!ok) tick();
        end
    endtask

    // -------------------------------------------------------
    // Test variables
    // -------------------------------------------------------
    integer test_num;
    integer ok0, ok1;
    reg [31:0] rd0, rd1;

    // -------------------------------------------------------
    // Main test sequence
    // -------------------------------------------------------
    initial begin
        // Initialize everything
        clk          = 0;
        rst          = 0;
        core_rd_req  = '0;
        core_wr_req  = '0;
        core_addr    = '0;
        core_wr_data = '0;
        contr_wr_en  = 0;
        contr_rd_en  = 0;
        contr_wr_addr = '0;
        contr_wr_data = '0;
        contr_rd_addr = '0;

        // Reset
        tick();
        tick();
        rst = 1;
        tick();
        tick();

        // Pre-load memory with known values via controller port
        preload_mem(0,  100);
        preload_mem(4,  200);
        preload_mem(8,  300);
        preload_mem(12, 400);

        // Quick sanity: readback a preloaded value via controller port
        $display("--- Sanity: readback preloaded addr 0 via controller ---");
        readback_mem(0, rd0);
        if (rd0 !== 100) begin
            $display("FAIL sanity: expected 100, got %0d", rd0);
            $finish;
        end
        $display("  OK: readback_mem works, got %0d", rd0);
        tick(); tick();

        // =======================================================
        // Test 1: Single read from core 0
        // =======================================================
        test_num = 1;
        $display("--- Test %0d: Core 0 reads address 0 ---", test_num);
        core_send_rd(0, 0);
        wait_core_ack(0, rd0, ok0);
        if (!ok0) begin
            $display("FAIL test %0d: core 0 never got ack", test_num);
            $finish;
        end
        if (rd0 !== 100) begin
            $display("FAIL test %0d: expected 100, got %0d", test_num, rd0);
            $finish;
        end
        $display("PASS test %0d: core 0 read %0d from addr 0", test_num, rd0);
        tick(); tick();

        // =======================================================
        // Test 2: Single read from core 1
        // =======================================================
        test_num = 2;
        $display("--- Test %0d: Core 1 reads address 4 ---", test_num);
        core_send_rd(1, 4);
        wait_core_ack(1, rd1, ok1);
        if (!ok1) begin
            $display("FAIL test %0d: core 1 never got ack", test_num);
            $finish;
        end
        if (rd1 !== 200) begin
            $display("FAIL test %0d: expected 200, got %0d", test_num, rd1);
            $finish;
        end
        $display("PASS test %0d: core 1 read %0d from addr 4", test_num, rd1);
        tick(); tick();

        // =======================================================
        // Test 3: Single write from core 0, then readback
        // =======================================================
        test_num = 3;
        $display("--- Test %0d: Core 0 writes 555 to addr 100 ---", test_num);
        core_send_wr(0, 32'd100, 32'd555);
        wait_core_ack(0, rd0, ok0);
        if (!ok0) begin
            $display("FAIL test %0d: core 0 write never got ack", test_num);
            $finish;
        end
        $display("  write ack received, reading back...");
        tick(); tick();
        readback_mem(100, rd0);
        if (rd0 !== 555) begin
            $display("FAIL test %0d: addr 100 expected 555, got %0d", test_num, rd0);
            $finish;
        end
        $display("PASS test %0d: wrote 555, read back %0d", test_num, rd0);
        tick(); tick();

        // =======================================================
        // Test 4: Single write from core 1, then readback
        // =======================================================
        test_num = 4;
        $display("--- Test %0d: Core 1 writes 666 to addr 104 ---", test_num);
        core_send_wr(1, 32'd104, 32'd666);
        wait_core_ack(1, rd1, ok1);
        if (!ok1) begin
            $display("FAIL test %0d: core 1 write never got ack", test_num);
            $finish;
        end
        $display("  write ack received, reading back...");
        tick(); tick();
        readback_mem(104, rd1);
        if (rd1 !== 666) begin
            $display("FAIL test %0d: addr 104 expected 666, got %0d", test_num, rd1);
            $finish;
        end
        $display("PASS test %0d: wrote 666, read back %0d", test_num, rd1);
        tick(); tick();

        // =======================================================
        // Test 5: Simultaneous reads from both cores
        // =======================================================
        test_num = 5;
        $display("--- Test %0d: Simultaneous reads ---", test_num);

        core_rd_req         <= 2'b11;
        core_addr[31:0]     <= 32'd8;
        core_addr[63:32]    <= 32'd12;
        tick();
        core_rd_req         <= 2'b00;

        ok0 = 0;
        ok1 = 0;
        rd0 = 0;
        rd1 = 0;
        begin : wait_both_reads
            integer wc;
            for (wc = 0; wc < 200; wc = wc + 1) begin
                if (core_ack[0] && !ok0) begin
                    rd0 = core_rd_data[31:0];
                    ok0 = 1;
                end
                if (core_ack[1] && !ok1) begin
                    rd1 = core_rd_data[63:32];
                    ok1 = 1;
                end
                if (ok0 && ok1) wc = 200;
                else tick();
            end
        end

        if (!ok0 || !ok1) begin
            $display("FAIL test %0d: not both cores got ack (ok0=%0d ok1=%0d)", test_num, ok0, ok1);
            $finish;
        end
        if (rd0 !== 300 || rd1 !== 400) begin
            $display("FAIL test %0d: expected (300,400) got (%0d,%0d)", test_num, rd0, rd1);
            $finish;
        end
        $display("PASS test %0d: core 0 read %0d, core 1 read %0d", test_num, rd0, rd1);
        tick(); tick();

        // =======================================================
        // Test 6: Simultaneous writes from both cores, then readback
        // =======================================================
        test_num = 6;
        $display("--- Test %0d: Simultaneous writes ---", test_num);

        core_wr_req          <= 2'b11;
        core_addr[31:0]      <= 32'd200;
        core_addr[63:32]     <= 32'd204;
        core_wr_data[31:0]   <= 32'd999;
        core_wr_data[63:32]  <= 32'd888;
        tick();
        core_wr_req          <= 2'b00;

        ok0 = 0;
        ok1 = 0;
        begin : wait_both_writes
            integer wc;
            for (wc = 0; wc < 200; wc = wc + 1) begin
                if (core_ack[0] && !ok0) ok0 = 1;
                if (core_ack[1] && !ok1) ok1 = 1;
                if (ok0 && ok1) wc = 200;
                else tick();
            end
        end

        if (!ok0 || !ok1) begin
            $display("FAIL test %0d: not both write acks (ok0=%0d ok1=%0d)", test_num, ok0, ok1);
            $finish;
        end
        $display("  both write acks received, reading back...");

        tick(); tick();
        readback_mem(200, rd0);
        readback_mem(204, rd1);

        if (rd0 !== 999) begin
            $display("FAIL test %0d: addr 200 expected 999, got %0d", test_num, rd0);
            $finish;
        end
        if (rd1 !== 888) begin
            $display("FAIL test %0d: addr 204 expected 888, got %0d", test_num, rd1);
            $finish;
        end
        $display("PASS test %0d: addr 200=%0d, addr 204=%0d", test_num, rd0, rd1);
        tick(); tick();

        // =======================================================
        // Test 7: Cross-read — core 0 reads what core 1 wrote, and vice versa
        // =======================================================
        test_num = 7;
        $display("--- Test %0d: Cross-read ---", test_num);

        // Core 0 reads addr 104 (written by core 1 in test 4 = 666)
        core_send_rd(0, 32'd104);
        wait_core_ack(0, rd0, ok0);
        if (!ok0 || rd0 !== 666) begin
            $display("FAIL test %0d: core 0 read addr 104 expected 666, got %0d (ok=%0d)", test_num, rd0, ok0);
            $finish;
        end

        // Core 1 reads addr 100 (written by core 0 in test 3 = 555)
        core_send_rd(1, 32'd100);
        wait_core_ack(1, rd1, ok1);
        if (!ok1 || rd1 !== 555) begin
            $display("FAIL test %0d: core 1 read addr 100 expected 555, got %0d (ok=%0d)", test_num, rd1, ok1);
            $finish;
        end
        $display("PASS test %0d: core 0 read %0d from core1's addr, core 1 read %0d from core0's addr", test_num, rd0, rd1);

        // =======================================================
        // Done
        // =======================================================
        $display("");
        $display("ALL 7 TESTS PASSED");
        $finish;
    end

    // Timeout watchdog
    initial begin
        #100000;
        $display("FAIL: global timeout");
        $finish;
    end
endmodule