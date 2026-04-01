/*
test_compute_unit.sv  — DIAGNOSTIC VERSION

Heavy instrumentation to find where instruction data is lost.
Verifies program in memory before launching.
Monitors all downstream memory bus events.
*/

module test_compute_unit();

    reg         clk, rst;
    reg         cu_clr, cu_ena;
    reg         cu_set_pc_req;
    reg  [31:0] cu_set_pc_addr;
    reg  [31:0] cu_base_thread_id;

    wire [31:0] cu_out;
    wire        cu_outen, cu_outflen;
    wire        cu_halt;

    wire        cu_mem_rd_req, cu_mem_wr_req;
    wire [31:0] cu_mem_addr, cu_mem_wr_data;
    wire [31:0] cu_mem_rd_data;
    wire        cu_mem_ack, cu_mem_busy;

    reg         contr_wr_en, contr_rd_en;
    reg  [31:0] contr_wr_addr, contr_wr_data;
    reg  [31:0] contr_rd_addr;
    wire [31:0] contr_rd_data;
    wire        contr_rd_ack;

    compute_unit #(.NUM_CORES(4)) cu(
        .rst(rst), .clk(clk), .clr(cu_clr), .ena(cu_ena),
        .set_pc_req(cu_set_pc_req), .set_pc_addr(cu_set_pc_addr),
        .base_thread_id(cu_base_thread_id),
        .out(cu_out), .outen(cu_outen), .outflen(cu_outflen),
        .halt(cu_halt),
        .mem_rd_req(cu_mem_rd_req), .mem_wr_req(cu_mem_wr_req),
        .mem_addr(cu_mem_addr), .mem_wr_data(cu_mem_wr_data),
        .mem_rd_data(cu_mem_rd_data),
        .mem_ack(cu_mem_ack), .mem_busy(cu_mem_busy)
    );

    global_mem_controller gmem(
        .clk(clk), .rst(rst),
        .core1_rd_req(cu_mem_rd_req), .core1_wr_req(cu_mem_wr_req),
        .core1_addr(cu_mem_addr), .core1_wr_data(cu_mem_wr_data),
        .core1_rd_data(cu_mem_rd_data),
        .core1_busy(cu_mem_busy), .core1_ack(cu_mem_ack),
        .contr_wr_en(contr_wr_en), .contr_rd_en(contr_rd_en),
        .contr_wr_addr(contr_wr_addr), .contr_wr_data(contr_wr_data),
        .contr_rd_addr(contr_rd_addr), .contr_rd_data(contr_rd_data),
        .contr_rd_ack(contr_rd_ack)
    );

    // -------------------------------------------------------
    // Helpers
    // -------------------------------------------------------
    task tick();
        #5 clk = 0;
        #5 clk = 1;
    endtask

    task write_mem(input [31:0] addr, input [31:0] data);
        contr_wr_en   <= 1;
        contr_wr_addr <= addr;
        contr_wr_data <= data;
        tick();
        contr_wr_en <= 0;
        tick();
    endtask

    task readback_mem(input [31:0] addr, output [31:0] data);
        contr_rd_en   <= 1;
        contr_rd_addr <= addr;
        tick();
        contr_rd_en   <= 0;
        tick();
        data = contr_rd_data;
    endtask

    task load_program();
        write_mem(128, 32'h00229313);  // slli x6, x5, 2
        write_mem(132, 32'h00532023);  // sw   x5, 0(x6)
        write_mem(136, 32'h000F43B7);  // lui  x7, 0xF4
        write_mem(140, 32'h24438393);  // addi x7, x7, 0x244
        write_mem(144, 32'h0003A023);  // sw   x0, 0(x7)
    endtask

    task clear_data_region(input [31:0] start_addr, input integer num_words);
        integer ci;
        for (ci = 0; ci < num_words; ci = ci + 1)
            write_mem(start_addr + ci * 4, 32'd0);
    endtask

    // -------------------------------------------------------
    // DIAGNOSTIC: Monitor downstream memory bus (first 50 events)
    // -------------------------------------------------------
    integer diag_count;
    initial diag_count = 0;

    always @(posedge clk) begin
        if (cu_ena && diag_count < 50) begin
            if (cu_mem_rd_req) begin
                $display("  [BUS] t=%0t rd_req addr=%0d (word=%0d)", $time, cu_mem_addr, cu_mem_addr >> 2);
                diag_count = diag_count + 1;
            end
            if (cu_mem_wr_req) begin
                $display("  [BUS] t=%0t wr_req addr=%0d data=%h", $time, cu_mem_addr, cu_mem_wr_data);
                diag_count = diag_count + 1;
            end
            if (cu_mem_ack) begin
                $display("  [BUS] t=%0t ack    rd_data=%h", $time, cu_mem_rd_data);
                diag_count = diag_count + 1;
            end
        end
    end

    // -------------------------------------------------------
    // Main test
    // -------------------------------------------------------
    integer test_num;
    reg [31:0] rd;

    initial begin
        clk = 0; rst = 0;
        cu_clr = 0; cu_ena = 0;
        cu_set_pc_req = 0; cu_set_pc_addr = 0;
        cu_base_thread_id = 0;
        contr_wr_en = 0; contr_rd_en = 0;
        contr_wr_addr = 0; contr_wr_data = 0; contr_rd_addr = 0;

        tick(); tick();
        rst = 1;
        tick(); tick();

        // ===== STEP A: Load and verify program =====
        $display("=== Step A: Load program and verify ===");
        load_program();

        readback_mem(128, rd); $display("  mem[128]=%h (expect 00229313)", rd);
        readback_mem(132, rd); $display("  mem[132]=%h (expect 00532023)", rd);
        readback_mem(136, rd); $display("  mem[136]=%h (expect 000f43b7)", rd);
        readback_mem(140, rd); $display("  mem[140]=%h (expect 24438393)", rd);
        readback_mem(144, rd); $display("  mem[144]=%h (expect 0003a023)", rd);
        $display("");

        // ===== STEP B: Clear data region and re-verify program =====
        $display("=== Step B: Clear data region, re-verify program ===");
        clear_data_region(0, 4);

        readback_mem(128, rd); $display("  mem[128]=%h (expect 00229313)", rd);
        if (rd !== 32'h00229313) begin
            $display("FAIL: clear_data_region corrupted program at addr 128!");
            $finish;
        end
        $display("  Program still intact after clear");
        $display("");

        // ===== STEP C: Launch 4 cores =====
        test_num = 1;
        $display("--- Test %0d: 4 cores, base_thread_id=0 ---", test_num);
        $display("  (First ~50 memory bus events logged below)");

        cu_base_thread_id <= 0;
        cu_ena <= 0;
        cu_clr <= 1;
        tick();
        cu_clr <= 0;
        tick();

        // Re-verify program is STILL in memory right before ena
        readback_mem(128, rd);
        $display("  PRE-ENA verify: mem[128]=%h (expect 00229313)", rd);
        if (rd !== 32'h00229313) begin
            $display("FAIL: program vanished before ena!");
            $finish;
        end

        cu_ena <= 1;
        tick();

        // Wait for halt
        begin
            integer cycle_count;
            integer done;
            done = 0;
            for (cycle_count = 0; cycle_count < 5000 && !done; cycle_count = cycle_count + 1) begin
                if (cu_halt) begin
                    done = 1;
                    $display("  all cores halted after %0d cycles", cycle_count);
                end
                if (!done) tick();
            end
            cu_ena <= 0;
            tick();
            if (!done) begin
                $display("FAIL: timeout — not all cores halted within 5000 cycles");
                $finish;
            end
        end

        // Read back results
        tick(); tick();
        readback_mem(0,  rd); $display("  mem[0]  = %0d (expect 0)", rd);
        if (rd !== 0) begin $display("FAIL"); $finish; end
        readback_mem(4,  rd); $display("  mem[4]  = %0d (expect 1)", rd);
        if (rd !== 1) begin $display("FAIL"); $finish; end
        readback_mem(8,  rd); $display("  mem[8]  = %0d (expect 2)", rd);
        if (rd !== 2) begin $display("FAIL"); $finish; end
        readback_mem(12, rd); $display("  mem[12] = %0d (expect 3)", rd);
        if (rd !== 3) begin $display("FAIL"); $finish; end

        $display("PASS test %0d", test_num);
        $display("");
        $display("ALL TESTS PASSED");
        $finish;
    end

    initial begin
        #500000;
        $display("FAIL: global timeout");
        $finish;
    end
endmodule
