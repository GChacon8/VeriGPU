/*
A compute unit containing NUM_CORES cores sharing a single memory port
via a round-robin arbiter.

All cores receive the SAME control signals (clr, ena, set_pc_req,
set_pc_addr), this is SIMT: same program, different thread IDs.

The external memory port has the same protocol as a single core's
port, so gpu_die.sv can swap a single core for a compute_unit
with zero wiring changes on the memory controller side.

NUM_CORES must be a power of 2 (constraint of the mem_arbiter).
*/

`default_nettype none
module compute_unit #(
    parameter NUM_CORES = 4
)(
    input rst,
    input clk,
    input clr,
    input ena,
    input set_pc_req,
    input [addr_width - 1:0] set_pc_addr,
    input [data_width - 1:0] base_thread_id,

    // Debug output (directly from core 0)
    output [data_width - 1:0] out,
    output outen,
    output outflen,

    // Collective halt: high when ALL cores have halted (latched)
    output halt,

    // Single memory port (arbiter downstream — same protocol as core)
    output                    mem_rd_req,
    output                    mem_wr_req,
    output [addr_width - 1:0] mem_addr,
    output [data_width - 1:0] mem_wr_data,
    input  [data_width - 1:0] mem_rd_data,
    input                     mem_ack,
    input                     mem_busy
);
    // -------------------------------------------------------
    // Internal wiring: cores <-> arbiter (packed buses)
    // -------------------------------------------------------
    wire [NUM_CORES - 1:0]                  arb_core_rd_req;
    wire [NUM_CORES - 1:0]                  arb_core_wr_req;
    wire [NUM_CORES * addr_width - 1:0]     arb_core_addr;
    wire [NUM_CORES * data_width - 1:0]     arb_core_wr_data;
    wire [NUM_CORES * data_width - 1:0]     arb_core_rd_data;
    wire [NUM_CORES - 1:0]                  arb_core_ack;
    wire [NUM_CORES - 1:0]                  arb_core_busy;

    // Per-core raw halt signals (momentary, 1-cycle pulses)
    wire [NUM_CORES - 1:0] core_halt_raw;

    // -------------------------------------------------------
    // Halt latching
    //
    // Each core's halt output is a momentary combinational pulse.
    // With round-robin memory access, cores reach halt at different
    // times. We latch each pulse so that once a core has halted,
    // its flag stays high until the next clr.
    // -------------------------------------------------------
    reg [NUM_CORES - 1:0] halted;

    always @(posedge clk, negedge rst) begin
        if (~rst) begin
            halted <= '0;
        end else if (clr) begin
            halted <= '0;
        end else begin
            halted <= halted | core_halt_raw;
        end
    end

    assign halt = &halted;

    // -------------------------------------------------------
    // Generate cores
    // -------------------------------------------------------
    genvar gi;
    generate
        for (gi = 0; gi < NUM_CORES; gi = gi + 1) begin : gen_core

            // Thread ID with explicit width to avoid iverilog warning
            wire [data_width - 1:0] this_thread_id;
            assign this_thread_id = base_thread_id + gi[data_width - 1:0];

            // Per-core output wires (memory side)
            wire [addr_width - 1:0] c_mem_addr;
            wire [data_width - 1:0] c_mem_wr_data;
            wire                    c_mem_rd_req;
            wire                    c_mem_wr_req;

            // Per-core debug output wires
            wire [data_width - 1:0] c_out;
            wire                    c_outen;
            wire                    c_outflen;

            core core_i(
                .rst    (rst),
                .clk    (clk),
                .clr    (clr),
                .ena    (ena),
                .set_pc_req  (set_pc_req),
                .set_pc_addr (set_pc_addr),

                .thread_id   (this_thread_id),

                .out     (c_out),
                .outen   (c_outen),
                .outflen (c_outflen),

                .halt    (core_halt_raw[gi]),

                .mem_addr    (c_mem_addr),
                .mem_rd_data (arb_core_rd_data[gi * data_width +: data_width]),
                .mem_wr_data (c_mem_wr_data),
                .mem_rd_req  (c_mem_rd_req),
                .mem_wr_req  (c_mem_wr_req),
                .mem_ack     (arb_core_ack[gi]),
                .mem_busy    (arb_core_busy[gi])
            );

            assign arb_core_addr   [gi * addr_width +: addr_width] = c_mem_addr;
            assign arb_core_wr_data[gi * data_width +: data_width] = c_mem_wr_data;
            assign arb_core_rd_req[gi] = c_mem_rd_req;
            assign arb_core_wr_req[gi] = c_mem_wr_req;
        end
    endgenerate

    // -------------------------------------------------------
    // Debug output from core 0 only
    // -------------------------------------------------------
    assign out     = gen_core[0].c_out;
    assign outen   = gen_core[0].c_outen;
    assign outflen = gen_core[0].c_outflen;

    // -------------------------------------------------------
    // Memory arbiter
    // -------------------------------------------------------
    mem_arbiter #(
        .NUM_CORES(NUM_CORES)
    ) arbiter (
        .clk (clk),
        .rst (rst),

        .core_rd_req  (arb_core_rd_req),
        .core_wr_req  (arb_core_wr_req),
        .core_addr    (arb_core_addr),
        .core_wr_data (arb_core_wr_data),
        .core_rd_data (arb_core_rd_data),
        .core_ack     (arb_core_ack),
        .core_busy    (arb_core_busy),

        .mem_rd_req  (mem_rd_req),
        .mem_wr_req  (mem_wr_req),
        .mem_addr    (mem_addr),
        .mem_wr_data (mem_wr_data),
        .mem_rd_data (mem_rd_data),
        .mem_ack     (mem_ack),
        .mem_busy    (mem_busy)
    );
endmodule
