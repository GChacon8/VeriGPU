/*
mem_arbiter.sv — Round-robin memory arbiter for multi-core

Multiplexes N core memory ports onto 1 downstream memory port
(matching the core1_* interface of global_mem_controller).

Protocol summary:
  - Cores send rd_req or wr_req as 1-cycle combinational pulses
  - Arbiter latches each request on the clock edge (1 cycle capture delay)
  - Round-robin scan selects the next pending request to forward downstream
  - When downstream acks, the ack and rd_data are forwarded combinationally
    to the requesting core (zero extra ack latency)

Total added latency vs direct connection: 1 cycle (for the latch).

Constraint: NUM_CORES must be a power of 2 (1, 2, 4, 8).

The controller port (contr_*) on global_mem_controller is NOT routed
through this arbiter — it stays separate, as the controller only
accesses memory when cores are disabled.

CP-3 FIX: Removed !pending_rd[i] && !pending_wr[i] guards from latch
conditions. These caused a deadlock when a core received ack and
immediately sent a new request on the same cycle (standard core behavior
for single-cycle instructions). The guard evaluated the pre-NBA pending
value (still 1), rejecting the new request even though the clear NBA
would set it to 0. Without the guard, the latch NBA (later in code)
correctly wins over the clear NBA for the same variable.
*/

`default_nettype none
module mem_arbiter #(
    parameter NUM_CORES = 4
)(
    input clk,
    input rst,

    // -------------------------------------------------------
    // Upstream: from cores
    // Bit vectors for single-bit signals.
    // Packed buses for addr/data: core i uses [i*32 +: 32].
    // -------------------------------------------------------
    input  [NUM_CORES-1:0]                core_rd_req,
    input  [NUM_CORES-1:0]                core_wr_req,
    input  [NUM_CORES*addr_width-1:0]     core_addr,
    input  [NUM_CORES*data_width-1:0]     core_wr_data,
    output reg [NUM_CORES*data_width-1:0] core_rd_data,
    output reg [NUM_CORES-1:0]            core_ack,
    output reg [NUM_CORES-1:0]            core_busy,

    // -------------------------------------------------------
    // Downstream: to global_mem_controller (core1_* port)
    // -------------------------------------------------------
    output reg                        mem_rd_req,
    output reg                        mem_wr_req,
    output reg [addr_width-1:0]       mem_addr,
    output reg [data_width-1:0]       mem_wr_data,
    input      [data_width-1:0]       mem_rd_data,
    input                             mem_ack,
    input                             mem_busy
);
    // -------------------------------------------------------
    // Internal parameters
    // -------------------------------------------------------
    localparam GRANT_W = (NUM_CORES > 1) ? $clog2(NUM_CORES) : 1;

    // -------------------------------------------------------
    // State machine
    // -------------------------------------------------------
    typedef enum bit[0:0] {
        ARB_IDLE,
        ARB_ACTIVE
    } e_arb_state;

    reg [0:0]          state;
    reg [0:0]          n_state;
    reg [GRANT_W-1:0]  grant;
    reg [GRANT_W-1:0]  n_grant;
    reg [GRANT_W-1:0]  active_core;
    reg [GRANT_W-1:0]  n_active_core;

    // -------------------------------------------------------
    // Pending request storage (latched from 1-cycle core pulses)
    // -------------------------------------------------------
    reg [NUM_CORES-1:0]     pending_rd;
    reg [NUM_CORES-1:0]     pending_wr;
    reg [addr_width-1:0]    pending_addr  [NUM_CORES];
    reg [data_width-1:0]    pending_wdata [NUM_CORES];

    // -------------------------------------------------------
    // Combinational helpers
    // -------------------------------------------------------
    reg                do_clear_pending;
    reg                found;
    reg [GRANT_W-1:0]  found_idx;
    integer            scan_i;
    reg [GRANT_W-1:0]  scan_idx;

    // -------------------------------------------------------
    // Combinational block
    // -------------------------------------------------------
    always @(*) begin
        n_state          = state;
        n_grant          = grant;
        n_active_core    = active_core;
        do_clear_pending = 0;

        core_ack     = '0;
        core_busy    = '0;
        core_rd_data = '0;

        mem_rd_req   = 0;
        mem_wr_req   = 0;
        mem_addr     = '0;
        mem_wr_data  = '0;

        `assert_known(state);
        case (state)
            ARB_IDLE: begin
                found     = 0;
                found_idx = '0;
                for (scan_i = 0; scan_i < NUM_CORES; scan_i = scan_i + 1) begin
                    scan_idx = grant + scan_i;
                    if (!found && (pending_rd[scan_idx] || pending_wr[scan_idx])) begin
                        found     = 1;
                        found_idx = scan_idx;
                    end
                end

                if (found) begin
                    n_active_core = found_idx;
                    n_state       = ARB_ACTIVE;
                    mem_addr      = pending_addr[found_idx];
                    if (pending_rd[found_idx]) begin
                        mem_rd_req = 1;
                    end else begin
                        mem_wr_req  = 1;
                        mem_wr_data = pending_wdata[found_idx];
                    end
                end
            end

            ARB_ACTIVE: begin
                `assert_known(mem_ack);
                if (mem_ack) begin
                    core_ack[active_core] = 1;
                    core_rd_data[active_core * data_width +: data_width] = mem_rd_data;
                    do_clear_pending = 1;
                    n_grant = active_core + 1;
                    n_state = ARB_IDLE;
                end
            end
        endcase
    end

    // -------------------------------------------------------
    // Sequential block
    // -------------------------------------------------------
    integer i;
    always @(posedge clk, negedge rst) begin
        `assert_known(rst);
        if (~rst) begin
            state       <= ARB_IDLE;
            grant       <= '0;
            active_core <= '0;
            pending_rd  <= '0;
            pending_wr  <= '0;
            for (i = 0; i < NUM_CORES; i = i + 1) begin
                pending_addr[i]  <= '0;
                pending_wdata[i] <= '0;
            end
        end else begin
            state       <= n_state;
            grant       <= n_grant;
            active_core <= n_active_core;

            // Clear FIRST so that a same-cycle new request from the
            // same core wins (last non-blocking assignment wins).
            if (do_clear_pending) begin
                pending_rd[active_core] <= 0;
                pending_wr[active_core] <= 0;
            end

            // Latch new requests SECOND.
            // FIX: No guard on !pending_rd/wr. A core only sends a new
            // request after receiving ack (it's blocked in C1/C2 until then).
            // The only overlap is: clear + new request on the same cycle.
            // Since this NBA comes after the clear NBA, it wins correctly.
            for (i = 0; i < NUM_CORES; i = i + 1) begin
                if (core_rd_req[i]) begin
                    pending_rd[i]    <= 1;
                    pending_addr[i]  <= core_addr[i * addr_width +: addr_width];
                end
                if (core_wr_req[i]) begin
                    pending_wr[i]    <= 1;
                    pending_addr[i]  <= core_addr[i * addr_width +: addr_width];
                    pending_wdata[i] <= core_wr_data[i * data_width +: data_width];
                end
            end
        end
    end
endmodule
