// Linear projection unit: y = x * W^T + b (torch.nn.Linear).
//
// Used by every Linear layer in Granite 4.0-H-350M (in_proj/out_proj, attention
// Q/K/V/O projections, MLP gate+up/down). bfloat16 data, parameterized
// (IN_FEATURES x OUT_FEATURES) weight matrix.
//
// Numerics (matching torch bf16 CPU semantics, verified empirically):
//   - bf16 inputs and weights are widened to binary32 exactly (zero-padded)
//   - the dot product is accumulated sequentially in binary32
//     (bf16 x bf16 products are exact in binary32, so each step is one add)
//   - the bias is added last in binary32
//   - the final sum is rounded once to bfloat16
// This matches torch F.linear bit-exactly on the vast majority of elements;
// rare (<<0.1%) 1-ULP differences can remain because ATen's blocked GEMM
// accumulation order is not replicated (documented in the in-loop test).
//
// Architecture:
//   - every pass computes ROWSPW = 2*LANES output rows: LANES complete binary32
//     MAC datapaths (own pipelined MUL and accumulator) each carry TWO
//     interleaved accumulator contexts (rows word*ROWSPW + lane and
//     word*ROWSPW + LANES + lane). Both arithmetic units are 2-stage
//     registered: the adder (fp_add_pipe2, 2-cycle latency) and the
//     multiplier (fp_mul_pipe2, 2-cycle latency). Context A submits on even
//     MAC slots, context B on odd slots, so the MAC issue rate stays 1/cycle
//     per lane while each row keeps the exact sequential fp32 add order of the
//     original single-cycle implementation (seed 0+0, then products in
//     declaration order, then the bias). The accumulator is the adder's own
//     output register: during a context's slot it always holds that context's
//     previous sum (updated every other cycle), so no extra accumulator
//     registers are needed.
//   - the multiplier operands are registered (MAC_FILL loads the first pair,
//     MAC prefetches two slots ahead) so the weight/input memory read is off
//     the multiplier input path; with the pipelined multiplier the product of
//     operands loaded at slot d appears at mul_y in slot d+3, so the prefetch
//     supplies A_k one slot earlier (odd slots) and B_k on even slots
//   - per pass: MAC_FILL + (2*IN_FEATURES + 6) MAC slots (2 seeds + 2*IN
//     accumulate + 2 bias + 2 result-latch cycles) + STORE + 2*LANES output
//     beats, i.e. the same cycles/output-row as the single-context version
//   - weight rows are packed ROWSPW per memory word (row = word*ROWSPW+slice),
//     so one read serves both contexts; bias_mem stays per row
//
// AXI-Stream:
//   - input : IN_FEATURES beats (tlast expected on the final beat), accepted
//             while idle (s_axis_tready = state == IDLE)
//   - output: OUT_FEATURES beats, tlast on the final beat, held under
//             downstream backpressure (m_axis_tready); blocks stream rows
//             0..ROWSPW-1 in row order and the final block masks the tail
//             (supports OUT_FEATURES % ROWSPW != 0)

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */

module matrix_unit #(
  parameter int IN_FEATURES  = 768,
  parameter int OUT_FEATURES = 768,
  parameter int W_DATA       = 16,
  parameter int LANES        = 1,
  parameter int IN_W         = $clog2(IN_FEATURES),
  parameter int OUT_W        = $clog2(OUT_FEATURES)
) (
  input  logic                  clk,
  input  logic                  rst_n,

  // Weight/bias load interface (one element per cycle, before streaming)
  input  logic                  load_en,
  input  logic [OUT_W-1:0]      load_out_idx,
  input  logic [IN_W-1:0]       load_in_idx,
  input  logic [W_DATA-1:0]     load_wdata,
  input  logic                  load_is_bias,   // 0 = weight, 1 = bias

  // AXI-Stream input (x vector: IN_FEATURES bf16 beats)
  input  logic                  s_axis_tvalid,
  output logic                  s_axis_tready,
  input  logic [W_DATA-1:0]     s_axis_tdata,
  input  logic                  s_axis_tlast,

  // AXI-Stream output (y vector: OUT_FEATURES bf16 beats)
  output logic                  m_axis_tvalid,
  input  logic                  m_axis_tready,
  output logic [W_DATA-1:0]     m_axis_tdata,
  output logic                  m_axis_tlast,

  output logic                  busy
);

  // ------------------------------------------------------------ geometry
  localparam int ROWSPW  = 2 * LANES;                     // rows per pass
  localparam int W_WORDS = (OUT_FEATURES + ROWSPW - 1) / ROWSPW;  // words
  localparam int N_ROWS  = W_WORDS * ROWSPW;              // padded rows
  localparam int W_PACK  = ROWSPW * W_DATA;               // packed word width
  localparam int BLK_W   = (W_WORDS < 2) ? 1 : $clog2(W_WORDS);
  localparam int LCNT_W  = (ROWSPW < 2) ? 1 : $clog2(ROWSPW);
  localparam int ROW_W   = (N_ROWS < 2) ? 1 : $clog2(N_ROWS + 1);
  // Largest slot is 2*IN_FEATURES + 5; this width covers 0..2*IN_FEATURES+5.
  localparam int SLOT_W  = $clog2(2 * IN_FEATURES + 6);
  localparam int CLIMIT  = 2 * IN_FEATURES;

  // ------------------------------------------------------------ storage
  logic [W_DATA-1:0] x_buf    [0:IN_FEATURES-1];
  logic [W_PACK-1:0] w_mem    [0:W_WORDS-1][0:IN_FEATURES-1];
  logic [W_DATA-1:0] bias_mem [0:N_ROWS-1];

  always_ff @(posedge clk) begin
    if (load_en) begin
      if (load_is_bias)
        bias_mem[load_out_idx] <= load_wdata;
      else
        w_mem[load_out_idx / ROWSPW][load_in_idx]
             [(load_out_idx % ROWSPW) * W_DATA +: W_DATA] <= load_wdata;
    end
  end

  // ------------------------------------------------------------ MAC lanes
  // LANES independent MAC datapaths, each with two interleaved accumulator
  // contexts through one 2-stage pipelined fp32 adder.
  logic [31:0]       mul_a;      // shared x operand (combinationally driven)
  logic [31:0]       mul_a_q;    // registered x operand
  logic [31:0]       mul_b     [0:LANES-1];
  logic [31:0]       mul_b_q   [0:LANES-1];
  logic [31:0]       mul_y     [0:LANES-1];
  logic [31:0]       add_a     [0:LANES-1];
  logic [31:0]       add_b     [0:LANES-1];
  logic [31:0]       add_y     [0:LANES-1];
  logic [31:0]       final_a   [0:LANES-1];  // context A biased sum (captured)
  logic [31:0]       final_b   [0:LANES-1];  // context B biased sum (captured)
  logic [W_DATA-1:0] rounded_a [0:LANES-1];
  logic [W_DATA-1:0] rounded_b [0:LANES-1];
  logic [W_DATA-1:0] out_buf   [0:ROWSPW-1];

  genvar l;
  generate
    for (l = 0; l < LANES; l++) begin : g_lane
      // 2-stage registered fp32 multiplier (RM_RNE), 2-cycle latency.
      fp_mul_pipe2 #(.W_EXP(8), .W_MANT(23)) u_fp_mul (
        .clk(clk), .rst_n(rst_n),
        .a(mul_a), .b(mul_b[l]), .y(mul_y[l])
      );

      // 2-stage registered fp32 adder (RM_RNE, OP_ADD), 2-cycle latency.
      fp_add_pipe2 #(.W_EXP(8), .W_MANT(23)) u_fp_add (
        .clk(clk), .rst_n(rst_n),
        .a(add_a[l]), .b(add_b[l]), .y(add_y[l])
      );

      // Final binary32 -> bfloat16 rounding (one rounding after the bias add).
      fp32_to_bf16_round u_round_a (.x(final_a[l]), .y(rounded_a[l]));
      fp32_to_bf16_round u_round_b (.x(final_b[l]), .y(rounded_b[l]));
    end
  endgenerate

  // ------------------------------------------------------------ FSM
  typedef enum logic [2:0] {
    IDLE,      // accept IN_FEATURES input beats
    MAC_FILL,  // register the first multiplier operands from memory
    MAC,       // one MAC issue per cycle per lane via the interleaved contexts
    STORE,     // latch the rounded biased sums (stable under backpressure)
    OUT        // stream the block's output beats (row order, then next block)
  } state_t;

  state_t state;
  logic [IN_W-1:0]   i_cnt;
  logic [BLK_W-1:0]  o_cnt;      // weight/row block index (row_base = o_cnt*ROWSPW)
  logic [LCNT_W-1:0] lane_cnt;   // row currently streaming in OUT
  logic [SLOT_W-1:0] slot_cnt;   // MAC slot: 0/1 seeds, then interleaved MACs

  // Absolute first row of the current block and the number of valid rows in
  // it (masked tail pass when OUT_FEATURES is not a multiple of ROWSPW).
  logic [ROW_W-1:0] row_base;
  logic [ROW_W-1:0] rows_left;
  logic [ROW_W-1:0] lanes_this;
  assign row_base   = ROW_W'(o_cnt) * ROW_W'(ROWSPW);
  assign rows_left  = ROW_W'(OUT_FEATURES) - row_base;
  assign lanes_this = (rows_left < ROW_W'(ROWSPW)) ? rows_left : ROW_W'(ROWSPW);

  // Prefetch element index: with the pipelined multiplier, operands loaded at
  // slot d appear at mul_y in slot d+3, so the product A_k consumed at slot
  // 2k+2 is loaded at the odd slot 2k-1 (element k, context A row) and B_k
  // consumed at slot 2k+3 is loaded at the even slot 2k (element k, context B
  // row): element index ceil(slot/2). See the MAC prefetch comment.
  logic [IN_W-1:0] pf_elem;
  assign pf_elem = IN_W'((slot_cnt + SLOT_W'(slot_cnt[0])) >> 1);

  assign s_axis_tready = (state == IDLE);
  assign busy          = (state != IDLE);

  assign m_axis_tvalid = (state == OUT);
  assign m_axis_tlast  = (state == OUT)
                      && (o_cnt == BLK_W'(W_WORDS - 1))
                      && (lane_cnt == LCNT_W'(lanes_this - 1'b1));
  assign m_axis_tdata  = out_buf[lane_cnt];

  always_comb begin
    mul_a = '0;

    for (int li = 0; li < LANES; li++) begin
      mul_b[li] = '0;
      add_a[li] = add_y[li];   // hold the pipeline output by default
      add_b[li] = '0;
    end

    case (state)
      MAC: begin
        mul_a = mul_a_q;
        for (int li = 0; li < LANES; li++) begin
          mul_b[li] = mul_b_q[li];
          if (slot_cnt <= SLOT_W'(1)) begin
            // Context seeds: 0 + 0 starts each accumulator exactly like the
            // original single-cycle schedule (RNE gives +0).
            add_a[li] = '0;
            add_b[li] = '0;
          end else if (slot_cnt <= SLOT_W'(CLIMIT + 1)) begin
            // Interleaved accumulation: even slots feed context A's row, odd
            // slots context B's row. add_y is that context's accumulator
            // (updated every other cycle), mul_y its next product.
            add_a[li] = add_y[li];
            add_b[li] = mul_y[li];
          end else if (slot_cnt == SLOT_W'(CLIMIT + 2)) begin
            add_a[li] = add_y[li];   // context A: bias added last in fp32
            add_b[li] = {bias_mem[row_base + li], {(32 - W_DATA) {1'b0}}};
          end else if (slot_cnt == SLOT_W'(CLIMIT + 3)) begin
            add_a[li] = add_y[li];   // context B: bias added last in fp32
            add_b[li] = {bias_mem[row_base + LANES + li],
                         {(32 - W_DATA) {1'b0}}};
          end
          // slots CLIMIT+4/CLIMIT+5: no submission (results are latched)
        end
      end

      default: ;
    endcase
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state <= IDLE;
      i_cnt <= '0;
      o_cnt <= '0;
      lane_cnt <= '0;
      slot_cnt <= '0;
    end else begin
      case (state)
        IDLE: begin
          if (s_axis_tvalid) begin
            x_buf[i_cnt] <= s_axis_tdata;
            // A well-formed vector is IN_FEATURES beats with tlast on the
            // final beat; the counter bound also ends a stream that omits it.
            if (s_axis_tlast || (i_cnt == IN_W'(IN_FEATURES - 1))) begin
              i_cnt <= '0;
              o_cnt <= '0;
              lane_cnt <= '0;
              slot_cnt <= '0;
              state <= MAC_FILL;
            end else begin
              i_cnt <= i_cnt + 1'b1;
            end
          end
        end

        MAC_FILL: begin
          // Product for the first context-A accumulate (slot 2): element 0.
          mul_a_q <= {x_buf[0], {(32 - W_DATA) {1'b0}}};
          for (int li = 0; li < LANES; li++)
            mul_b_q[li] <= {w_mem[o_cnt][0][(li * W_DATA) +: W_DATA],
                            {(32 - W_DATA) {1'b0}}};
          state <= MAC;
        end

        MAC: begin
          slot_cnt <= slot_cnt + 1'b1;

          // Prefetch the operands whose product is consumed at slot+3: the
          // multiplier has a registered operand stage (q) plus the 2-cycle
          // fp_mul_pipe2, so a load at slot c appears at mul_y in slot c+3.
          // Operands alternate contexts: even c loads B's element c/2, odd c
          // loads A's element (c+1)/2. The last needed products (both A and B
          // element IN_FEATURES-1) are loaded by slot 2*IN_FEATURES-2.
          if (slot_cnt < SLOT_W'(CLIMIT - 1)) begin
            mul_a_q <= {x_buf[pf_elem], {(32 - W_DATA) {1'b0}}};
            for (int li = 0; li < LANES; li++)
              mul_b_q[li] <= {w_mem[o_cnt][pf_elem]
                                   [(li + (slot_cnt[0] ? 0 : LANES)) * W_DATA +: W_DATA],
                              {(32 - W_DATA) {1'b0}}};
          end

          // Context A's biased sum is valid at slot CLIMIT+4 and context B's
          // at CLIMIT+5 (the adder's 2-cycle latency); latch them before the
          // shared adder output register moves on.
          if (slot_cnt == SLOT_W'(CLIMIT + 4))
            for (int li = 0; li < LANES; li++)
              final_a[li] <= add_y[li];

          if (slot_cnt == SLOT_W'(CLIMIT + 5)) begin
            for (int li = 0; li < LANES; li++)
              final_b[li] <= add_y[li];
            slot_cnt <= '0;
            state <= STORE;
          end
        end

        STORE: begin
          for (int li = 0; li < LANES; li++) begin
            out_buf[li]        <= rounded_a[li];
            out_buf[LANES + li] <= rounded_b[li];
          end
          lane_cnt <= '0;
          state <= OUT;
        end

        OUT: begin
          if (m_axis_tready) begin
            if (lane_cnt == LCNT_W'(lanes_this - 1'b1)) begin
              if (o_cnt == BLK_W'(W_WORDS - 1)) begin
                state <= IDLE;
              end else begin
                o_cnt <= o_cnt + 1'b1;
                i_cnt <= '0;
                slot_cnt <= '0;
                lane_cnt <= '0;
                state <= MAC_FILL;
              end
            end else begin
              lane_cnt <= lane_cnt + 1'b1;
            end
          end
        end

        default: state <= IDLE;
      endcase
    end
  end

endmodule
