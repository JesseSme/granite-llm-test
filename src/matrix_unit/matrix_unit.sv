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
//   - one binary32 fp_unit for OP_MUL and one for OP_ADD
//   - the accumulator lives in the adder's output register (add_y feeds its
//     own add_a input), so one MAC completes per cycle
//   - per output: IN_FEATURES MAC cycles + last-product add + bias add + output
//
// AXI-Stream:
//   - input : IN_FEATURES beats (tlast expected on the final beat), accepted
//             while idle (s_axis_tready = state == IDLE)
//   - output: OUT_FEATURES beats, tlast on the final beat, held under
//             downstream backpressure (m_axis_tready)

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */

module matrix_unit #(
  parameter int IN_FEATURES  = 768,
  parameter int OUT_FEATURES = 768,
  parameter int W_DATA       = 16,
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

  // ------------------------------------------------------------ storage
  logic [W_DATA-1:0] x_buf    [0:IN_FEATURES-1];
  logic [W_DATA-1:0] w_mem    [0:OUT_FEATURES-1][0:IN_FEATURES-1];
  logic [W_DATA-1:0] bias_mem [0:OUT_FEATURES-1];

  always_ff @(posedge clk) begin
    if (load_en) begin
      if (load_is_bias)
        bias_mem[load_out_idx] <= load_wdata;
      else
        w_mem[load_out_idx][load_in_idx] <= load_wdata;
    end
  end

  // ------------------------------------------------------------ FPU: MUL
  // The multiplier operands are registered (mul_a_q/mul_b_q) so the weight
  // and input memory read is off the multiplier input path. MAC_FILL loads
  // element i_cnt; during MAC(i) the MUL output is still the product of
  // element i-1, exactly as before (the add schedule is unchanged).
  fp_pkg::op_t       mul_mode;
  fp_pkg::rounding_t mul_rm;
  logic [31:0]       mul_a, mul_b, mul_y;
  logic [31:0]       mul_a_q, mul_b_q;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0]        mul_cmp;
  logic [4:0]        mul_flags;
  logic              unused_out_valid_mul;
  /* verilator lint_on UNUSEDSIGNAL */

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_fp_mul (
    .clk(clk), .rst_n(rst_n),
    .mode(mul_mode), .rm(mul_rm),
    .a(mul_a), .b(mul_b), .c('0),
    .y(mul_y), .cmp(mul_cmp), .flags(mul_flags),
    .in_valid(1'b1), .out_valid(unused_out_valid_mul)
  );

  // ------------------------------------------------------------ FPU: ADD
  // The accumulator is held in this unit's output register (add_a = add_y).
  fp_pkg::op_t       add_mode;
  fp_pkg::rounding_t add_rm;
  logic [31:0]       add_a, add_b, add_y;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0]        add_cmp;
  logic [4:0]        add_flags;
  logic              unused_out_valid_add;
  /* verilator lint_on UNUSEDSIGNAL */

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_fp_add (
    .clk(clk), .rst_n(rst_n),
    .mode(add_mode), .rm(add_rm),
    .a(add_a), .b(add_b), .c('0),
    .y(add_y), .cmp(add_cmp), .flags(add_flags),
    .in_valid(1'b1), .out_valid(unused_out_valid_add)
  );

  // Final binary32 -> bfloat16 rounding (one rounding after the bias add).
  logic [15:0] rounded_sum;
  fp32_to_bf16_round u_round (.x(add_y), .y(rounded_sum));

  // ------------------------------------------------------------ FSM
  typedef enum logic [2:0] {
    IDLE,      // accept IN_FEATURES input beats
    MAC_FILL,  // register the first multiplier operands from memory
    MAC,       // issue one multiply-add per cycle
    ADD_LAST,  // add the final product (fp_unit pipeline depth)
    BIAS,      // add bias in binary32
    OUT        // stream one output beat (handshake)
  } state_t;

  state_t state;
  logic [IN_W-1:0]  i_cnt;
  logic [OUT_W-1:0] o_cnt;

  assign s_axis_tready = (state == IDLE);
  assign busy          = (state != IDLE);

  assign m_axis_tvalid = (state == OUT);
  assign m_axis_tlast  = (state == OUT) && (o_cnt == OUT_W'(OUT_FEATURES - 1));
  assign m_axis_tdata  = rounded_sum;

  always_comb begin
    mul_mode = fp_pkg::OP_MUL;
    mul_rm   = fp_pkg::RM_RNE;
    mul_a    = '0;
    mul_b    = '0;

    add_mode = fp_pkg::OP_ADD;
    add_rm   = fp_pkg::RM_RNE;
    add_a    = add_y;   // hold accumulator by default
    add_b    = '0;

    case (state)
      MAC: begin
        mul_a = mul_a_q;
        mul_b = mul_b_q;
        if (i_cnt == IN_W'(0)) begin
          add_a = '0;      // start a fresh accumulation
          add_b = '0;
        end else begin
          add_a = add_y;
          add_b = mul_y;   // product of element i_cnt-1
        end
      end

      ADD_LAST: begin
        add_a = add_y;
        add_b = mul_y;     // final product of the row
      end

      BIAS: begin
        add_a = add_y;
        add_b = {bias_mem[o_cnt], {(32 - W_DATA) {1'b0}}};
      end

      default: ;
    endcase
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state <= IDLE;
      i_cnt <= '0;
      o_cnt <= '0;
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
              state <= MAC_FILL;
            end else begin
              i_cnt <= i_cnt + 1'b1;
            end
          end
        end

        MAC_FILL: begin
          mul_a_q <= {x_buf[i_cnt], {(32 - W_DATA) {1'b0}}};
          mul_b_q <= {w_mem[o_cnt][i_cnt], {(32 - W_DATA) {1'b0}}};
          state <= MAC;
        end

        MAC: begin
          if (i_cnt == IN_W'(IN_FEATURES - 1)) begin
            i_cnt <= '0;
            state <= ADD_LAST;
          end else begin
            // Prefetch element i+1; during MAC(i) the MUL output remains the
            // product of element i-1, so the accumulation order is untouched.
            mul_a_q <= {x_buf[i_cnt + 1'b1], {(32 - W_DATA) {1'b0}}};
            mul_b_q <= {w_mem[o_cnt][i_cnt + 1'b1], {(32 - W_DATA) {1'b0}}};
            i_cnt <= i_cnt + 1'b1;
          end
        end

        ADD_LAST: state <= BIAS;

        BIAS: state <= OUT;

        OUT: begin
          if (m_axis_tready) begin
            if (o_cnt == OUT_W'(OUT_FEATURES - 1)) begin
              state <= IDLE;
            end else begin
              o_cnt <= o_cnt + 1'b1;
              i_cnt <= '0;
              state <= MAC_FILL;
            end
          end
        end

        default: state <= IDLE;
      endcase
    end
  end

endmodule
