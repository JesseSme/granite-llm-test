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
//   - LANES complete binary32 MAC datapaths (own MUL fp_unit, ADD fp_unit and
//     accumulator) compute LANES different output rows simultaneously. The
//     lanes are independent rows and never exchange data, so every row keeps
//     the exact sequential fp32 accumulation order (bit-identical to LANES=1).
//   - the accumulator lives in the adder's output register (add_y feeds its
//     own add_a input), so one MAC completes per cycle per lane
//   - the multiplier operands are registered (MAC_FILL loads element 0, MAC
//     prefetches element i+1) so the weight/input memory read is off the
//     multiplier input path; during MAC(i) the MUL output is still the product
//     of element i-1 and the add sequence is unchanged
//   - per block: IN_FEATURES MAC cycles + last-product add + bias add + one
//     STORE cycle (latches the LANES biased sums) + LANES output beats
//   - weight rows are packed LANES per memory word (row = word*LANES+lane), so
//     one read serves every lane; bias_mem stays per row
//
// AXI-Stream:
//   - input : IN_FEATURES beats (tlast expected on the final beat), accepted
//             while idle (s_axis_tready = state == IDLE)
//   - output: OUT_FEATURES beats, tlast on the final beat, held under
//             downstream backpressure (m_axis_tready); blocks stream lanes
//             0..LANES-1 in row order and the final block masks to the tail
//             (supports OUT_FEATURES % LANES != 0)

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
  localparam int W_WORDS = (OUT_FEATURES + LANES - 1) / LANES;  // row blocks
  localparam int N_ROWS  = W_WORDS * LANES;                     // padded rows
  localparam int W_PACK  = LANES * W_DATA;
  localparam int BLK_W   = (W_WORDS < 2) ? 1 : $clog2(W_WORDS);
  localparam int LANE_W  = (LANES < 2) ? 1 : $clog2(LANES);
  localparam int ROW_W   = (N_ROWS < 2) ? 1 : $clog2(N_ROWS + 1);

  // ------------------------------------------------------------ storage
  logic [W_DATA-1:0] x_buf    [0:IN_FEATURES-1];
  logic [W_PACK-1:0] w_mem    [0:W_WORDS-1][0:IN_FEATURES-1];
  logic [W_DATA-1:0] bias_mem [0:N_ROWS-1];

  always_ff @(posedge clk) begin
    if (load_en) begin
      if (load_is_bias)
        bias_mem[load_out_idx] <= load_wdata;
      else
        w_mem[load_out_idx / LANES][load_in_idx]
             [(load_out_idx % LANES) * W_DATA +: W_DATA] <= load_wdata;
    end
  end

  // ------------------------------------------------------------ MAC lanes
  // LANES independent MAC datapaths sharing x_buf and the input counter.
  logic [31:0]       mul_a;      // shared x operand (combinationally driven)
  logic [31:0]       mul_a_q;    // registered x operand
  logic [31:0]       mul_b     [0:LANES-1];
  logic [31:0]       mul_b_q   [0:LANES-1];
  logic [31:0]       mul_y     [0:LANES-1];
  logic [31:0]       add_a     [0:LANES-1];
  logic [31:0]       add_b     [0:LANES-1];
  logic [31:0]       add_y     [0:LANES-1];
  logic [W_DATA-1:0] rounded_sum [0:LANES-1];
  logic [W_DATA-1:0] out_buf     [0:LANES-1];

  genvar l;
  generate
    for (l = 0; l < LANES; l++) begin : g_lane
      fp_pkg::op_t       mul_mode;
      fp_pkg::rounding_t mul_rm;
      /* verilator lint_off UNUSEDSIGNAL */
      logic [1:0]        mul_cmp;
      logic [4:0]        mul_flags;
      logic              unused_out_valid_mul;
      /* verilator lint_on UNUSEDSIGNAL */

      assign mul_mode = fp_pkg::OP_MUL;
      assign mul_rm   = fp_pkg::RM_RNE;

      fp_unit #(.W_EXP(8), .W_MANT(23)) u_fp_mul (
        .clk(clk), .rst_n(rst_n),
        .mode(mul_mode), .rm(mul_rm),
        .a(mul_a), .b(mul_b[l]), .c('0),
        .y(mul_y[l]), .cmp(mul_cmp), .flags(mul_flags),
        .in_valid(1'b1), .out_valid(unused_out_valid_mul)
      );

      fp_pkg::op_t       add_mode;
      fp_pkg::rounding_t add_rm;
      /* verilator lint_off UNUSEDSIGNAL */
      logic [1:0]        add_cmp;
      logic [4:0]        add_flags;
      logic              unused_out_valid_add;
      /* verilator lint_on UNUSEDSIGNAL */

      assign add_mode = fp_pkg::OP_ADD;
      assign add_rm   = fp_pkg::RM_RNE;

      fp_unit #(.W_EXP(8), .W_MANT(23)) u_fp_add (
        .clk(clk), .rst_n(rst_n),
        .mode(add_mode), .rm(add_rm),
        .a(add_a[l]), .b(add_b[l]), .c('0),
        .y(add_y[l]), .cmp(add_cmp), .flags(add_flags),
        .in_valid(1'b1), .out_valid(unused_out_valid_add)
      );

      // Final binary32 -> bfloat16 rounding (one rounding after the bias add).
      fp32_to_bf16_round u_round (.x(add_y[l]), .y(rounded_sum[l]));
    end
  endgenerate

  // ------------------------------------------------------------ FSM
  typedef enum logic [2:0] {
    IDLE,      // accept IN_FEATURES input beats
    MAC_FILL,  // register the first multiplier operands from memory
    MAC,       // one multiply-add per cycle per lane
    ADD_LAST,  // add the final product (fp_unit pipeline depth)
    BIAS,      // add bias in binary32
    STORE,     // latch the biased sums (stable under backpressure)
    OUT        // stream the block's output beats (lane order, then next block)
  } state_t;

  state_t state;
  logic [IN_W-1:0]   i_cnt;
  logic [BLK_W-1:0]  o_cnt;      // row block index (rows o_cnt*LANES + l)
  logic [LANE_W-1:0] lane_cnt;   // lane currently streaming in OUT

  // Absolute first row of the current block and the number of valid rows in
  // it (masked tail pass when OUT_FEATURES is not a multiple of LANES).
  logic [ROW_W-1:0] row_base;
  logic [ROW_W-1:0] rows_left;
  logic [ROW_W-1:0] lanes_this;
  assign row_base   = ROW_W'(o_cnt) * ROW_W'(LANES);
  assign rows_left  = ROW_W'(OUT_FEATURES) - row_base;
  assign lanes_this = (rows_left < ROW_W'(LANES)) ? rows_left : ROW_W'(LANES);

  assign s_axis_tready = (state == IDLE);
  assign busy          = (state != IDLE);

  assign m_axis_tvalid = (state == OUT);
  assign m_axis_tlast  = (state == OUT)
                      && (o_cnt == BLK_W'(W_WORDS - 1))
                      && (lane_cnt == LANE_W'(lanes_this - 1'b1));
  assign m_axis_tdata  = out_buf[lane_cnt];

  always_comb begin
    mul_a = '0;

    for (int li = 0; li < LANES; li++) begin
      mul_b[li] = '0;
      add_a[li] = add_y[li];   // hold accumulator by default
      add_b[li] = '0;
    end

    case (state)
      MAC: begin
        mul_a = mul_a_q;
        for (int li = 0; li < LANES; li++) begin
          mul_b[li] = mul_b_q[li];
          if (i_cnt == IN_W'(0)) begin
            add_a[li] = '0;      // start a fresh accumulation
            add_b[li] = '0;
          end else begin
            add_a[li] = add_y[li];
            add_b[li] = mul_y[li];   // product of element i_cnt-1
          end
        end
      end

      ADD_LAST: begin
        for (int li = 0; li < LANES; li++) begin
          add_a[li] = add_y[li];
          add_b[li] = mul_y[li];     // final product of the row
        end
      end

      BIAS: begin
        for (int li = 0; li < LANES; li++) begin
          add_a[li] = add_y[li];
          add_b[li] = {bias_mem[row_base + li], {(32 - W_DATA) {1'b0}}};
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
              state <= MAC_FILL;
            end else begin
              i_cnt <= i_cnt + 1'b1;
            end
          end
        end

        MAC_FILL: begin
          mul_a_q <= {x_buf[i_cnt], {(32 - W_DATA) {1'b0}}};
          for (int li = 0; li < LANES; li++)
            mul_b_q[li] <= {w_mem[o_cnt][i_cnt][(li * W_DATA) +: W_DATA],
                            {(32 - W_DATA) {1'b0}}};
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
            for (int li = 0; li < LANES; li++)
              mul_b_q[li] <= {w_mem[o_cnt][i_cnt + 1'b1][(li * W_DATA) +: W_DATA],
                              {(32 - W_DATA) {1'b0}}};
            i_cnt <= i_cnt + 1'b1;
          end
        end

        ADD_LAST: state <= BIAS;

        BIAS: state <= STORE;

        STORE: begin
          for (int li = 0; li < LANES; li++)
            out_buf[li] <= rounded_sum[li];
          lane_cnt <= '0;
          state <= OUT;
        end

        OUT: begin
          if (m_axis_tready) begin
            if (lane_cnt == LANE_W'(lanes_this - 1'b1)) begin
              if (o_cnt == BLK_W'(W_WORDS - 1)) begin
                state <= IDLE;
              end else begin
                o_cnt <= o_cnt + 1'b1;
                i_cnt <= '0;
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
