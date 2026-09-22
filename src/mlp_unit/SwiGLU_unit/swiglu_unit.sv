// SwiGLU MLP block — GraniteMoeHybridMLP.forward() (all 32 decoder layers).
//
//   combined = gate_up_proj(x)                 (HIDDEN -> 2*INTER, no bias)
//   gate, up = combined.chunk(2, dim=-1)       gate = first INTER, up = last
//   gated    = silu(gate) * up                 (bf16 element-wise)
//   y        = down_proj(gated)                (INTER -> HIDDEN, no bias)
//
// x, weights and all intermediate values are bfloat16; both projections reuse
// matrix_unit (sequential fp32 accumulation, single bf16 rounding).
//
// The activation is an accurate inline silu instead of the LUT-based
// silu_unit: sigmoid(x) = 1/(1+exp(-x)) evaluated in binary32 with the
// fp_exp_seq polynomial exponential, then silu = x*sigmoid(x) rounded once to
// bf16. The LUT sigmoid (<0.5% error, 1/32 input grid) compounds through the
// down projection to up to ~8% relative output error, whereas the accurate
// path matches torch's bf16 F.silu to within one bf16 rounding step.
//
// The gating multiply is a binary32 fp_unit multiply followed by one bf16
// rounding, which is exactly a bf16 multiply with RNE (bf16 x bf16 products
// are exact in binary32).
//
// AXI-Stream: input frame = HIDDEN bf16 beats (tlast on the last), output
// frame = HIDDEN bf16 beats (tlast on the last), held under downstream
// backpressure. Weights are loaded through a shared port:
//   load_sel 0 = gate_up (OUT 2*INTER x IN HIDDEN)
//   load_sel 1 = down    (OUT HIDDEN  x IN INTER)

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off SYNCASYNCNET */

module swiglu_unit #(
  parameter int HIDDEN = 768,
  parameter int INTER  = 2048,
  parameter int W_DATA = 16,
  parameter int X_W    = $clog2(HIDDEN),
  parameter int G_W    = $clog2(2 * INTER),
  parameter int I_W    = $clog2(INTER),
  // Load indices must address both projections.
  parameter int LO_W   = $clog2(((2 * INTER) > HIDDEN) ? (2 * INTER) : HIDDEN),
  parameter int LI_W   = $clog2((HIDDEN > INTER) ? HIDDEN : INTER)
) (
  input  logic              clk,
  input  logic              rst_n,

  input  logic              load_en,
  input  logic              load_sel,
  input  logic [LO_W-1:0]   load_out_idx,
  input  logic [LI_W-1:0]   load_in_idx,
  input  logic [W_DATA-1:0] load_wdata,

  input  logic              s_axis_tvalid,
  output logic              s_axis_tready,
  input  logic [W_DATA-1:0] s_axis_tdata,
  input  logic              s_axis_tlast,

  output logic              m_axis_tvalid,
  input  logic              m_axis_tready,
  output logic [W_DATA-1:0] m_axis_tdata,
  output logic              m_axis_tlast,

  output logic              busy
);

  localparam int  GU     = 2 * INTER;
  localparam logic [31:0] C_ONE = 32'h3F800000;

  // ------------------------------------------------------------ storage
  logic [W_DATA-1:0] x_buf  [0:HIDDEN-1];
  logic [W_DATA-1:0] gu_buf [0:GU-1];

  /* verilator lint_off PINCONNECTEMPTY */
  /* verilator lint_off UNUSEDSIGNAL */
  logic        gu_s_tvalid, gu_s_tready, gu_s_tlast, gu_m_tvalid, gu_m_tready, gu_m_tlast;
  logic [15:0] gu_s_tdata,  gu_m_tdata;
  logic        dn_s_tvalid, dn_s_tready, dn_s_tlast, dn_m_tvalid, dn_m_tlast;
  logic [15:0] dn_s_tdata,  dn_m_tdata;

  matrix_unit #(.IN_FEATURES(HIDDEN), .OUT_FEATURES(GU)) u_gateup (
    .clk(clk), .rst_n(rst_n), .load_en(load_en && !load_sel),
    .load_out_idx(load_out_idx), .load_in_idx(load_in_idx),
    .load_wdata(load_wdata), .load_is_bias(1'b0),
    .s_axis_tvalid(gu_s_tvalid), .s_axis_tready(gu_s_tready),
    .s_axis_tdata(gu_s_tdata), .s_axis_tlast(gu_s_tlast),
    .m_axis_tvalid(gu_m_tvalid), .m_axis_tready(gu_m_tready),
    .m_axis_tdata(gu_m_tdata), .m_axis_tlast(gu_m_tlast),
    .busy(/*unused*/)
  );

  matrix_unit #(.IN_FEATURES(INTER), .OUT_FEATURES(HIDDEN)) u_down (
    .clk(clk), .rst_n(rst_n), .load_en(load_en && load_sel),
    .load_out_idx(load_out_idx), .load_in_idx(load_in_idx),
    .load_wdata(load_wdata), .load_is_bias(1'b0),
    .s_axis_tvalid(dn_s_tvalid), .s_axis_tready(dn_s_tready),
    .s_axis_tdata(dn_s_tdata), .s_axis_tlast(dn_s_tlast),
    .m_axis_tvalid(dn_m_tvalid), .m_axis_tready(m_axis_tready),
    .m_axis_tdata(dn_m_tdata), .m_axis_tlast(dn_m_tlast),
    .busy(/*unused*/)
  );

  // ------------------------------------------------------------ silu units
  logic [31:0] exp_x, exp_y;
  logic        exp_start, exp_done;

  fp_exp_seq u_exp (
    .clk(clk), .rst_n(rst_n),
    .start(exp_start), .x(exp_x), .y(exp_y), .done(exp_done)
  );

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_add (
    .clk(clk), .rst_n(rst_n),
    .mode(fp_pkg::OP_ADD), .rm(fp_pkg::RM_RNE),
    .a(add_a), .b(add_b), .c('0),
    .y(add_y), .cmp(/*unused*/), .flags(/*unused*/)
  );

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_div (
    .clk(clk), .rst_n(rst_n),
    .mode(fp_pkg::OP_DIV), .rm(fp_pkg::RM_RNE),
    .a(div_a), .b(div_b), .c('0),
    .y(div_y), .cmp(/*unused*/), .flags(/*unused*/)
  );

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_mul (
    .clk(clk), .rst_n(rst_n),
    .mode(fp_pkg::OP_MUL), .rm(fp_pkg::RM_RNE),
    .a(mul_a), .b(mul_b), .c('0),
    .y(mul_y), .cmp(/*unused*/), .flags(/*unused*/)
  );

  logic [31:0] add_a, add_b, add_y;
  logic [31:0] div_a, div_b, div_y;
  logic [31:0] mul_a, mul_b, mul_y;

  logic [31:0] e_reg, sig_reg;      // 1+exp(-x) and sigmoid(x)
  logic [15:0] act_reg, gated_reg;  // bf16 silu(x) and bf16 silu(x)*up
  logic [15:0] act_bf, gated_bf;
  fp32_to_bf16_round u_round_act   (.x(mul_y), .y(act_bf));
  fp32_to_bf16_round u_round_gated (.x(mul_y), .y(gated_bf));
  /* verilator lint_on UNUSEDSIGNAL */
  /* verilator lint_on PINCONNECTEMPTY */

  // Output frame is a pure pass-through of the down-projection stream.
  assign m_axis_tvalid = dn_m_tvalid;
  assign m_axis_tdata  = dn_m_tdata;
  assign m_axis_tlast  = dn_m_tlast;

  // ------------------------------------------------------------ counters
  logic [X_W-1:0] in_cnt, f_cnt, o_cnt;
  logic [G_W-1:0] d_cnt;
  logic [I_W-1:0] i_cnt;

  // x widened with the sign flipped: exp argument -x (bit 31 is the bf16 sign)
  wire [31:0] neg_x = {~gu_buf[i_cnt][15], gu_buf[i_cnt][14:0], 16'b0};
  wire [31:0] gate_w   = {gu_buf[i_cnt], 16'b0};
  wire [31:0] up_w  = {gu_buf[INTER + i_cnt], 16'b0};

  // ------------------------------------------------------------ FSM
  typedef enum logic [3:0] {
    IDLE,       // accept HIDDEN input beats
    FEED,       // stream x_buf into the gate+up projection
    DRAIN_GU,   // collect the 2*INTER combined outputs
    G_EXP,      // start exp(-gate)
    G_EXP_W,    // wait for the exponential
    G_ADD,      // 1 + exp(-gate)
    G_ADD_W,
    G_DIV,      // sigmoid = 1 / (1 + exp(-gate))
    G_DIV_W,
    G_SILU,     // silu = gate * sigmoid
    G_SILU_W,
    G_GATE,     // gated = silu * up
    G_GATE_W,
    G_STORE,    // bf16 round and feed the down projection
    WAIT_DOWN   // stream the down projection output = unit output
  } state_t;

  state_t state;

  assign s_axis_tready = (state == IDLE);
  assign busy          = (state != IDLE);

  assign gu_m_tready   = (state == DRAIN_GU);
  assign gu_s_tvalid   = (state == FEED);
  assign gu_s_tdata    = x_buf[f_cnt];
  assign gu_s_tlast    = (state == FEED) && (f_cnt == X_W'(HIDDEN - 1));

  // The down projection is idle while G_STORE feeds it, so its input is never
  // stalled (s_axis_tready stays high until the tlast beat is accepted).
  assign dn_s_tvalid = (state == G_STORE);
  assign dn_s_tdata  = gated_reg;
  assign dn_s_tlast  = (state == G_STORE) && (i_cnt == I_W'(INTER - 1));

  assign exp_start = (state == G_EXP);
  assign exp_x     = neg_x;

  always_comb begin
    add_a = '0; add_b = '0;
    div_a = '0; div_b = '0;
    mul_a = '0; mul_b = '0;

    case (state)
      G_ADD:  begin add_a = C_ONE; add_b = exp_y;  end
      G_DIV:  begin div_a = C_ONE; div_b = e_reg;  end
      G_SILU: begin mul_a = gate_w;   mul_b = sig_reg; end
      G_GATE: begin mul_a = {act_reg, 16'b0}; mul_b = up_w; end
      default: ;
    endcase
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state  <= IDLE;
      in_cnt <= '0;
      f_cnt  <= '0;
      d_cnt  <= '0;
      i_cnt  <= '0;
      o_cnt  <= '0;
      e_reg  <= '0;
      sig_reg <= '0;
      act_reg <= '0;
      gated_reg <= '0;
    end else begin
      case (state)
        // ---------------------------------------------------- input frame
        IDLE: begin
          if (s_axis_tvalid) begin
            x_buf[in_cnt] <= s_axis_tdata;
            if (s_axis_tlast || (in_cnt == X_W'(HIDDEN - 1))) begin
              in_cnt <= '0;
              f_cnt  <= '0;
              state  <= FEED;
            end else begin
              in_cnt <= in_cnt + 1'b1;
            end
          end
        end

        // ---------------------------------------------------- gate+up proj
        FEED: begin
          if (f_cnt == X_W'(HIDDEN - 1)) begin
            f_cnt <= '0;
            d_cnt <= '0;
            state <= DRAIN_GU;
          end else begin
            f_cnt <= f_cnt + 1'b1;
          end
        end

        DRAIN_GU: begin
          if (gu_m_tvalid) begin
            gu_buf[d_cnt] <= gu_m_tdata;
            if (d_cnt == G_W'(GU - 1)) begin
              d_cnt <= '0;
              i_cnt <= '0;
              state <= G_EXP;
            end else begin
              d_cnt <= d_cnt + 1'b1;
            end
          end
        end

        // ---------------------------------------------------- accurate silu
        G_EXP:   state <= G_EXP_W;
        G_EXP_W: if (exp_done) state <= G_ADD;
        G_ADD:   state <= G_ADD_W;
        G_ADD_W: begin e_reg   <= add_y; state <= G_DIV; end
        G_DIV:   state <= G_DIV_W;
        G_DIV_W: begin sig_reg <= div_y; state <= G_SILU; end
        G_SILU:  state <= G_SILU_W;
        G_SILU_W: begin act_reg <= act_bf; state <= G_GATE; end
        G_GATE:  state <= G_GATE_W;
        G_GATE_W: begin gated_reg <= gated_bf; state <= G_STORE; end

        G_STORE: begin
          if (i_cnt == I_W'(INTER - 1)) begin
            o_cnt <= '0;
            state <= WAIT_DOWN;
          end else begin
            i_cnt <= i_cnt + 1'b1;
            state <= G_EXP;
          end
        end

        // ---------------------------------------------------- down proj
        WAIT_DOWN: begin
          if (dn_m_tvalid && m_axis_tready) begin
            if (o_cnt == X_W'(HIDDEN - 1)) begin
              o_cnt <= '0;
              state <= IDLE;
            end else begin
              o_cnt <= o_cnt + 1'b1;
            end
          end
        end

        default: state <= IDLE;
      endcase
    end
  end

endmodule
