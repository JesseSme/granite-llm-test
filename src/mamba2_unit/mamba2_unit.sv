// Full Mamba2 mixer - GraniteMoeHybridMambaLayer.forward() (28 of 32 decoder
// layers; layers 10/13/17/27 use attention_unit).
//
// Per token (one AXI-Stream transaction, hidden states in / mixer output out):
//
//   1. in_proj:      projected = x @ W_in^T                 768 -> 3376 bf16
//                    split: gate = [0,1536), B_C = [1536,3328), dt = [3328,3376)
//   2. causal depthwise conv1d (kernel 4, groups 1792, padding 3) over B_C
//      followed by SiLU (activation of the model's causal_conv1d_fn)
//   3. SSM: x = conv_out[0,1536), B = conv_out[1536,1664), C = [1664,1792),
//      dt from the projection -> ssm_unit recurrence (state carried per token)
//   4. gated norm (GraniteMoeHybridRMSNormGated):
//        y = RMSNorm_fp32( ssm_out * silu(gate) ) * weight      (fp32, then bf16)
//   5. out_proj:     y @ W_out^T                            1536 -> 768 bf16
//
// The conv history (3 previous B_C vectors) and the SSM state live inside the
// sub-units and persist across tokens; rst_n starts a new sequence.
//
// Numerics: in_proj/out_proj and the conv reuse the verified matrix_unit /
// conv1d_unit (sequential fp32 accumulation, one bf16 rounding); the two SiLU
// activations use the accurate silu_seq (fp32 result, bf16-rounded only where
// the model casts back to bf16 - the conv path - while the gate branch stays
// fp32 as in the model). The gated norm is computed in fp32 with a single bf16
// rounding of its output, matching the layer's `.to(dtype)` before out_proj.
//
// Weight load (shared port, load_sel):
//   0 = in_proj (3376x768)   1 = out_proj (768x1536)
//   2 = conv weight (1792x4) 3 = conv bias (1792)
//   4 = gated-norm weight (1536)
//   5/6/7 = SSM A_log / D / dt_bias (48 each)

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off SYNCASYNCNET */

module mamba2_unit #(
  parameter int HIDDEN    = 768,
  parameter int INTER     = 1536,   // NUM_HEADS * HEAD_DIM
  parameter int NUM_HEADS = 48,
  parameter int HEAD_DIM  = 32,
  parameter int D_STATE   = 128,
  parameter int W_DATA    = 16,
  parameter int CONV_CH   = INTER + 2 * D_STATE,
  parameter int PROJ      = INTER + CONV_CH + NUM_HEADS,
  parameter int SSM_FRAME = INTER + 2 * D_STATE + NUM_HEADS,
  parameter int CW_W      = $clog2(CONV_CH),
  parameter int PW_W      = $clog2(PROJ),
  parameter int IW_W      = $clog2(INTER),
  parameter int LI_W      = $clog2((HIDDEN > INTER) ? HIDDEN : INTER),
  parameter int NH_W      = $clog2(NUM_HEADS)
) (
  input  logic              clk,
  input  logic              rst_n,

  input  logic              load_en,
  input  logic [2:0]        load_sel,
  input  logic [PW_W-1:0]   load_out_idx,
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

  localparam logic [31:0] C_EPS = 32'h358637BD;  // 1e-6
  localparam logic [31:0] C_ONE = 32'h3F800000;

  // fp32 bit pattern of the positive integer INTER (INTER < 2^24, exact).
  function automatic int msb_idx(input int unsigned v);
    int r;
    begin
      r = 0;
      for (int i = 0; i < 32; i = i + 1)
        if ((v >> i) != 0) r = i;
      msb_idx = r;
    end
  endfunction

  function automatic logic [31:0] int_to_fp32(input int unsigned v);
    int e;
    logic [22:0] m;
    begin
      e = msb_idx(v);
      m = (v << (23 - e));
      int_to_fp32 = {1'b0, 8'(e + 127), m};
    end
  endfunction

  localparam logic [31:0] C_N = int_to_fp32(INTER);

  // ------------------------------------------------------------ storage
  logic [W_DATA-1:0] x_buf    [0:HIDDEN-1];
  logic [W_DATA-1:0] proj_buf [0:PROJ-1];
  logic [W_DATA-1:0] conv_raw [0:CONV_CH-1];
  logic [W_DATA-1:0] conv_out [0:CONV_CH-1];
  logic [31:0]       ssm_buf  [0:INTER-1];
  logic [31:0]       gsilu    [0:INTER-1];
  logic [W_DATA-1:0] norm_w   [0:INTER-1];

  // ------------------------------------------------------------ sub-units
  /* verilator lint_off PINCONNECTEMPTY */
  /* verilator lint_off UNUSEDSIGNAL */
  logic        ip_s_tvalid, ip_s_tready, ip_s_tlast, ip_m_tvalid, ip_m_tready, ip_m_tlast;
  logic [15:0] ip_s_tdata,  ip_m_tdata;
  logic        op_s_tvalid, op_s_tready, op_s_tlast, op_m_tvalid, op_m_tlast;
  logic [15:0] op_s_tdata,  op_m_tdata;
  logic        cv_valid_i, cv_ready_o, cv_valid_o;
  logic [15:0] cv_data_i,  cv_data_o;
  logic        sl_valid_i, sl_ready_o, sl_valid_o;
  logic [15:0] sl_data_i;
  logic [31:0] sl_data_o;
  logic        ssm_s_tvalid, ssm_s_tready, ssm_s_tlast;
  logic [15:0] ssm_s_tdata;
  logic        ssm_m_tvalid, ssm_m_tready, ssm_m_tlast;
  logic [31:0] ssm_m_tdata;

  matrix_unit #(.IN_FEATURES(HIDDEN), .OUT_FEATURES(PROJ)) u_inproj (
    .clk(clk), .rst_n(rst_n), .load_en(load_en && (load_sel == 3'd0)),
    .load_out_idx(load_out_idx), .load_in_idx(load_in_idx),
    .load_wdata(load_wdata), .load_is_bias(1'b0),
    .s_axis_tvalid(ip_s_tvalid), .s_axis_tready(ip_s_tready),
    .s_axis_tdata(ip_s_tdata), .s_axis_tlast(ip_s_tlast),
    .m_axis_tvalid(ip_m_tvalid), .m_axis_tready(ip_m_tready),
    .m_axis_tdata(ip_m_tdata), .m_axis_tlast(ip_m_tlast),
    .busy(/*unused*/)
  );

  matrix_unit #(.IN_FEATURES(INTER), .OUT_FEATURES(HIDDEN)) u_outproj (
    .clk(clk), .rst_n(rst_n), .load_en(load_en && (load_sel == 3'd1)),
    .load_out_idx(load_out_idx), .load_in_idx(load_in_idx),
    .load_wdata(load_wdata), .load_is_bias(1'b0),
    .s_axis_tvalid(op_s_tvalid), .s_axis_tready(op_s_tready),
    .s_axis_tdata(op_s_tdata), .s_axis_tlast(op_s_tlast),
    .m_axis_tvalid(op_m_tvalid), .m_axis_tready(m_axis_tready),
    .m_axis_tdata(op_m_tdata), .m_axis_tlast(op_m_tlast),
    .busy(/*unused*/)
  );

  conv1d_unit #(.CHANNELS(CONV_CH), .KERNEL(4), .PADDING(3)) u_conv (
    .clk(clk), .rst_n(rst_n),
    .valid_i(cv_valid_i), .data_i(cv_data_i), .ready_o(cv_ready_o),
    .valid_o(cv_valid_o), .data_o(cv_data_o),
    .load_en(load_en && ((load_sel == 3'd2) || (load_sel == 3'd3))),
    .load_ch(load_out_idx[CW_W-1:0]), .load_tap(load_in_idx[1:0]),
    .load_wdata(load_wdata), .load_is_bias(load_sel == 3'd3)
  );

  silu_seq u_silu (
    .clk(clk), .rst_n(rst_n),
    .valid_i(sl_valid_i), .ready_o(sl_ready_o), .data_i(sl_data_i),
    .valid_o(sl_valid_o), .data_o(sl_data_o)
  );

  ssm_unit #(.NUM_HEADS(NUM_HEADS), .HEAD_DIM(HEAD_DIM), .D_STATE(D_STATE)) u_ssm (
    .clk(clk), .rst_n(rst_n),
    .load_en(load_en && (load_sel >= 3'd5)),
    .load_sel(load_sel[1:0] - 2'd1),
    .load_idx(load_out_idx[NH_W-1:0]), .load_wdata(load_wdata),
    .s_axis_tvalid(ssm_s_tvalid), .s_axis_tready(ssm_s_tready),
    .s_axis_tdata(ssm_s_tdata), .s_axis_tlast(ssm_s_tlast),
    .m_axis_tvalid(ssm_m_tvalid), .m_axis_tready(ssm_m_tready),
    .m_axis_tdata(ssm_m_tdata), .m_axis_tlast(ssm_m_tlast),
    .busy(/*unused*/)
  );

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_mul (
    .clk(clk), .rst_n(rst_n),
    .mode(fp_pkg::OP_MUL), .rm(fp_pkg::RM_RNE),
    .a(mul_a), .b(mul_b), .c('0),
    .y(mul_y), .cmp(/*unused*/), .flags(/*unused*/)
  );

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_add (
    .clk(clk), .rst_n(rst_n),
    .mode(fp_pkg::OP_ADD), .rm(fp_pkg::RM_RNE),
    .a(add_a), .b(add_b), .c('0),
    .y(add_y), .cmp(/*unused*/), .flags(/*unused*/)
  );

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_sqrt (
    .clk(clk), .rst_n(rst_n),
    .mode(fp_pkg::OP_SQRT), .rm(fp_pkg::RM_RNE),
    .a(sqrt_a), .b('0), .c('0),
    .y(sqrt_y), .cmp(/*unused*/), .flags(/*unused*/)
  );

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_div (
    .clk(clk), .rst_n(rst_n),
    .mode(fp_pkg::OP_DIV), .rm(fp_pkg::RM_RNE),
    .a(div_a), .b(div_b), .c('0),
    .y(div_y), .cmp(/*unused*/), .flags(/*unused*/)
  );
  /* verilator lint_on UNUSEDSIGNAL */
  /* verilator lint_on PINCONNECTEMPTY */

  logic [31:0] mul_a, mul_b, mul_y;
  logic [31:0] add_a, add_b, add_y;
  logic [31:0] sqrt_a, sqrt_y;
  logic [31:0] div_a, div_b, div_y;

  logic [15:0] conv_silu_bf, norm_out_bf;
  fp32_to_bf16_round u_round_conv (.x(sl_data_o), .y(conv_silu_bf));
  fp32_to_bf16_round u_round_norm (.x(mul_y),    .y(norm_out_bf));

  assign m_axis_tvalid = op_m_tvalid;
  assign m_axis_tdata  = op_m_tdata;
  assign m_axis_tlast  = op_m_tlast;

  // ------------------------------------------------------------ registers
  logic [31:0] xn_reg, sq_reg, sumsq_reg, mean_reg, var_reg, rms_reg, rsq_reg, t_reg;
  logic [15:0] out_reg;

  // ------------------------------------------------------------ counters
  logic [11:0] in_cnt, f_cnt, dp_cnt, cvi_cnt, cvo_cnt, silu_cnt, gs_cnt, o_cnt;
  logic [11:0] ssm_cnt, sm_cnt, sq_cnt, out_cnt;

  // ------------------------------------------------------------ FSM
  typedef enum logic [4:0] {
    IDLE, FEED_IN, DRAIN_IN,
    CONV_FEED, CONV_SILU,
    SSM_FEED, SSM_DRAIN,
    GATE_SILU,
    N_SQ1, N_SQ1_W, N_SQ2, N_SQ2_W, N_SQ3, N_SQ3_W,
    N_MEAN, N_MEAN_W, N_EPS, N_EPS_W, N_SQRT, N_SQRT_W, N_RSQ, N_RSQ_W,
    N_OUT1, N_OUT1_W, N_OUT2, N_OUT2_W, N_OUT3, N_OUT3_W, N_OUT4,
    DRAIN_OUT
  } state_t;

  state_t state;

  assign s_axis_tready = (state == IDLE);
  assign busy          = (state != IDLE);

  // ------------------------------------------------------------ op inputs
  always_comb begin
    mul_a = '0; mul_b = '0;
    add_a = '0; add_b = '0;
    sqrt_a = '0;
    div_a = '0; div_b = '0;

    case (state)
      N_SQ1: begin
        mul_a = ssm_buf[sq_cnt];
        mul_b = gsilu[sq_cnt];
      end
      N_OUT1: begin
        mul_a = ssm_buf[out_cnt];
        mul_b = gsilu[out_cnt];
      end
      N_SQ2: begin
        mul_a = xn_reg; mul_b = xn_reg;
      end
      N_SQ3: begin
        add_a = (sq_cnt == 12'd0) ? 32'h0 : sumsq_reg;
        add_b = sq_reg;
      end
      N_MEAN: begin div_a = sumsq_reg; div_b = C_N; end
      N_EPS:  begin add_a = mean_reg;  add_b = C_EPS; end
      N_SQRT: sqrt_a = var_reg;
      N_RSQ:  begin div_a = C_ONE; div_b = rms_reg; end
      N_OUT2: begin mul_a = xn_reg; mul_b = rsq_reg; end
      N_OUT3: begin mul_a = t_reg;  mul_b = {norm_w[out_cnt], 16'b0}; end
      default: ;
    endcase
  end

  // ------------------------------------------------------------ feeds
  assign ip_s_tvalid = (state == FEED_IN);
  assign ip_s_tdata  = x_buf[f_cnt];
  assign ip_s_tlast  = (state == FEED_IN) && (f_cnt == 12'(HIDDEN - 1));

  assign ip_m_tready = (state == DRAIN_IN);

  assign cv_valid_i = (state == CONV_FEED) && (cvi_cnt < 12'(CONV_CH));
  assign cv_data_i  = proj_buf[INTER + cvi_cnt];

  assign sl_valid_i = ((state == CONV_SILU) && (silu_cnt < 12'(CONV_CH))) ||
                      ((state == GATE_SILU) && (gs_cnt < 12'(INTER)));
  assign sl_data_i  = (state == CONV_SILU) ? conv_raw[silu_cnt]
                                           : proj_buf[gs_cnt];

  assign ssm_s_tvalid = (state == SSM_FEED) && (ssm_cnt < 12'(SSM_FRAME));
  wire [11:0] dt_idx = ssm_cnt - 12'(INTER + 2 * D_STATE);
  assign ssm_s_tdata  = (ssm_cnt < 12'(INTER + 2 * D_STATE))
                        ? conv_out[ssm_cnt]
                        : proj_buf[INTER + CONV_CH + dt_idx];
  assign ssm_s_tlast  = (state == SSM_FEED) && (ssm_cnt == 12'(SSM_FRAME - 1));
  assign ssm_m_tready = (state == SSM_DRAIN);

  assign op_s_tvalid = (state == N_OUT4);
  assign op_s_tdata  = out_reg;
  assign op_s_tlast  = (state == N_OUT4) && (out_cnt == 12'(INTER - 1));

  // ------------------------------------------------------------ sequential
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state     <= IDLE;
      in_cnt    <= '0;  f_cnt   <= '0;  dp_cnt  <= '0;
      cvi_cnt   <= '0;  cvo_cnt <= '0;  silu_cnt <= '0;  gs_cnt <= '0;
      o_cnt     <= '0;  ssm_cnt <= '0;  sm_cnt  <= '0;
      sq_cnt    <= '0;  out_cnt <= '0;
      xn_reg    <= '0;  sq_reg  <= '0;  sumsq_reg <= '0;
      mean_reg  <= '0;  var_reg <= '0;  rms_reg <= '0;
      rsq_reg   <= '0;  t_reg   <= '0;  out_reg <= '0;
    end else begin
      case (state)
        // -------------------------------------------------- input frame
        IDLE: begin
          if (s_axis_tvalid) begin
            x_buf[in_cnt] <= s_axis_tdata;
            if (s_axis_tlast || (in_cnt == 12'(HIDDEN - 1))) begin
              in_cnt <= '0; f_cnt <= '0;
              state  <= FEED_IN;
            end else begin
              in_cnt <= in_cnt + 1'b1;
            end
          end
        end

        // -------------------------------------------------- in_proj
        FEED_IN: begin
          if (f_cnt == 12'(HIDDEN - 1)) begin
            f_cnt <= '0; dp_cnt <= '0;
            state <= DRAIN_IN;
          end else begin
            f_cnt <= f_cnt + 1'b1;
          end
        end

        DRAIN_IN: begin
          if (ip_m_tvalid) begin
            proj_buf[dp_cnt] <= ip_m_tdata;
            if (dp_cnt == 12'(PROJ - 1)) begin
              dp_cnt  <= '0;
              cvi_cnt <= '0;
              cvo_cnt <= '0;
              state   <= CONV_FEED;
            end else begin
              dp_cnt <= dp_cnt + 1'b1;
            end
          end
        end

        // -------------------------------------------------- conv1d (+history)
        CONV_FEED: begin
          if (cv_valid_i && cv_ready_o) cvi_cnt <= cvi_cnt + 1'b1;
          if (cv_valid_o) begin
            conv_raw[cvo_cnt] <= cv_data_o;
            if (cvo_cnt == 12'(CONV_CH - 1)) begin
              cvo_cnt  <= '0;
              silu_cnt <= '0;
              state    <= CONV_SILU;
            end else begin
              cvo_cnt <= cvo_cnt + 1'b1;
            end
          end
        end

        // -------------------------------------------------- conv SiLU -> bf16
        CONV_SILU: begin
          if (sl_valid_o) begin
            conv_out[silu_cnt] <= conv_silu_bf;
            if (silu_cnt == 12'(CONV_CH - 1)) begin
              silu_cnt <= '0;
              ssm_cnt  <= '0;
              state    <= SSM_FEED;
            end else begin
              silu_cnt <= silu_cnt + 1'b1;
            end
          end
        end

        // -------------------------------------------------- SSM
        SSM_FEED: begin
          if (ssm_s_tvalid && ssm_s_tready) begin
            if (ssm_cnt == 12'(SSM_FRAME - 1)) begin
              ssm_cnt <= '0;
              sm_cnt  <= '0;
              state   <= SSM_DRAIN;
            end else begin
              ssm_cnt <= ssm_cnt + 1'b1;
            end
          end
        end

        SSM_DRAIN: begin
          if (ssm_m_tvalid) begin
            ssm_buf[sm_cnt] <= ssm_m_tdata;
            if (sm_cnt == 12'(INTER - 1)) begin
              sm_cnt <= '0; gs_cnt <= '0;
              state  <= GATE_SILU;
            end else begin
              sm_cnt <= sm_cnt + 1'b1;
            end
          end
        end

        // -------------------------------------------------- gate SiLU (fp32)
        GATE_SILU: begin
          if (sl_valid_o) begin
            gsilu[gs_cnt] <= sl_data_o;
            if (gs_cnt == 12'(INTER - 1)) begin
              gs_cnt <= '0; sq_cnt <= '0;
              state  <= N_SQ1;
            end else begin
              gs_cnt <= gs_cnt + 1'b1;
            end
          end
        end

        // -------------------------------------------------- gated norm: sum of squares
        N_SQ1:   state <= N_SQ1_W;
        N_SQ1_W: begin xn_reg <= mul_y; state <= N_SQ2; end
        N_SQ2:   state <= N_SQ2_W;
        N_SQ2_W: begin sq_reg <= mul_y; state <= N_SQ3; end
        N_SQ3:   state <= N_SQ3_W;
        N_SQ3_W: begin
          sumsq_reg <= add_y;
          if (sq_cnt == 12'(INTER - 1)) begin
            sq_cnt <= '0;
            state  <= N_MEAN;
          end else begin
            sq_cnt <= sq_cnt + 1'b1;
            state  <= N_SQ1;
          end
        end

        // -------------------------------------------------- rms / reciprocal
        N_MEAN:   state <= N_MEAN_W;
        N_MEAN_W: begin mean_reg <= div_y; state <= N_EPS; end
        N_EPS:    state <= N_EPS_W;
        N_EPS_W:  begin var_reg <= add_y; state <= N_SQRT; end
        N_SQRT:   state <= N_SQRT_W;
        N_SQRT_W: begin rms_reg <= sqrt_y; state <= N_RSQ; end
        N_RSQ:    state <= N_RSQ_W;
        N_RSQ_W:  begin rsq_reg <= div_y; out_cnt <= '0; state <= N_OUT1; end

        // -------------------------------------------------- norm output -> out_proj
        N_OUT1:   state <= N_OUT1_W;
        N_OUT1_W: begin xn_reg <= mul_y; state <= N_OUT2; end
        N_OUT2:   state <= N_OUT2_W;
        N_OUT2_W: begin t_reg <= mul_y; state <= N_OUT3; end
        N_OUT3:   state <= N_OUT3_W;
        N_OUT3_W: begin out_reg <= norm_out_bf; state <= N_OUT4; end
        N_OUT4: begin
          if (out_cnt == 12'(INTER - 1)) begin
            out_cnt <= '0; o_cnt <= '0;
            state   <= DRAIN_OUT;
          end else begin
            out_cnt <= out_cnt + 1'b1;
            state   <= N_OUT1;
          end
        end

        // -------------------------------------------------- out_proj output
        DRAIN_OUT: begin
          if (op_m_tvalid && m_axis_tready) begin
            if (o_cnt == 12'(HIDDEN - 1)) begin
              o_cnt <= '0;
              state <= IDLE;
            end else begin
              o_cnt <= o_cnt + 1'b1;
            end
          end
        end

        default: state <= IDLE;
      endcase

      // norm weight load (persists across tokens)
      if (load_en && (load_sel == 3'd4))
        norm_w[load_out_idx[IW_W-1:0]] <= load_wdata;
    end
  end

endmodule
