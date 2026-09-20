// Grouped-query attention (GQA) unit — Granite 4.0-H-350M attention layers
// (decoder positions 10, 13, 17, 27): GraniteMoeHybridAttention.forward().
//
// Per token (one AXI-Stream transaction):
//   1. Q/K/V projections with four matrix_unit instances (bf16 in/out,
//      sequential fp32 accumulation — bit-compatible with torch F.linear):
//        Q (768->768), K (768->256), V (768->256)
//   2. K/V are appended to an on-chip KV cache at the current position
//      (the position counter is reset only by rst_n, so a new sequence
//      starts with a reset; cache rows beyond the position are never read).
//   3. For each of the 12 query heads (each sharing one of the 4 KV heads,
//      3 query heads per KV head, matching repeat_interleave GQA expansion):
//        score_j = sum_d q_d * k_j[d]                       (fp32, 64 MACs)
//        score_j = bf16(bf16(fp32(score_j)) * 0.015625)     (torch rounds the
//          bf16 QK^T product and the scalar multiply separately)
//        p = softmax_j(score)                               (fp32, accurate exp;
//          probabilities rounded once to bf16 like torch's .to(query.dtype))
//        context_d = sum_j bf16(p_j) * v_j[d]               (bf16 x bf16, fp32
//          accumulation, single bf16 rounding)
//   4. O projection (768->768) via a fourth matrix_unit.
//
// The scaling constant is config.attention_multiplier = 0.015625 (not
// 1/sqrt(head_dim)); there is no RoPE (NoPE) and no bias in the projections.
//
// The output frame is HIDDEN bf16 beats (tlast on the last beat), held stable
// under downstream backpressure.

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off SYNCASYNCNET */

module attention_unit #(
  parameter int HIDDEN       = 768,
  parameter int NUM_HEADS    = 12,
  parameter int NUM_KV_HEADS = 4,
  parameter int HEAD_DIM     = 64,
  parameter int MAX_SEQ      = 64,
  parameter int W_DATA       = 16,
  parameter int H_W          = $clog2(NUM_HEADS),
  parameter int D_W          = $clog2(HEAD_DIM),
  parameter int X_W          = $clog2(HIDDEN),
  parameter int J_W          = $clog2(MAX_SEQ)
) (
  input  logic              clk,
  input  logic              rst_n,

  // Weight load: sel 0=Q, 1=K, 2=V, 3=O (bf16 values, no biases)
  input  logic              load_en,
  input  logic [1:0]        load_sel,
  input  logic [X_W-1:0]    load_out_idx,
  input  logic [X_W-1:0]    load_in_idx,
  input  logic [W_DATA-1:0] load_wdata,

  // AXI-Stream input: hidden state (HIDDEN bf16 beats)
  input  logic              s_axis_tvalid,
  output logic              s_axis_tready,
  input  logic [W_DATA-1:0] s_axis_tdata,
  input  logic              s_axis_tlast,

  // AXI-Stream output: attention output (HIDDEN bf16 beats)
  output logic              m_axis_tvalid,
  input  logic              m_axis_tready,
  output logic [W_DATA-1:0] m_axis_tdata,
  output logic              m_axis_tlast,

  output logic              busy
);

  localparam int QN = NUM_HEADS * HEAD_DIM;      // 768
  localparam int KN = NUM_KV_HEADS * HEAD_DIM;   // 256
  localparam int GROUPS = NUM_HEADS / NUM_KV_HEADS;
  localparam int MAXK = NUM_KV_HEADS * MAX_SEQ * HEAD_DIM;
  localparam int KW   = $clog2(MAXK);

  localparam logic [31:0] C_SCALE = 32'h3C800000;  // attention_multiplier 0.015625

  // ------------------------------------------------------------ storage
  logic [W_DATA-1:0] x_buf   [0:HIDDEN-1];
  logic [W_DATA-1:0] q_buf   [0:QN-1];
  logic [W_DATA-1:0] k_cache [0:MAXK-1];
  logic [W_DATA-1:0] v_cache [0:MAXK-1];
  logic [W_DATA-1:0] ctx_buf [0:HIDDEN-1];

  logic [J_W-1:0] pos;   // tokens already in the cache (current token index)

  // ------------------------------------------------------------ matrix units
  /* verilator lint_off PINCONNECTEMPTY */
  /* verilator lint_off UNUSEDSIGNAL */
  logic        q_s_tvalid, q_s_tready, q_s_tlast, q_m_tvalid, q_m_tready, q_m_tlast;
  logic [15:0] q_s_tdata,  q_m_tdata;
  logic        k_s_tvalid, k_s_tready, k_s_tlast, k_m_tvalid, k_m_tready, k_m_tlast;
  logic [15:0] k_s_tdata,  k_m_tdata;
  logic        v_s_tvalid, v_s_tready, v_s_tlast, v_m_tvalid, v_m_tready, v_m_tlast;
  logic [15:0] v_s_tdata,  v_m_tdata;
  logic        o_s_tvalid, o_s_tready, o_s_tlast, o_m_tvalid, o_m_tlast;
  logic [15:0] o_s_tdata,  o_m_tdata;

  wire ld_q = load_en && (load_sel == 2'd0);
  wire ld_k = load_en && (load_sel == 2'd1);
  wire ld_v = load_en && (load_sel == 2'd2);
  wire ld_o = load_en && (load_sel == 2'd3);

  matrix_unit #(.IN_FEATURES(HIDDEN), .OUT_FEATURES(QN)) u_q (
    .clk(clk), .rst_n(rst_n), .load_en(ld_q),
    .load_out_idx(load_out_idx), .load_in_idx(load_in_idx),
    .load_wdata(load_wdata), .load_is_bias(1'b0),
    .s_axis_tvalid(q_s_tvalid), .s_axis_tready(q_s_tready),
    .s_axis_tdata(q_s_tdata), .s_axis_tlast(q_s_tlast),
    .m_axis_tvalid(q_m_tvalid), .m_axis_tready(q_m_tready),
    .m_axis_tdata(q_m_tdata), .m_axis_tlast(q_m_tlast),
    .busy(/*unused*/)
  );

  matrix_unit #(.IN_FEATURES(HIDDEN), .OUT_FEATURES(KN)) u_k (
    .clk(clk), .rst_n(rst_n), .load_en(ld_k),
    .load_out_idx(load_out_idx), .load_in_idx(load_in_idx),
    .load_wdata(load_wdata), .load_is_bias(1'b0),
    .s_axis_tvalid(k_s_tvalid), .s_axis_tready(k_s_tready),
    .s_axis_tdata(k_s_tdata), .s_axis_tlast(k_s_tlast),
    .m_axis_tvalid(k_m_tvalid), .m_axis_tready(k_m_tready),
    .m_axis_tdata(k_m_tdata), .m_axis_tlast(k_m_tlast),
    .busy(/*unused*/)
  );

  matrix_unit #(.IN_FEATURES(HIDDEN), .OUT_FEATURES(KN)) u_v (
    .clk(clk), .rst_n(rst_n), .load_en(ld_v),
    .load_out_idx(load_out_idx), .load_in_idx(load_in_idx),
    .load_wdata(load_wdata), .load_is_bias(1'b0),
    .s_axis_tvalid(v_s_tvalid), .s_axis_tready(v_s_tready),
    .s_axis_tdata(v_s_tdata), .s_axis_tlast(v_s_tlast),
    .m_axis_tvalid(v_m_tvalid), .m_axis_tready(v_m_tready),
    .m_axis_tdata(v_m_tdata), .m_axis_tlast(v_m_tlast),
    .busy(/*unused*/)
  );

  matrix_unit #(.IN_FEATURES(HIDDEN), .OUT_FEATURES(QN)) u_o (
    .clk(clk), .rst_n(rst_n), .load_en(ld_o),
    .load_out_idx(load_out_idx), .load_in_idx(load_in_idx),
    .load_wdata(load_wdata), .load_is_bias(1'b0),
    .s_axis_tvalid(o_s_tvalid), .s_axis_tready(o_s_tready),
    .s_axis_tdata(o_s_tdata), .s_axis_tlast(o_s_tlast),
    .m_axis_tvalid(o_m_tvalid), .m_axis_tready(m_axis_tready),
    .m_axis_tdata(o_m_tdata), .m_axis_tlast(o_m_tlast),
    .busy(/*unused*/)
  );

  /* verilator lint_on UNUSEDSIGNAL */
  /* verilator lint_on PINCONNECTEMPTY */

  // Output frame is a pure pass-through of the O projection stream.
  assign m_axis_tvalid = o_m_tvalid;
  assign m_axis_tdata  = o_m_tdata;
  assign m_axis_tlast  = o_m_tlast;

  // ------------------------------------------------------------ softmax row
  logic        sp_valid, sp_last, sp_done;
  logic [31:0] sp_data;
  logic [31:0] sp_rd_data;

  attn_softmax_seq #(.N(MAX_SEQ)) u_softmax (
    .clk(clk), .rst_n(rst_n),
    .valid_in(sp_valid), .data_in(sp_data), .last_in(sp_last),
    .done(sp_done),
    .rd_idx(j_cnt), .rd_data(sp_rd_data)
  );

  // ------------------------------------------------------------ score FPUs
  /* verilator lint_off PINCONNECTEMPTY */
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
  /* verilator lint_on PINCONNECTEMPTY */

  logic [31:0] mul_a, mul_b, mul_y;
  logic [31:0] add_a, add_b, add_y;

  logic [15:0] score_bf, scaled_bf, ctx_bf;
  fp32_to_bf16_round u_round_score  (.x(add_y), .y(score_bf));
  fp32_to_bf16_round u_round_scaled (.x(mul_y), .y(scaled_bf));
  fp32_to_bf16_round u_round_ctx    (.x(add_y), .y(ctx_bf));

  // ------------------------------------------------------------ counters
  logic [X_W-1:0] in_cnt, f_cnt, dq_cnt;
  logic [7:0]     dk_cnt, dv_cnt;
  logic [H_W-1:0] h_cnt;
  logic [J_W-1:0] j_cnt;
  logic [D_W-1:0] d_cnt;
  logic [X_W-1:0] do_cnt;

  wire [H_W-1:0] kv_head = h_cnt / H_W'(GROUPS);

  // Flat KV-cache indices.
  wire [KW-1:0] k_wr_addr =
      (dk_cnt / HEAD_DIM) * (MAX_SEQ * HEAD_DIM) + pos * HEAD_DIM +
      (dk_cnt % HEAD_DIM);
  wire [KW-1:0] k_rd_addr =
      kv_head * (MAX_SEQ * HEAD_DIM) + j_cnt * HEAD_DIM + d_cnt;
  wire [KW-1:0] v_wr_addr =
      (dv_cnt / HEAD_DIM) * (MAX_SEQ * HEAD_DIM) + pos * HEAD_DIM +
      (dv_cnt % HEAD_DIM);
  wire [KW-1:0] v_rd_addr =
      kv_head * (MAX_SEQ * HEAD_DIM) + j_cnt * HEAD_DIM + d_cnt;

  wire [X_W-1:0] q_addr = h_cnt * HEAD_DIM + d_cnt;
  wire [X_W-1:0] c_addr = h_cnt * HEAD_DIM + d_cnt;

  // ------------------------------------------------------------ FSM
  typedef enum logic [4:0] {
    IDLE,
    FEED,       // broadcast x_buf into Q/K/V units
    DRQ,        // drain Q projection -> q_buf
    DRK,        // drain K projection -> k_cache[pos]
    DRV,        // drain V projection -> v_cache[pos]
    A_S_INIT,   // start a query head: j = 0
    A_S_MAC,    // score_j dot product over d
    A_S_LAST,   // add final product
    A_S_SCALE,  // bf16 round + multiply by attention_multiplier
    A_S_STORE,  // bf16 round + push score into the softmax row
    A_S_WAIT,   // wait for the softmax row
    A_C_MAC,    // context_d dot product over cached j
    A_C_LAST,   // add final product
    A_C_STORE,  // bf16 round + store context
    FEED_O,     // broadcast ctx_buf into the O projection
    DRAIN_O     // stream O projection output = unit output
  } state_t;

  state_t state;

  assign s_axis_tready = (state == IDLE);
  assign busy          = (state != IDLE);

  assign q_m_tready = (state == DRQ);
  assign k_m_tready = (state == DRK);
  assign v_m_tready = (state == DRV);

  assign sp_valid = (state == A_S_STORE);
  assign sp_last  = (state == A_S_STORE) && (j_cnt == pos);
  assign sp_data  = {scaled_bf, 16'b0};

  always_comb begin
    mul_a = '0;
    mul_b = '0;
    add_a = '0;
    add_b = '0;

    q_s_tvalid = 1'b0;
    k_s_tvalid = 1'b0;
    v_s_tvalid = 1'b0;
    o_s_tvalid = 1'b0;
    q_s_tdata  = '0;
    k_s_tdata  = '0;
    v_s_tdata  = '0;
    o_s_tdata  = '0;

    case (state)
      FEED: begin
        q_s_tvalid = 1'b1;
        q_s_tdata  = x_buf[f_cnt];
        k_s_tvalid = 1'b1;
        k_s_tdata  = x_buf[f_cnt];
        v_s_tvalid = 1'b1;
        v_s_tdata  = x_buf[f_cnt];
      end

      A_S_MAC: begin
        mul_a = {q_buf[q_addr], 16'b0};
        mul_b = {k_cache[k_rd_addr], 16'b0};
        if (d_cnt == D_W'(0)) begin
          add_a = '0;
          add_b = '0;
        end else begin
          add_a = add_y;
          add_b = mul_y;
        end
      end

      A_S_LAST: begin
        add_a = add_y;
        add_b = mul_y;
      end

      A_S_SCALE: begin
        mul_a = {score_bf, 16'b0};
        mul_b = C_SCALE;
      end

      A_C_MAC: begin
        mul_a = sp_rd_data;
        mul_b = {v_cache[v_rd_addr], 16'b0};
        if (j_cnt == J_W'(0)) begin
          add_a = '0;
          add_b = '0;
        end else begin
          add_a = add_y;
          add_b = mul_y;
        end
      end

      A_C_LAST: begin
        add_a = add_y;
        add_b = mul_y;
      end

      FEED_O: begin
        o_s_tvalid = 1'b1;
        o_s_tdata  = ctx_buf[f_cnt];
      end

      default: ;
    endcase
  end

  // tlast for the broadcast feeds (asserted on the final beat)
  wire feed_last  = (state == FEED)   && (f_cnt == X_W'(HIDDEN - 1));
  wire feedo_last = (state == FEED_O) && (f_cnt == X_W'(HIDDEN - 1));
  assign q_s_tlast = feed_last;
  assign k_s_tlast = feed_last;
  assign v_s_tlast = feed_last;
  assign o_s_tlast = feedo_last;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state   <= IDLE;
      in_cnt  <= '0;
      f_cnt   <= '0;
      dq_cnt  <= '0;
      dk_cnt  <= '0;
      dv_cnt  <= '0;
      do_cnt  <= '0;
      h_cnt   <= '0;
      j_cnt   <= '0;
      d_cnt   <= '0;
      pos     <= '0;
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

        // ------------------------------------- Q/K/V projections (parallel)
        FEED: begin
          if (f_cnt == X_W'(HIDDEN - 1)) begin
            f_cnt  <= '0;
            dq_cnt <= '0;
            state  <= DRQ;
          end else begin
            f_cnt <= f_cnt + 1'b1;
          end
        end

        DRQ: begin
          if (q_m_tvalid) begin
            q_buf[dq_cnt] <= q_m_tdata;
            if (dq_cnt == X_W'(QN - 1)) begin
              dq_cnt <= '0;
              dk_cnt <= '0;
              state  <= DRK;
            end else begin
              dq_cnt <= dq_cnt + 1'b1;
            end
          end
        end

        DRK: begin
          if (k_m_tvalid) begin
            k_cache[k_wr_addr] <= k_m_tdata;
            if (dk_cnt == 8'(KN - 1)) begin
              dk_cnt <= '0;
              dv_cnt <= '0;
              state  <= DRV;
            end else begin
              dk_cnt <= dk_cnt + 1'b1;
            end
          end
        end

        DRV: begin
          if (v_m_tvalid) begin
            v_cache[v_wr_addr] <= v_m_tdata;
            if (dv_cnt == 8'(KN - 1)) begin
              dv_cnt <= '0;
              h_cnt  <= '0;
              state  <= A_S_INIT;
            end else begin
              dv_cnt <= dv_cnt + 1'b1;
            end
          end
        end

        // ---------------------------------------------------- per-head attention
        A_S_INIT: begin
          j_cnt <= '0;
          d_cnt <= '0;
          state <= A_S_MAC;
        end

        A_S_MAC: begin
          if (d_cnt == D_W'(HEAD_DIM - 1)) begin
            d_cnt <= '0;
            state <= A_S_LAST;
          end else begin
            d_cnt <= d_cnt + 1'b1;
          end
        end

        A_S_LAST:  state <= A_S_SCALE;
        A_S_SCALE: state <= A_S_STORE;

        A_S_STORE: begin
          if (j_cnt == pos) begin
            state <= A_S_WAIT;
          end else begin
            j_cnt <= j_cnt + 1'b1;
            d_cnt <= '0;
            state <= A_S_MAC;
          end
        end

        A_S_WAIT: begin
          if (sp_done) begin
            j_cnt <= '0;
            d_cnt <= '0;
            state <= A_C_MAC;
          end
        end

        A_C_MAC: begin
          if (j_cnt == pos) begin
            state <= A_C_LAST;
          end else begin
            j_cnt <= j_cnt + 1'b1;
          end
        end

        A_C_LAST: state <= A_C_STORE;

        A_C_STORE: begin
          ctx_buf[c_addr] <= ctx_bf;
          j_cnt <= '0;
          if (d_cnt == D_W'(HEAD_DIM - 1)) begin
            d_cnt <= '0;
            if (h_cnt == H_W'(NUM_HEADS - 1)) begin
              f_cnt <= '0;
              state <= FEED_O;
            end else begin
              h_cnt <= h_cnt + 1'b1;
              state <= A_S_INIT;
            end
          end else begin
            d_cnt <= d_cnt + 1'b1;
            state <= A_C_MAC;
          end
        end

        // ---------------------------------------------------- O projection
        FEED_O: begin
          if (f_cnt == X_W'(HIDDEN - 1)) begin
            f_cnt  <= '0;
            do_cnt <= '0;
            state  <= DRAIN_O;
          end else begin
            f_cnt <= f_cnt + 1'b1;
          end
        end

        DRAIN_O: begin
          if (o_m_tvalid && m_axis_tready) begin
            if (do_cnt == X_W'(QN - 1)) begin
              do_cnt <= '0;
              pos    <= pos + 1'b1;   // cache one more token
              state  <= IDLE;
            end else begin
              do_cnt <= do_cnt + 1'b1;
            end
          end
        end

        default: state <= IDLE;
      endcase
    end
  end

endmodule
