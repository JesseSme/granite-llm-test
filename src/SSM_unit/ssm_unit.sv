// Selective State Space Model (S6) recurrent unit — Mamba2 scan core.
//
// Implements the recurrent form of GraniteMoeHybridMambaLayer's SSM step
// (modeling_granitemoehybrid.py, `mamba2_chunk_scan` fallback):
//
//   dtp_t = softplus(dt_t + dt_bias)          (bf16, matching the model)
//   A_h   = -exp(A_log_h)                     (fp32)
//   dA_th = exp(A_h * dtp_th)                 (fp32 decay per head/step)
//   h_t   = dA_th * h_{t-1} + (dtp_th * B_t) * x_t     (fp32 state)
//   y_t   = D_h * x_t + sum_s h_t[s] * C_t[s]          (fp32 output)
//
// All arithmetic in binary32; x/B/C/dt inputs are bfloat16. The model runs the
// mathematically equivalent chunked scan (different rounding/order), so the
// in-loop test compares against the model within tolerance rather than
// bit-exactly (softplus itself reproduces torch's bf16 result exactly).
//
// One token (sequence position) per AXI-Stream transaction:
//   input  frame : x (NUM_HEADS*HEAD_DIM), B (D_STATE), C (D_STATE),
//                  dt (NUM_HEADS)  — all bf16 beats, in that order
//   output frame : y (NUM_HEADS*HEAD_DIM) fp32 beats
// The internal state persists across tokens; reset clears it.
//
// Weight load: A_log, D, dt_bias (bf16 values, one per head).
//
// Parallelism (LANES, default 4): the (head, dim) outputs are independent, so
// LANES consecutive dims of the same head are computed in parallel by LANES
// pipelined MAC datapaths. Each lane keeps the exact per-element dataflow of
// the original single-lane design
//   h[s] = dA*h[s] + (w[s]*x);  acc = acc + h[s]*C[s]   (s ascending)
// with the same operand order, so every y element is bit-identical to LANES=1
// (the lane only changes when an operation is issued, not the operations
// themselves). The lanes cover dims d_base..d_base+LANES-1 of one head; the
// last block of a head masks lanes with d >= HEAD_DIM (they do not write state
// and do not produce output beats).
//
// Per-lane element pipeline (3 MUL + 2 ADD fp_units per lane): the five FP
// operations of element s overlap across elements, issuing one element per
// cycle. With t counted from 0 for the first element of an output:
//   t   : m1 = dA*h[t]      m2 = w[t]*x      m3 = D*x (t = 0 only)
//   t+1 : a1 = m1 + m2
//   t+2 : h[t] <= a1        m3 = a1 * C[t]
//   t+3 : a2 = acc + m3     (acc = a2 of the previous element)
// so one output is ready after D_STATE + 4 cycles; the units' own output
// registers are the pipeline registers (no additional staging), the first a2
// operand is the registered D*x, and the a2 chain is the output accumulator.
// The last pipeline stage drains into out_buf (registers, stable under
// backpressure) and the block's valid beats are streamed in dim order.
//
// The recurrent state (NUM_HEADS*HEAD_DIM*D_STATE fp32 words) is cleared
// sequentially after reset (one word per cycle, ~196k cycles at full size):
// a single-cycle clear of a state this large is not implementable, and a
// parallel clear loop also makes formal/synthesis tools unroll the whole
// array. `s_axis_tready` stays low until the clear finishes.

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off SYNCASYNCNET */

module ssm_unit #(
  parameter int NUM_HEADS = 48,
  parameter int HEAD_DIM  = 32,
  parameter int D_STATE   = 128,
  parameter int W_DATA    = 16,
  parameter int LANES     = 4,   // parallel output lanes (dims of one head)
  parameter int H_W       = $clog2(NUM_HEADS),
  parameter int D_W       = $clog2(HEAD_DIM),
  parameter int S_W       = (D_STATE < 2) ? 1 : $clog2(D_STATE),
  parameter int L_W       = (LANES < 2) ? 1 : $clog2(LANES),
  parameter int T_W       = $clog2(D_STATE + 4)   // pipeline time counter
) (
  input  logic              clk,
  input  logic              rst_n,

  // Weight load: sel 0=A_log, 1=D, 2=dt_bias (bf16 values)
  input  logic              load_en,
  input  logic [1:0]        load_sel,
  input  logic [H_W-1:0]    load_idx,
  input  logic [W_DATA-1:0] load_wdata,

  // AXI-Stream input frame
  input  logic              s_axis_tvalid,
  output logic              s_axis_tready,
  input  logic [W_DATA-1:0] s_axis_tdata,
  input  logic              s_axis_tlast,

  // AXI-Stream output (fp32)
  output logic              m_axis_tvalid,
  input  logic              m_axis_tready,
  output logic [31:0]       m_axis_tdata,
  output logic              m_axis_tlast,

  output logic              busy
);

  localparam int XN = NUM_HEADS * HEAD_DIM;
  localparam int TOTAL_BEATS = XN + 2 * D_STATE + NUM_HEADS;
  localparam int CNT_W = $clog2(TOTAL_BEATS);
  localparam int HSTATE_N = NUM_HEADS * HEAD_DIM * D_STATE;
  localparam int CLR_W = $clog2(HSTATE_N);
  // d_base holds HEAD_DIM, and d_lane = d_base + LANES - 1 must not wrap even
  // when LANES exceeds HEAD_DIM (D_W + 1 keeps the historical width as the
  // floor for parameter compatibility).
  localparam int D_BW = ((D_W + 1) > $clog2(HEAD_DIM + LANES))
                      ? (D_W + 1) : $clog2(HEAD_DIM + LANES);

  // ------------------------------------------------------------ storage
  logic [W_DATA-1:0] a_log_q [0:NUM_HEADS-1];
  logic [W_DATA-1:0] d_q     [0:NUM_HEADS-1];
  logic [W_DATA-1:0] dtb_q   [0:NUM_HEADS-1];

  logic [W_DATA-1:0] x_buf [0:XN-1];
  logic [W_DATA-1:0] b_buf [0:D_STATE-1];
  logic [W_DATA-1:0] c_buf [0:D_STATE-1];
  logic [W_DATA-1:0] dt_buf [0:NUM_HEADS-1];

  logic [31:0] h_state [0:HSTATE_N-1];
  logic [CLR_W-1:0] clr_cnt;

  logic [31:0]       dA_q  [0:NUM_HEADS-1];  // exp(A * dtp)
  logic [31:0]       w_buf [0:D_STATE-1];    // dtp * B_s for the current head

  always_ff @(posedge clk) begin
    if (load_en) begin
      case (load_sel)
        2'd0: a_log_q[load_idx] <= load_wdata;
        2'd1: d_q[load_idx]     <= load_wdata;
        default: dtb_q[load_idx] <= load_wdata;
      endcase
    end
  end

  // ------------------------------------------------------------ fp units
  // Softplus (time-step discretization)
  logic [31:0] sp_z, sp_y;
  logic        sp_start, sp_done;

  fp_softplus_seq u_softplus (
    .clk(clk), .rst_n(rst_n),
    .start(sp_start), .z(sp_z), .y(sp_y), .done(sp_done)
  );

  // exp (for A_h and dA_h) — accurate sequential exp (the shared LUT
  // fp_exp is only ~8-bit accurate, insufficient for the recurrence)
  logic [31:0] exp_x, exp_y;
  logic        exp_start, exp_done;

  fp_exp_seq u_exp (
    .clk(clk), .rst_n(rst_n),
    .start(exp_start), .x(exp_x), .y(exp_y), .done(exp_done)
  );

  assign exp_start = (state == H_AEXP) || (state == H_DEXP);

  // Prep MUL / ADD (per-head preparation and the w = dtp * B loop; the MAC
  // lanes below have their own units).
  fp_pkg::op_t       mul_mode;
  fp_pkg::rounding_t mul_rm;
  logic [31:0]       mul_a, mul_b, mul_y;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0]        mul_cmp;
  logic [4:0]        mul_flags;
  logic              unused_out_valid_mul;
  /* verilator lint_on UNUSEDSIGNAL */

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_mul (
    .clk(clk), .rst_n(rst_n),
    .mode(mul_mode), .rm(mul_rm),
    .a(mul_a), .b(mul_b), .c('0),
    .y(mul_y), .cmp(mul_cmp), .flags(mul_flags),
    .in_valid(1'b1), .out_valid(unused_out_valid_mul)
  );

  fp_pkg::op_t       add_mode;
  fp_pkg::rounding_t add_rm;
  logic [31:0]       add_a, add_b, add_y;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0]        add_cmp;
  logic [4:0]        add_flags;
  logic              unused_out_valid_add;
  /* verilator lint_on UNUSEDSIGNAL */

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_add (
    .clk(clk), .rst_n(rst_n),
    .mode(add_mode), .rm(add_rm),
    .a(add_a), .b(add_b), .c('0),
    .y(add_y), .cmp(add_cmp), .flags(add_flags),
    .in_valid(1'b1), .out_valid(unused_out_valid_add)
  );

  assign mul_mode = fp_pkg::OP_MUL;
  assign mul_rm   = fp_pkg::RM_RNE;
  assign add_mode = fp_pkg::OP_ADD;
  assign add_rm   = fp_pkg::RM_RNE;

  // ------------------------------------------------------------ MAC lanes
  // Per-lane element pipeline: m1 = dA*h, m2 = w*x, a1 = m1+m2 (new h),
  // m3 = a1*C, a2 = acc+m3 (output accumulator). See the header comment.
  logic [31:0] m1_a [0:LANES-1], m1_b [0:LANES-1], m1_y [0:LANES-1];
  logic [31:0] m2_a [0:LANES-1], m2_b [0:LANES-1], m2_y [0:LANES-1];
  logic [31:0] m3_a [0:LANES-1], m3_b [0:LANES-1], m3_y [0:LANES-1];
  logic [31:0] a1_a [0:LANES-1], a1_b [0:LANES-1], a1_y [0:LANES-1];
  logic [31:0] a2_a [0:LANES-1], a2_b [0:LANES-1], a2_y [0:LANES-1];
  logic [31:0] acc_q   [0:LANES-1];   // first a2 operand: D*x
  logic [31:0] out_buf [0:LANES-1];   // drained y, stable under backpressure

  /* verilator lint_off PINCONNECTEMPTY */
  genvar gl;
  generate
    for (gl = 0; gl < LANES; gl++) begin : g_lane
      fp_unit #(.W_EXP(8), .W_MANT(23)) u_m1 (
        .clk(clk), .rst_n(rst_n),
        .mode(fp_pkg::OP_MUL), .rm(fp_pkg::RM_RNE),
        .a(m1_a[gl]), .b(m1_b[gl]), .c('0),
        .y(m1_y[gl]), .cmp(/*unused*/), .flags(/*unused*/),
        .in_valid(1'b1), .out_valid(/*unused*/)
      );
      fp_unit #(.W_EXP(8), .W_MANT(23)) u_m2 (
        .clk(clk), .rst_n(rst_n),
        .mode(fp_pkg::OP_MUL), .rm(fp_pkg::RM_RNE),
        .a(m2_a[gl]), .b(m2_b[gl]), .c('0),
        .y(m2_y[gl]), .cmp(/*unused*/), .flags(/*unused*/),
        .in_valid(1'b1), .out_valid(/*unused*/)
      );
      fp_unit #(.W_EXP(8), .W_MANT(23)) u_m3 (
        .clk(clk), .rst_n(rst_n),
        .mode(fp_pkg::OP_MUL), .rm(fp_pkg::RM_RNE),
        .a(m3_a[gl]), .b(m3_b[gl]), .c('0),
        .y(m3_y[gl]), .cmp(/*unused*/), .flags(/*unused*/),
        .in_valid(1'b1), .out_valid(/*unused*/)
      );
      fp_unit #(.W_EXP(8), .W_MANT(23)) u_a1 (
        .clk(clk), .rst_n(rst_n),
        .mode(fp_pkg::OP_ADD), .rm(fp_pkg::RM_RNE),
        .a(a1_a[gl]), .b(a1_b[gl]), .c('0),
        .y(a1_y[gl]), .cmp(/*unused*/), .flags(/*unused*/),
        .in_valid(1'b1), .out_valid(/*unused*/)
      );
      fp_unit #(.W_EXP(8), .W_MANT(23)) u_a2 (
        .clk(clk), .rst_n(rst_n),
        .mode(fp_pkg::OP_ADD), .rm(fp_pkg::RM_RNE),
        .a(a2_a[gl]), .b(a2_b[gl]), .c('0),
        .y(a2_y[gl]), .cmp(/*unused*/), .flags(/*unused*/),
        .in_valid(1'b1), .out_valid(/*unused*/)
      );
    end
  endgenerate
  /* verilator lint_on PINCONNECTEMPTY */

  // bf16 rounding
  logic [15:0] z_bf16;
  logic [15:0] sp_bf16;
  fp32_to_bf16_round u_round_z  (.x(add_y), .y(z_bf16));
  fp32_to_bf16_round u_round_sp (.x(sp_y),  .y(sp_bf16));

  // ------------------------------------------------------------ FSM
  typedef enum logic [4:0] {
    CLEAR,     // zero the recurrent state (one word per cycle)
    IDLE,
    H_Z,       // issue z = dt + dt_bias
    H_ZDEC,    // latch z, start softplus on bf16(z)
    H_SP,      // wait softplus, store dtp (bf16)
    H_AEXP,    // start exp(A_log)
    H_AWAIT,   // wait, latch A = -exp(A_log)
    H_AMUL,    // issue MUL(A, dtp)
    H_DEXP,    // start exp(A*dtp)
    H_DWAIT,   // wait, store dA
    W_ISSUE,   // issue w_s = dtp * B_s
    W_STORE,   // store w_s, advance s
    MAC,       // pipelined element issue for one LANES-wide output block
    OUT        // stream the block's y beats (handshake)
  } state_t;

  state_t state;
  logic [CNT_W-1:0] in_cnt;
  logic [H_W-1:0]   h_cnt;
  logic [S_W-1:0]   s_cnt;
  logic [D_BW-1:0]  d_base;    // first dim of the current LANES block
  logic [L_W-1:0]   out_cnt;   // output beat counter within the block
  logic [T_W-1:0]   t_cnt;     // MAC pipeline time counter

  logic [31:0] dtp_reg, a_reg;

  // LANES block geometry. int arithmetic: LANES can exceed the D_BW range.
  logic [L_W:0] lanes_this;
  always_comb begin
    if ((HEAD_DIM - int'(d_base)) >= LANES)
      lanes_this = (L_W + 1)'(LANES);
    else
      lanes_this = (L_W + 1)'(HEAD_DIM - int'(d_base));
  end

  assign s_axis_tready = (state == IDLE);
  assign busy          = (state != IDLE);

  // Start softplus in the same cycle the z add result is on the adder output,
  // so its step-0 edge samples the correct z.
  assign sp_start = (state == H_ZDEC);

  assign m_axis_tvalid = (state == OUT);
  assign m_axis_tdata  = out_buf[out_cnt];
  assign m_axis_tlast  = (state == OUT) &&
                         (h_cnt == H_W'(NUM_HEADS - 1)) &&
                         ((d_base + D_BW'(out_cnt)) == D_BW'(HEAD_DIM - 1));

  // ------------------------------------------------------------ op inputs
  // Per-lane combinational helpers (indices/validity for the current block).
  logic [D_BW-1:0]  d_lane  [0:LANES-1];
  logic [D_BW-1:0]  d_rd    [0:LANES-1];
  logic             lane_ok [0:LANES-1];
  logic [CLR_W-1:0] hidx_r  [0:LANES-1];
  logic [CLR_W-1:0] hidx_w  [0:LANES-1];
  logic [31:0]      x_lane  [0:LANES-1];

  always_comb begin
    sp_z  = {z_bf16, {(32 - W_DATA) {1'b0}}};
    exp_x = '0;
    mul_a = '0;
    mul_b = '0;
    add_a = '0;
    add_b = '0;

    for (int l = 0; l < LANES; l++) begin
      m1_a[l] = '0; m1_b[l] = '0;
      m2_a[l] = '0; m2_b[l] = '0;
      m3_a[l] = '0; m3_b[l] = '0;
      a1_a[l] = '0; a1_b[l] = '0;
      a2_a[l] = '0; a2_b[l] = '0;
    end

    // Block geometry and per-lane state/input indices (invalid lanes clamp to
    // dim 0; their state writes and output beats are masked).
    for (int l = 0; l < LANES; l++) begin
      d_lane[l]  = d_base + D_BW'(l);
      lane_ok[l] = (d_lane[l] < D_BW'(HEAD_DIM));
      d_rd[l]    = lane_ok[l] ? d_lane[l] : D_BW'(0);
      hidx_r[l]  = CLR_W'(h_cnt) * CLR_W'(HEAD_DIM * D_STATE)
                 + CLR_W'(d_rd[l]) * CLR_W'(D_STATE)
                 + CLR_W'(t_cnt);
      hidx_w[l]  = CLR_W'(h_cnt) * CLR_W'(HEAD_DIM * D_STATE)
                 + CLR_W'(d_rd[l]) * CLR_W'(D_STATE)
                 + CLR_W'((t_cnt >= T_W'(2)) ? (t_cnt - T_W'(2)) : T_W'(0));
      x_lane[l]  = {x_buf[h_cnt * HEAD_DIM + d_rd[l]],
                    {(32 - W_DATA) {1'b0}}};
    end

    case (state)
      H_Z: begin
        add_a = {dt_buf[h_cnt], {(32 - W_DATA) {1'b0}}};
        add_b = {dtb_q[h_cnt],  {(32 - W_DATA) {1'b0}}};
      end

      H_AEXP: exp_x = {a_log_q[h_cnt], {(32 - W_DATA) {1'b0}}};

      H_AMUL: begin
        mul_a = a_reg;
        mul_b = dtp_reg;
      end

      H_DEXP: exp_x = mul_y;   // A * dtp (issued in H_AMUL)

      W_ISSUE: begin
        mul_a = dtp_reg;
        mul_b = {b_buf[s_cnt], {(32 - W_DATA) {1'b0}}};
      end

      MAC: begin
        for (int l = 0; l < LANES; l++) begin
          // m1 = dA*h[t] and m2 = w[t]*x: elements 0..D_STATE-1 (t = s).
          if (t_cnt < T_W'(D_STATE)) begin
            m1_a[l] = dA_q[h_cnt];
            m1_b[l] = h_state[hidx_r[l]];
            m2_a[l] = w_buf[t_cnt];
            m2_b[l] = x_lane[l];
          end
          // a1 = m1 + m2 (new h) for element t-1.
          if ((t_cnt >= T_W'(1)) && (t_cnt <= T_W'(D_STATE))) begin
            a1_a[l] = m1_y[l];
            a1_b[l] = m2_y[l];
          end
          // m3 = D*x (t = 0, first a2 operand) or m3 = a1*C[t-2].
          if (t_cnt == T_W'(0)) begin
            m3_a[l] = {d_q[h_cnt], {(32 - W_DATA) {1'b0}}};
            m3_b[l] = x_lane[l];
          end else if ((t_cnt >= T_W'(2)) && (t_cnt <= T_W'(D_STATE + 1))) begin
            m3_a[l] = a1_y[l];
            m3_b[l] = {c_buf[t_cnt - T_W'(2)], {(32 - W_DATA) {1'b0}}};
          end
          // a2 = acc + m3 (output accumulation) for element t-3; the first
          // add uses the registered D*x, later ones the a2 unit's own output.
          if ((t_cnt >= T_W'(3)) && (t_cnt <= T_W'(D_STATE + 2))) begin
            a2_a[l] = (t_cnt == T_W'(3)) ? acc_q[l] : a2_y[l];
            a2_b[l] = m3_y[l];
          end
        end
      end

      default: ;
    endcase
  end

  // ------------------------------------------------------------ sequential
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state   <= CLEAR;
      clr_cnt <= '0;
      in_cnt  <= '0;
      h_cnt   <= '0;
      s_cnt   <= '0;
      d_base  <= '0;
      out_cnt <= '0;
      t_cnt   <= '0;
      dtp_reg <= '0;
      a_reg   <= '0;
      for (int l = 0; l < LANES; l++) begin
        acc_q[l]   <= '0;
        out_buf[l] <= '0;
      end
    end else begin
      case (state)
        // ---------------------------------------------------- state clear
        CLEAR: begin
          h_state[clr_cnt] <= 32'h0;
          if (clr_cnt == CLR_W'(HSTATE_N - 1)) begin
            state <= IDLE;
          end else begin
            clr_cnt <= clr_cnt + 1'b1;
          end
        end

        // ---------------------------------------------------- input frame
        IDLE: begin
          if (s_axis_tvalid) begin
            if (in_cnt < XN) begin
              x_buf[in_cnt] <= s_axis_tdata;
            end else if (in_cnt < XN + D_STATE) begin
              b_buf[in_cnt - XN] <= s_axis_tdata;
            end else if (in_cnt < XN + 2 * D_STATE) begin
              c_buf[in_cnt - XN - D_STATE] <= s_axis_tdata;
            end else begin
              dt_buf[in_cnt - XN - 2 * D_STATE] <= s_axis_tdata;
            end

            // A well-formed frame is TOTAL_BEATS beats with tlast on the final
            // beat; the counter bound also ends a frame that omits tlast.
            if (s_axis_tlast || (in_cnt == CNT_W'(TOTAL_BEATS - 1))) begin
              in_cnt <= '0;
              h_cnt  <= '0;
              s_cnt  <= '0;
              state  <= H_Z;
            end else begin
              in_cnt <= in_cnt + 1'b1;
            end
          end
        end

        // ---------------------------------------------------- per-head prep
        H_Z:     state <= H_ZDEC;
        H_ZDEC: state <= H_SP;   // softplus starts combinationally this cycle
        H_SP: begin
          if (sp_done) begin
            dtp_reg <= {sp_bf16, {(32 - W_DATA) {1'b0}}};
            state   <= H_AEXP;
          end
        end
        H_AEXP:  state <= H_AWAIT;              // exp(A_log) started combinationally
        H_AWAIT: begin
          if (exp_done) begin
            a_reg <= {~exp_y[31], exp_y[30:0]}; // A = -exp(A_log)
            state <= H_AMUL;
          end
        end
        H_AMUL:  state <= H_DEXP;               // MUL(A, dtp) issued combinationally
        H_DEXP:  state <= H_DWAIT;              // exp(A*dtp) started combinationally
        H_DWAIT: begin
          if (exp_done) begin
            dA_q[h_cnt] <= exp_y;               // dA = exp(A*dtp)
            s_cnt       <= '0;
            state       <= W_ISSUE;
          end
        end

        // ---------------------------------------------------- w = dtp * B
        W_ISSUE: state <= W_STORE;
        W_STORE: begin
          w_buf[s_cnt] <= mul_y;
          if (s_cnt == S_W'(D_STATE - 1)) begin
            s_cnt   <= '0;
            d_base  <= '0;
            out_cnt <= '0;
            t_cnt   <= '0;
            state   <= MAC;
          end else begin
            s_cnt <= s_cnt + 1'b1;
            state <= W_ISSUE;
          end
        end

        // ---------------------------------------------------- element pipeline
        MAC: begin
          if (t_cnt == T_W'(D_STATE + 3)) begin
            // Last a2 result is on the units' outputs; drain to out_buf.
            for (int l = 0; l < LANES; l++)
              out_buf[l] <= a2_y[l];
            t_cnt   <= '0;
            out_cnt <= '0;
            state   <= OUT;
          end else begin
            t_cnt <= t_cnt + 1'b1;
            if (t_cnt == T_W'(1))
              for (int l = 0; l < LANES; l++)
                acc_q[l] <= m3_y[l];            // registered D*x (t = 0 inputs)
            if ((t_cnt >= T_W'(2)) && (t_cnt <= T_W'(D_STATE + 1)))
              for (int l = 0; l < LANES; l++)
                if (lane_ok[l])
                  h_state[hidx_w[l]] <= a1_y[l]; // h[s] = new h
          end
        end

        // ---------------------------------------------------- output
        OUT: begin
          if (m_axis_tready) begin
            if (out_cnt == L_W'(lanes_this - 1'b1)) begin
              if ((int'(d_base) + LANES) >= HEAD_DIM) begin
                // Last dim block of this head.
                if (h_cnt == H_W'(NUM_HEADS - 1)) begin
                  state <= IDLE;
                end else begin
                  h_cnt <= h_cnt + 1'b1;
                  state <= H_Z;
                end
              end else begin
                d_base <= d_base + D_BW'(LANES);
                t_cnt  <= '0;
                state  <= MAC;
              end
            end else begin
              out_cnt <= out_cnt + 1'b1;
            end
          end
        end

        default: state <= IDLE;
      endcase
    end
  end

endmodule
