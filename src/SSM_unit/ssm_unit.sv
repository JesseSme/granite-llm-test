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
// LANES consecutive dims of one head are computed in parallel by lanes with
// 3 MUL + 2 ADD fp_unit instances each. Masked lanes (d >= HEAD_DIM) do not
// write state and do not contribute output beats.
//
// fp_unit protocol: every operation is started by a 1-cycle in_valid pulse and
// the result is latched on out_valid (binary32 latencies: MUL 4, ADD 3 cycles
// from the start cycle). The lane element pipeline is scheduled around the
// add-latency-bound accumulator:
//
//   t = 3s     : m1 = mul(dA, h[s])      m2 = mul(w[s], x)
//   t = 3s + 4 : a1 = add(m1, m2)        (new h[s], written back at 3s + 7)
//   t = 3s + 7 : m3 = mul(a1, C[s])
//   t = 3s + 11: a2 = add(acc, m3)       (accumulator; a2 of s-1 is 3 cycles
//                                         earlier, so the chain never stalls)
//   t = 0      : m3 carries the D*x seed (latched into acc_q at t = 4)
// One LANES-wide output block takes 3*D_STATE + 12 cycles, then the block's
// valid beats are streamed from out_buf in dim order. The per-element
// operations and operands match the previous LANES=1 design, so results are
// bit-identical.
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
  parameter int T_W       = $clog2(3 * D_STATE + 13)  // MAC time counter
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
  // w-stream counter width: at least S_W (kept for interface compatibility)
  // and enough to count up to D_STATE.
  localparam int W_CNT_W = (S_W >= $clog2(D_STATE + 1)) ? S_W : $clog2(D_STATE + 1);

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

  // exp (for A_h and dA_h) — accurate sequential exp
  logic [31:0] exp_x, exp_y;
  logic        exp_start, exp_done;

  fp_exp_seq u_exp (
    .clk(clk), .rst_n(rst_n),
    .start(exp_start), .x(exp_x), .y(exp_y), .done(exp_done)
  );

  assign exp_start = (state == H_AEXP) || (state == H_DEXP);

  // Prep MUL / ADD (per-head preparation and the w = dtp * B stream; the MAC
  // lanes below have their own units).
  fp_pkg::op_t       mul_mode;
  fp_pkg::rounding_t mul_rm;
  logic [31:0]       mul_a, mul_b, mul_y;
  logic              prep_mul_valid, prep_mul_ov;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0]        mul_cmp;
  logic [4:0]        mul_flags;
  /* verilator lint_on UNUSEDSIGNAL */

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_mul (
    .clk(clk), .rst_n(rst_n),
    .mode(mul_mode), .rm(mul_rm),
    .a(mul_a), .b(mul_b), .c('0),
    .y(mul_y), .cmp(mul_cmp), .flags(mul_flags),
    .in_valid(prep_mul_valid), .out_valid(prep_mul_ov)
  );

  fp_pkg::op_t       add_mode;
  fp_pkg::rounding_t add_rm;
  logic [31:0]       add_a, add_b, add_y;
  logic              prep_add_valid, prep_add_ov;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0]        add_cmp;
  logic [4:0]        add_flags;
  /* verilator lint_on UNUSEDSIGNAL */

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_add (
    .clk(clk), .rst_n(rst_n),
    .mode(add_mode), .rm(add_rm),
    .a(add_a), .b(add_b), .c('0),
    .y(add_y), .cmp(add_cmp), .flags(add_flags),
    .in_valid(prep_add_valid), .out_valid(prep_add_ov)
  );

  assign mul_mode = fp_pkg::OP_MUL;
  assign mul_rm   = fp_pkg::RM_RNE;
  assign add_mode = fp_pkg::OP_ADD;
  assign add_rm   = fp_pkg::RM_RNE;

  // ------------------------------------------------------------ MAC lanes
  // Per-lane datapath: m1 = dA*h, m2 = w*x, a1 = m1+m2 (new h),
  // m3 = a1*C, a2 = acc+m3. See the header for the 3-cycle element schedule.
  logic [31:0] m1_a [0:LANES-1], m1_b [0:LANES-1], m1_y [0:LANES-1];
  logic [31:0] m2_a [0:LANES-1], m2_b [0:LANES-1], m2_y [0:LANES-1];
  logic [31:0] m3_a [0:LANES-1], m3_b [0:LANES-1], m3_y [0:LANES-1];
  logic [31:0] a1_a [0:LANES-1], a1_b [0:LANES-1], a1_y [0:LANES-1];
  logic [31:0] a2_a [0:LANES-1], a2_b [0:LANES-1], a2_y [0:LANES-1];
  logic [31:0] acc_q   [0:LANES-1];   // D*x seed
  logic [31:0] out_buf [0:LANES-1];   // drained y, stable under backpressure
  /* verilator lint_off UNUSEDSIGNAL */
  logic        m1_ov [0:LANES-1], m2_ov [0:LANES-1], m3_ov [0:LANES-1];
  logic        a1_ov [0:LANES-1], a2_ov [0:LANES-1];
  /* verilator lint_on UNUSEDSIGNAL */

  logic mac_m1, mac_m2, mac_m3, mac_m3_inc, mac_a1, mac_a2, mac_hwr, mac_dmul;

  /* verilator lint_off PINCONNECTEMPTY */
  genvar gl;
  generate
    for (gl = 0; gl < LANES; gl++) begin : g_lane
      fp_unit #(.W_EXP(8), .W_MANT(23)) u_m1 (
        .clk(clk), .rst_n(rst_n),
        .mode(fp_pkg::OP_MUL), .rm(fp_pkg::RM_RNE),
        .a(m1_a[gl]), .b(m1_b[gl]), .c('0),
        .y(m1_y[gl]), .cmp(/*unused*/), .flags(/*unused*/),
        .in_valid(mac_m1), .out_valid(m1_ov[gl])
      );
      fp_unit #(.W_EXP(8), .W_MANT(23)) u_m2 (
        .clk(clk), .rst_n(rst_n),
        .mode(fp_pkg::OP_MUL), .rm(fp_pkg::RM_RNE),
        .a(m2_a[gl]), .b(m2_b[gl]), .c('0),
        .y(m2_y[gl]), .cmp(/*unused*/), .flags(/*unused*/),
        .in_valid(mac_m2), .out_valid(m2_ov[gl])
      );
      fp_unit #(.W_EXP(8), .W_MANT(23)) u_m3 (
        .clk(clk), .rst_n(rst_n),
        .mode(fp_pkg::OP_MUL), .rm(fp_pkg::RM_RNE),
        .a(m3_a[gl]), .b(m3_b[gl]), .c('0),
        .y(m3_y[gl]), .cmp(/*unused*/), .flags(/*unused*/),
        .in_valid(mac_m3), .out_valid(m3_ov[gl])
      );
      fp_unit #(.W_EXP(8), .W_MANT(23)) u_a1 (
        .clk(clk), .rst_n(rst_n),
        .mode(fp_pkg::OP_ADD), .rm(fp_pkg::RM_RNE),
        .a(a1_a[gl]), .b(a1_b[gl]), .c('0),
        .y(a1_y[gl]), .cmp(/*unused*/), .flags(/*unused*/),
        .in_valid(mac_a1), .out_valid(a1_ov[gl])
      );
      fp_unit #(.W_EXP(8), .W_MANT(23)) u_a2 (
        .clk(clk), .rst_n(rst_n),
        .mode(fp_pkg::OP_ADD), .rm(fp_pkg::RM_RNE),
        .a(a2_a[gl]), .b(a2_b[gl]), .c(32'h0),
        .y(a2_y[gl]), .cmp(/*unused*/), .flags(/*unused*/),
        .in_valid(mac_a2), .out_valid(a2_ov[gl])
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
    H_Z_W,     // wait for the z add
    H_ZDEC,    // start softplus on bf16(z)
    H_SP,      // wait softplus, store dtp (bf16)
    H_AEXP,    // start exp(A_log)
    H_AWAIT,   // wait, latch A = -exp(A_log)
    H_AMUL,    // issue MUL(A, dtp)
    H_AMUL_W,  // wait
    H_DEXP,    // start exp(A*dtp)
    H_DWAIT,   // wait, store dA
    W_ISSUE,   // stream w_s = dtp * B_s
    MAC,       // lane element schedule for one LANES-wide output block
    OUT        // stream the block's y beats (handshake)
  } state_t;

  state_t state;
  logic [CNT_W-1:0] in_cnt;
  logic [H_W-1:0]   h_cnt;
  logic [W_CNT_W-1:0] w_cnt;   // w stream: next element to issue
  logic [W_CNT_W-1:0] w_out;   // w stream: next result to store
  logic [3:0]       wv;        // w stream valid pipeline
  logic [D_BW-1:0]  d_base;    // first dim of the current LANES block
  logic [L_W-1:0]   out_cnt;   // output beat counter within the block

  // MAC schedule counters
  logic [T_W-1:0]   t_cnt;
  logic [1:0]       ph;        // t_cnt mod 3
  logic [W_CNT_W-1:0] s_m1, s_a1, s_m3, s_a2;

  logic [31:0] dtp_reg, a_reg;

  // LANES block geometry.
  logic [L_W:0] lanes_this;
  always_comb begin
    if ((HEAD_DIM - int'(d_base)) >= LANES)
      lanes_this = (L_W + 1)'(LANES);
    else
      lanes_this = (L_W + 1)'(HEAD_DIM - int'(d_base));
  end

  assign s_axis_tready = (state == IDLE);
  assign busy          = (state != IDLE);

  assign sp_start = (state == H_ZDEC);

  assign m_axis_tvalid = (state == OUT);
  assign m_axis_tdata  = out_buf[out_cnt];
  assign m_axis_tlast  = (state == OUT) &&
                         (h_cnt == H_W'(NUM_HEADS - 1)) &&
                         ((d_base + D_BW'(out_cnt)) == D_BW'(HEAD_DIM - 1));

  // ------------------------------------------------------------ MAC schedule
  // ph = t_cnt mod 3; the element index of each issue is tracked by the
  // s_* counters (see the header comment for the timeline).
  always_comb begin
    mac_dmul = (state == MAC) && (t_cnt == T_W'(0));
    mac_m1   = (state == MAC) && (ph == 2'd0) && (s_m1 < W_CNT_W'(D_STATE));
    mac_m2   = mac_m1;
    mac_a1   = (state == MAC) && (ph == 2'd1) && (s_a1 < W_CNT_W'(D_STATE)) &&
               (t_cnt >= T_W'(4));
    mac_m3_inc = (state == MAC) && (ph == 2'd1) &&
                 (s_m3 < W_CNT_W'(D_STATE)) && (t_cnt >= T_W'(7));
    mac_m3   = mac_dmul || mac_m3_inc;
    mac_a2   = (state == MAC) && (ph == 2'd2) && (s_a2 < W_CNT_W'(D_STATE)) &&
               (t_cnt >= T_W'(11));
    mac_hwr  = mac_m3_inc;
  end

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
    prep_mul_valid = 1'b0;
    prep_add_valid = 1'b0;

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
                 + CLR_W'(s_m1);
      hidx_w[l]  = CLR_W'(h_cnt) * CLR_W'(HEAD_DIM * D_STATE)
                 + CLR_W'(d_rd[l]) * CLR_W'(D_STATE)
                 + CLR_W'(s_m3);
      x_lane[l]  = {x_buf[h_cnt * HEAD_DIM + d_rd[l]],
                    {(32 - W_DATA) {1'b0}}};
    end

    case (state)
      H_Z: begin
        prep_add_valid = 1'b1;
        add_a = {dt_buf[h_cnt], {(32 - W_DATA) {1'b0}}};
        add_b = {dtb_q[h_cnt],  {(32 - W_DATA) {1'b0}}};
      end

      H_AEXP: exp_x = {a_log_q[h_cnt], {(32 - W_DATA) {1'b0}}};

      H_AMUL: begin
        prep_mul_valid = 1'b1;
        mul_a = a_reg;
        mul_b = dtp_reg;
      end

      H_DEXP: exp_x = mul_y;   // A * dtp (issued in H_AMUL)

      W_ISSUE: begin
        if (w_cnt < W_CNT_W'(D_STATE)) begin
          prep_mul_valid = 1'b1;
          mul_a = dtp_reg;
          mul_b = {b_buf[w_cnt], {(32 - W_DATA) {1'b0}}};
        end
      end

      MAC: begin
        for (int l = 0; l < LANES; l++) begin
          // m1 = dA*h[s], m2 = w[s]*x
          if (mac_m1) begin
            m1_a[l] = dA_q[h_cnt];
            m1_b[l] = h_state[hidx_r[l]];
            m2_a[l] = w_buf[s_m1];
            m2_b[l] = x_lane[l];
          end
          // a1 = m1 + m2 (new h)
          if (mac_a1) begin
            a1_a[l] = m1_y[l];
            a1_b[l] = m2_y[l];
          end
          // m3 = D*x (t = 0) or a1 * C[s]
          if (mac_dmul) begin
            m3_a[l] = {d_q[h_cnt], {(32 - W_DATA) {1'b0}}};
            m3_b[l] = x_lane[l];
          end else if (mac_m3_inc) begin
            m3_a[l] = a1_y[l];
            m3_b[l] = {c_buf[s_m3], {(32 - W_DATA) {1'b0}}};
          end
          // a2 = acc + m3 (first element uses the registered D*x seed)
          if (mac_a2) begin
            a2_a[l] = (s_a2 == W_CNT_W'(0)) ? acc_q[l] : a2_y[l];
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
      w_cnt   <= '0;
      w_out   <= '0;
      wv      <= 4'b0;
      d_base  <= '0;
      out_cnt <= '0;
      t_cnt   <= '0;
      ph      <= 2'd0;
      s_m1    <= '0;
      s_a1    <= '0;
      s_m3    <= '0;
      s_a2    <= '0;
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
              state  <= H_Z;
            end else begin
              in_cnt <= in_cnt + 1'b1;
            end
          end
        end

        // ---------------------------------------------------- per-head prep
        H_Z:   state <= H_Z_W;   // add started this cycle
        H_Z_W: begin
          if (prep_add_ov) state <= H_ZDEC;
        end
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
        H_AMUL:  state <= H_AMUL_W;             // MUL(A, dtp) issued combinationally
        H_AMUL_W: begin
          if (prep_mul_ov) state <= H_DEXP;     // mul_y holds A*dtp
        end
        H_DEXP:  state <= H_DWAIT;              // exp(A*dtp) started combinationally
        H_DWAIT: begin
          if (exp_done) begin
            dA_q[h_cnt] <= exp_y;               // dA = exp(A*dtp)
            w_cnt       <= '0;
            w_out       <= '0;
            wv          <= 4'b0;
            state       <= W_ISSUE;
          end
        end

        // ---------------------------------------------------- w = dtp * B
        // Streaming: one MUL start per cycle, results latched 4 cycles later.
        W_ISSUE: begin
          wv[0] <= (w_cnt < W_CNT_W'(D_STATE));
          wv[1] <= wv[0];
          wv[2] <= wv[1];
          wv[3] <= wv[2];
          if (w_cnt < W_CNT_W'(D_STATE))
            w_cnt <= w_cnt + 1'b1;
          if (wv[3]) begin
            w_buf[w_out] <= mul_y;
            if (w_out == W_CNT_W'(D_STATE - 1)) begin
              w_out   <= '0;
              t_cnt   <= '0;
              ph      <= 2'd0;
              s_m1    <= '0;
              s_a1    <= '0;
              s_m3    <= '0;
              s_a2    <= '0;
              d_base  <= '0;
              out_cnt <= '0;
              state   <= MAC;
            end else begin
              w_out <= w_out + 1'b1;
            end
          end
        end

        // ---------------------------------------------------- element schedule
        MAC: begin
          t_cnt <= t_cnt + 1'b1;
          ph    <= (ph == 2'd2) ? 2'd0 : (ph + 2'd1);
          if (mac_m1)   s_m1 <= s_m1 + 1'b1;
          if (mac_a1)   s_a1 <= s_a1 + 1'b1;
          if (mac_m3_inc) s_m3 <= s_m3 + 1'b1;
          if (mac_a2)   s_a2 <= s_a2 + 1'b1;

          if (t_cnt == T_W'(4))
            for (int l = 0; l < LANES; l++)
              acc_q[l] <= m3_y[l];            // registered D*x

          if (mac_hwr)
            for (int l = 0; l < LANES; l++)
              if (lane_ok[l])
                h_state[hidx_w[l]] <= a1_y[l]; // h[s] = new h

          if (t_cnt == T_W'(3 * D_STATE + 11)) begin
            // Last a2 result is on the units' outputs; drain to out_buf.
            for (int l = 0; l < LANES; l++) begin
              out_buf[l] <= a2_y[l];
            end
            t_cnt   <= '0;
            out_cnt <= '0;
            state   <= OUT;
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
                // Next dim block: restart the MAC schedule counters.
                d_base <= d_base + D_BW'(LANES);
                t_cnt  <= '0;
                ph     <= 2'd0;
                s_m1   <= '0;
                s_a1   <= '0;
                s_m3   <= '0;
                s_a2   <= '0;
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
