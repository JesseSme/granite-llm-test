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
  parameter int H_W       = $clog2(NUM_HEADS),
  parameter int D_W       = $clog2(HEAD_DIM),
  parameter int S_W       = $clog2(D_STATE)
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

  // exp (for A_h and dA_h") — accurate sequential exp (the shared LUT
  // fp_exp is only ~8-bit accurate, insufficient for the recurrence)
  logic [31:0] exp_x, exp_y;
  logic        exp_start, exp_done;

  fp_exp_seq u_exp (
    .clk(clk), .rst_n(rst_n),
    .start(exp_start), .x(exp_x), .y(exp_y), .done(exp_done)
  );

  assign exp_start = (state == H_AEXP) || (state == H_DEXP);

  // MUL
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

  // ADD
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
    D_INIT,    // issue acc = D_h * x_hd
    D_ACC,     // latch acc
    U0, U1, U2, U3, U4, U5,   // per-element update + y accumulation
    OUT        // stream y beat (handshake)
  } state_t;

  state_t state;
  logic [CNT_W-1:0] in_cnt;
  logic [H_W-1:0]   h_cnt;
  logic [D_W-1:0]   d_cnt;
  logic [S_W-1:0]   s_cnt;

  logic [31:0] dtp_reg, a_reg, xd_reg;
  logic [31:0] acc_reg, p1_reg;

  assign s_axis_tready = (state == IDLE);
  assign busy          = (state != IDLE);

  // Flat index into the recurrent state.
  wire [CLR_W-1:0] hidx =
      CLR_W'(h_cnt) * (HEAD_DIM * D_STATE) + CLR_W'(d_cnt) * D_STATE + CLR_W'(s_cnt);

  // Start softplus in the same cycle the z add result is on the adder output,
  // so its step-0 edge samples the correct z.
  assign sp_start = (state == H_ZDEC);

  assign m_axis_tvalid = (state == OUT);
  assign m_axis_tdata  = acc_reg;
  assign m_axis_tlast  = (state == OUT) &&
                         (h_cnt == H_W'(NUM_HEADS - 1)) &&
                         (d_cnt == D_W'(HEAD_DIM - 1));

  // ------------------------------------------------------------ op inputs
  always_comb begin
    sp_z  = {z_bf16, {(32 - W_DATA) {1'b0}}};
    exp_x = '0;
    mul_a = '0;
    mul_b = '0;
    add_a = '0;
    add_b = '0;

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

      D_INIT: begin
        mul_a = {d_q[h_cnt], {(32 - W_DATA) {1'b0}}};
        mul_b = {x_buf[h_cnt * HEAD_DIM + d_cnt], {(32 - W_DATA) {1'b0}}};
      end

      U0: begin
        mul_a = dA_q[h_cnt];
        mul_b = h_state[hidx];
      end

      U1: begin
        mul_a = w_buf[s_cnt];
        mul_b = xd_reg;
      end

      U2: begin
        add_a = p1_reg;
        add_b = mul_y;
      end

      U3: begin
        mul_a = add_y;
        mul_b = {c_buf[s_cnt], {(32 - W_DATA) {1'b0}}};
      end

      U4: begin
        add_a = acc_reg;
        add_b = mul_y;
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
      d_cnt   <= '0;
      s_cnt   <= '0;
      dtp_reg <= '0;
      a_reg   <= '0;
      xd_reg  <= '0;
      acc_reg <= '0;
      p1_reg  <= '0;
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
              d_cnt  <= '0;
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
            s_cnt <= '0;
            d_cnt <= '0;
            state <= D_INIT;
          end else begin
            s_cnt <= s_cnt + 1'b1;
            state <= W_ISSUE;
          end
        end

        // ---------------------------------------------------- d loop
        D_INIT: begin
          xd_reg <= {x_buf[h_cnt * HEAD_DIM + d_cnt], {(32 - W_DATA) {1'b0}}};
          state  <= D_ACC;
        end
        D_ACC: begin
          acc_reg <= mul_y;   // D_h * x_hd
          s_cnt   <= '0;
          state   <= U0;
        end

        // ---------------------------------------------------- element update
        U0: state <= U1;                       // issue dA*h
        U1: begin
          p1_reg <= mul_y;                     // dA*h
          state  <= U2;                        // issue w*x
        end
        U2: state <= U3;                       // issue newh = p1 + p2
        U3: begin
          h_state[hidx] <= add_y;    // newh
          state <= U4;               // issue newh*C_s
        end
        U4: state <= U5;                       // issue acc + newh*C_s
        U5: begin
          acc_reg <= add_y;
          if (s_cnt == S_W'(D_STATE - 1)) begin
            state <= OUT;          // this (h,d) output is ready
          end else begin
            s_cnt <= s_cnt + 1'b1;
            state <= U0;
          end
        end

        // ---------------------------------------------------- output
        OUT: begin
          if (m_axis_tready) begin
            if (h_cnt == H_W'(NUM_HEADS - 1) && d_cnt == D_W'(HEAD_DIM - 1)) begin
              state <= IDLE;
            end else if (d_cnt == D_W'(HEAD_DIM - 1)) begin
              h_cnt <= h_cnt + 1'b1;
              d_cnt <= '0;
              state <= H_Z;
            end else begin
              d_cnt <= d_cnt + 1'b1;
              state <= D_INIT;
            end
          end
        end

        default: state <= IDLE;
      endcase
    end
  end

endmodule
