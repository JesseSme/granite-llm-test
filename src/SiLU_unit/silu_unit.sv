// SiLU activation function: SiLU(x) = x * sigmoid(x)
//
// Pipeline:
//   Cycle 0: input arrives, sigmoid_unit starts computing sigmoid(x)
//   Cycle 1: sigmoid(x) ready, x is delayed 1 cycle, multiply begins (combinational)
//   Cycle 2: result registered (2-cycle total latency)
//
// bfloat16 I/O. Sigmoid unit is instantiated as a sub-component.

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */

module silu_unit (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        valid_in,
  input  logic [15:0] data_in,
  output logic [15:0] data_out,
  output logic        valid_out
);

  localparam int W      = 16;
  localparam int W_EXP  = 8;
  localparam int W_MANT = 7;
  localparam int EXP_ALL = (1 << W_EXP) - 1;
  localparam int BIAS = (1 << (W_EXP-1)) - 1;
  localparam int EMAX = BIAS;
  localparam int EMIN = 1 - BIAS;

  // Signed comparison bounds to avoid unsigned comparison traps
  localparam logic signed [W_EXP+4:0] EMAX_S = (W_EXP+5)'(EMAX);
  localparam logic signed [W_EXP+4:0] EMIN_S = (W_EXP+5)'(EMIN);

  // ------------------------------------------------------- sigmoid instance
  logic [W-1:0] sigmoid_result;
  logic         sigmoid_valid;

  sigmoid_unit u_sigmoid (
    .clk      (clk),
    .rst_n    (rst_n),
    .valid_in (valid_in),
    .data_in  (data_in),
    .data_out (sigmoid_result),
    .valid_out(sigmoid_valid)
  );

  // ------------------------------------- delay input by 1 cycle (align with sigmoid)
  logic [W-1:0] x_delayed;
  logic         x_delayed_valid;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      x_delayed       <= '0;
      x_delayed_valid <= 1'b0;
    end else begin
      x_delayed       <= data_in;
      x_delayed_valid <= valid_in;
    end
  end

  // ------------------------------------------- bfloat16 multiply (combinational)
  // SiLU(x) = x * sigmoid(x)
  // Simplified bfloat16 multiply: handles NaN, Inf, Zero, normals, subnormals.
  // bfloat16 format: sign[15] exp[14:7] mant[6:0], bias=127

  // Unpack x_delayed
  logic        x_sign;
  logic [W_EXP-1:0]  x_exp;
  logic [W_MANT-1:0] x_mant;
  assign x_sign = x_delayed[W-1];
  assign x_exp  = x_delayed[W-2 -: W_EXP];
  assign x_mant = x_delayed[W_MANT-1:0];

  // Unpack sigmoid_result
  logic        s_sign;
  logic [W_EXP-1:0]  s_exp;
  logic [W_MANT-1:0] s_mant;
  assign s_sign = sigmoid_result[W-1];
  assign s_exp  = sigmoid_result[W-2 -: W_EXP];
  assign s_mant = sigmoid_result[W_MANT-1:0];

  // Classify x
  logic x_nan, x_inf, x_zero, x_sub;
  assign x_nan  = (x_exp == EXP_ALL[W_EXP-1:0]) && (x_mant != '0);
  assign x_inf  = (x_exp == EXP_ALL[W_EXP-1:0]) && (x_mant == '0);
  assign x_zero = (x_exp == '0) && (x_mant == '0);
  assign x_sub  = (x_exp == '0) && (x_mant != '0);

  // Classify sigmoid
  logic s_nan, s_inf, s_zero, s_sub;
  assign s_nan  = (s_exp == EXP_ALL[W_EXP-1:0]) && (s_mant != '0);
  assign s_inf  = (s_exp == EXP_ALL[W_EXP-1:0]) && (s_mant == '0);
  assign s_zero = (s_exp == '0) && (s_mant == '0);
  assign s_sub  = (s_exp == '0) && (s_mant != '0);

  // Sign of product = XOR of operand signs
  logic rsign;
  assign rsign = x_sign ^ s_sign;

  // Build significands with implicit bit (8 bits for bfloat16)
  logic [W_MANT:0] s_x, s_s;
  assign s_x = x_sub ? {1'b0, x_mant} : {1'b1, x_mant};
  assign s_s = s_sub ? {1'b0, s_mant} : {1'b1, s_mant};

  // Exact product (16 bits)
  logic [2*(W_MANT+1)-1:0] P;
  assign P = s_x * s_s;

  // Product MSB position (bit length of P)
  function automatic int unsigned pbitlen(input logic [2*(W_MANT+1)-1:0] x);
    pbitlen = 0;
    for (int i = 0; i < 2*(W_MANT+1); i++)
      if (x[i]) pbitlen = i + 1;
  endfunction

  // Leading-bit exponents
  logic signed [W_EXP+2:0] e_x, e_s;
  always_comb begin
    e_x = (W_EXP+3)'(x_exp - 8'd127);
    if (x_sub) e_x = (W_EXP+3)'(-126 - W_MANT);
    e_s = (W_EXP+3)'(s_exp - 8'd127);
    if (s_sub) e_s = (W_EXP+3)'(-126 - W_MANT);
  end

  // Result exponent
  logic signed [W_EXP+3:0] e_res;
  assign e_res = (W_EXP+4)'(e_x) + (W_EXP+4)'(e_s)
               + (W_EXP+4)'(pbitlen(P) == (W_MANT+1) + (W_MANT+1));

  // Place product into fixed-point: target MSB at bit (W_MANT+1+2)
  localparam int S_W = W_MANT + 1;
  localparam int TARGET = S_W + 2;
  localparam int FW = S_W + 3;
  localparam int P_W = 2 * S_W;

  logic [FW-1:0] placed;
  logic          norm_sticky;

  always_comb begin
    int B;
    B = int'(pbitlen(P));
    norm_sticky = 1'b0;
    if (B == 0) begin
      placed = '0;
      norm_sticky = 1'b0;
    end else if (B <= FW) begin
      placed = FW'(P) << (FW - B);
      norm_sticky = 1'b0;
    end else begin
      placed = FW'(P >> (B - FW));
      norm_sticky = |(P & (((P_W)'(1) << (B - FW)) - 1));
    end
  end

  // Extract significand and guard/round/sticky
  logic [S_W-1:0] sig;
  logic           g, r, s_bit;
  assign sig = placed[TARGET -: S_W];
  assign g   = placed[2];
  assign r   = placed[1];
  assign s_bit = placed[0] | norm_sticky;

  // RNE rounding
  logic           round_up, carry;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [S_W-1:0] sig_rnd;
  /* verilator lint_on UNUSEDSIGNAL */
  logic signed [W_EXP+3:0] e_rnd;

  assign round_up = g & (r | s_bit) | (g & ~r & ~s_bit & sig[0]);
  assign carry = round_up && (sig == {S_W{1'b1}});

  always_comb begin
    if (carry) begin
      sig_rnd = {1'b1, {W_MANT{1'b0}}};
      e_rnd   = e_res + 1;
    end else begin
      sig_rnd = sig + { {S_W-1{1'b0}}, round_up };
      e_rnd   = e_res;
    end
  end

  // Re-encode result
  logic [W_EXP-1:0] exp_fld;
  logic [W-1:0]     mul_normal;

  assign exp_fld = e_rnd[W_EXP-1:0] + 8'd127;

  /* verilator lint_off LATCH */
  always_comb begin
    if ($signed(e_rnd) > EMAX_S) begin
      // Overflow -> infinity
      mul_normal = {rsign, EXP_ALL[W_EXP-1:0], {W_MANT{1'b0}}};
    end else if ($signed(e_rnd) < EMIN_S) begin
      // Subnormal result: shift significand right
      int shift;
      logic [FW-1:0] shifted;
      logic [S_W-1:0] sig_s;
      logic gs, rs2, ss;
      shift = -126 - int'(e_rnd);
      shifted = '0;
      sig_s = '0;
      gs = 1'b0;
      rs2 = 1'b0;
      ss = 1'b0;
      if (shift > 0) shifted = placed >> shift;
      else           shifted = placed;
      sig_s = shifted[TARGET -: S_W];
      gs    = shifted[2];
      rs2   = shifted[1];
      ss    = shifted[0] | norm_sticky
             | |(placed & (((FW)'(1) << (shift > 0 ? shift : 0)) - 1));
      if (gs & (rs2 | ss) | (gs & ~rs2 & ~ss & sig_s[0]))
        sig_s = sig_s + 1;
      if (sig_s == {1'b1, {W_MANT{1'b0}}})
        mul_normal = {rsign, 8'd1, {W_MANT{1'b0}}};
      else
        mul_normal = {rsign, {W_EXP{1'b0}}, sig_s[W_MANT-1:0]};
    end else begin
      mul_normal = {rsign, exp_fld, sig_rnd[W_MANT-1:0]};
    end
  end
  /* verilator lint_on LATCH */

  // Special case handling for multiply
  logic [W-1:0] y_special;
  logic         special_case;

  localparam logic [W_MANT-1:0] QUIET = (1 << W_MANT) - 1; // all ones = quiet NaN

  always_comb begin
    y_special    = '0;
    special_case = 1'b0;
    if (x_nan || s_nan) begin
      y_special = {1'b0, EXP_ALL[W_EXP-1:0], x_nan ? (x_mant | QUIET) : (s_mant | QUIET)};
      special_case = 1'b1;
    end else if (x_inf || s_inf) begin
      if ((x_inf && s_zero) || (s_inf && x_zero)) begin
        y_special = {1'b0, EXP_ALL[W_EXP-1:0], QUIET};
        special_case = 1'b1;
      end else begin
        y_special = {rsign, EXP_ALL[W_EXP-1:0], {W_MANT{1'b0}}};
        special_case = 1'b1;
      end
    end else if (x_zero || s_zero) begin
      y_special = {rsign, {W_EXP{1'b0}}, {W_MANT{1'b0}}};
      special_case = 1'b1;
    end
  end

  // Final multiply result
  logic [W-1:0] mul_result_comb;
  assign mul_result_comb = special_case ? y_special : mul_normal;

  // --------------------------------------------------- output pipeline register
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      data_out  <= '0;
      valid_out <= 1'b0;
    end else begin
      data_out  <= mul_result_comb;
      valid_out <= x_delayed_valid & sigmoid_valid;
    end
  end

endmodule
