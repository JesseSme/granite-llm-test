// Sequential accurate fp32 exp: y = exp(x), x clamped to [-16, 16].
//
// Used by the SSM unit for A = -exp(A_log) and the per-step decay
// dA = exp(A * dtp), by the attention softmax and by the SwiGLU silu. The
// shared `fp_exp` LUT units are only ~8-bit accurate (max relative error ~20%
// measured), which is not sufficient inside a recurrent state update, so this
// unit uses range reduction plus a degree-6 polynomial and scaling by
// squaring:
//
//   exp(x) = exp(x/32)^32,   exp(r) ~= 1 + r*(1 + r*(1/2 + r*(1/6 +
//                                        r*(1/24 + r*(1/120 + r/720)))))
// with |x/32| <= 0.5, giving ~1e-6 relative error.
//
// fp_unit protocol: the datapath holds one fp_unit and issues one transaction
// at a time (1-cycle in_valid pulse, result latched on out_valid). The
// op sequence and operand order are unchanged from the previous 1-cycle
// schedule, so the result is bit-identical; only the latency is longer
// (18 operations of 3-4 cycles each).
//
// The caller pulses `start` with `x`; `done` pulses one cycle with `y` valid
// (held until the next start).

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */

module fp_exp_seq (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  input  logic [31:0] x,
  output logic [31:0] y,
  output logic        done
);

  localparam logic [31:0] C_SCALE = 32'h3D000000;  // 2^-5
  localparam logic [31:0] C_6     = 32'h3AB60B61;  // 1/720
  localparam logic [31:0] C_5     = 32'h3C088889;  // 1/120
  localparam logic [31:0] C_4     = 32'h3D2AAAAB;  // 1/24
  localparam logic [31:0] C_3     = 32'h3E2AAAAB;  // 1/6
  localparam logic [31:0] C_2     = 32'h3F000000;  // 1/2
  localparam logic [31:0] C_1     = 32'h3F800000;  // 1.0
  localparam logic [31:0] C_MAX   = 32'h41800000;  // 16.0

  logic [31:0] x_reg, r_reg, h_reg, prev_y, y_reg;
  logic [4:0]  op;       // 0..17, see the operand table below
  logic        running;  // a start has been accepted
  logic        pend;     // the current op is in flight

  // Clamp |x| to 16.0, preserving the sign.
  wire [31:0] clamped_mag = (x[30:0] > C_MAX[30:0]) ? {x[31], C_MAX[30:0]} : x;

  // ------------------------------------------------------- FPU interface
  fp_pkg::op_t       mode;
  fp_pkg::rounding_t rm;
  logic [31:0]       a, b, fpu_y;
  logic              in_valid, out_valid;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0]        cmp;
  logic [4:0]        flags;
  /* verilator lint_on UNUSEDSIGNAL */

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_fp (
    .clk(clk), .rst_n(rst_n),
    .mode(mode), .rm(rm),
    .a(a), .b(b), .c('0),
    .y(fpu_y), .cmp(cmp), .flags(flags),
    .in_valid(in_valid), .out_valid(out_valid)
  );

  // ------------------------------------------------------- op inputs
  // The sequence (same order as the previous fixed-schedule version):
  //   0  r = x/32                      on done: r_reg = y, h_reg = 1/720
  //   1  h*r                          2  + 1/120 (h_reg = y)
  //   3  h*r                          4  + 1/24  (h_reg = y)
  //   5  h*r                          6  + 1/6   (h_reg = y)
  //   7  h*r                          8  + 1/2   (h_reg = y)
  //   9  h*r                         10  + 1     (h_reg = y)
  //  11  r*h                         12  1 + r*h (p)
  //  13..17  p = p*p (5 squarings; op 17 completes with y = p^32)
  always_comb begin
    mode      = fp_pkg::OP_ADD;
    a         = '0;
    b         = '0;
    in_valid  = running && !pend;
    rm        = fp_pkg::RM_RNE;

    case (op)
      5'd0:  begin mode = fp_pkg::OP_MUL; a = x_reg;  b = C_SCALE; end
      5'd1,
      5'd3,
      5'd5,
      5'd7,
      5'd9:  begin mode = fp_pkg::OP_MUL; a = h_reg;  b = r_reg;   end
      5'd2:  begin mode = fp_pkg::OP_ADD; a = prev_y; b = C_5;     end
      5'd4:  begin mode = fp_pkg::OP_ADD; a = prev_y; b = C_4;     end
      5'd6:  begin mode = fp_pkg::OP_ADD; a = prev_y; b = C_3;     end
      5'd8:  begin mode = fp_pkg::OP_ADD; a = prev_y; b = C_2;     end
      5'd10: begin mode = fp_pkg::OP_ADD; a = prev_y; b = C_1;     end
      5'd11: begin mode = fp_pkg::OP_MUL; a = r_reg;  b = h_reg;   end
      5'd12: begin mode = fp_pkg::OP_ADD; a = C_1;    b = prev_y;  end
      5'd13,
      5'd14,
      5'd15,
      5'd16,
      5'd17: begin mode = fp_pkg::OP_MUL; a = prev_y; b = prev_y;  end
      default: ;
    endcase
  end

  // ------------------------------------------------------- FSM
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      op      <= 5'd0;
      running <= 1'b0;
      pend    <= 1'b0;
      x_reg   <= '0;
      r_reg   <= '0;
      h_reg   <= '0;
      prev_y  <= '0;
      y_reg   <= '0;
      done    <= 1'b0;
    end else begin
      done <= 1'b0;
      if (!running) begin
        if (start) begin
          x_reg   <= clamped_mag;
          op      <= 5'd0;
          pend    <= 1'b0;
          running <= 1'b1;
        end
      end else if (!pend) begin
        pend <= 1'b1;   // in_valid is high this cycle; wait for out_valid
      end else if (out_valid) begin
        pend   <= 1'b0;
        prev_y <= fpu_y;
        case (op)
          5'd0:       begin r_reg <= fpu_y; h_reg <= C_6; end
          5'd2,
          5'd4,
          5'd6,
          5'd8,
          5'd10:      h_reg <= fpu_y;
          5'd17: begin
            y_reg   <= fpu_y;
            done    <= 1'b1;
            running <= 1'b0;
          end
          default: ;
        endcase
        if (op != 5'd17)
          op <= op + 5'd1;
      end
    end
  end

  assign y = y_reg;

endmodule
