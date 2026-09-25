// Sequential row softmax with an accurate exponential.
//
// Streams in one row of up to N binary32 logits (valid_in/data_in, last_in on
// the final element) and computes
//
//   p_j = exp(s_j - max_k s_k) / sum_k exp(s_k - max_k s_k)     (binary32)
//
// rounding each probability once to bfloat16 (matching torch's
// `softmax(..., dtype=torch.float32).to(query.dtype)`). Probabilities are held
// in an internal bf16 buffer that can be read back at any index
// (rd_idx -> rd_data, widened to binary32) once `done` has pulsed.
//
// The exponential uses fp_exp_seq (accurate polynomial exp), because the
// LUT-based fp_exp inside src/softmax_unit/ has ~20% relative error and would
// dominate the attention error budget. Structure follows softmax_unit.sv.
//
// `start`-style streaming: the row is complete when a beat with last_in is
// accepted; a single-element row completes on the first beat.

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off SYNCASYNCNET */

module attn_softmax_seq #(
  parameter int N     = 64,
  parameter int IDX_W = $clog2(N)
) (
  input  logic            clk,
  input  logic            rst_n,
  input  logic            valid_in,
  input  logic [31:0]     data_in,
  input  logic            last_in,
  output logic            done,
  input  logic [IDX_W-1:0] rd_idx,
  output logic [31:0]     rd_data
);

  localparam int CNT_W = $clog2(N + 1);

  logic [31:0]      s_buf [0:N-1];
  logic [31:0]      e_buf [0:N-1];
  logic [15:0]      p_buf [0:N-1];
  logic [CNT_W-1:0] row_len;

  assign rd_data = {p_buf[rd_idx], 16'b0};

  // ------------------------------------------------------- units
  /* verilator lint_off PINCONNECTEMPTY */
  /* verilator lint_off UNUSEDSIGNAL */
  logic unused_out_valid_max, unused_out_valid_sub;
  logic unused_out_valid_add, unused_out_valid_div;
  /* verilator lint_on UNUSEDSIGNAL */
  fp_unit #(.W_EXP(8), .W_MANT(23)) u_max (
    .clk(clk), .rst_n(rst_n),
    .mode(fp_pkg::OP_MAX), .rm(fp_pkg::RM_RNE),
    .a(max_a), .b(max_b), .c('0),
    .y(max_y), .cmp(/*unused*/), .flags(/*unused*/),
    .in_valid(1'b1), .out_valid(unused_out_valid_max)
  );

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_sub (
    .clk(clk), .rst_n(rst_n),
    .mode(fp_pkg::OP_SUB), .rm(fp_pkg::RM_RNE),
    .a(sub_a), .b(sub_b), .c('0),
    .y(sub_y), .cmp(/*unused*/), .flags(/*unused*/),
    .in_valid(1'b1), .out_valid(unused_out_valid_sub)
  );

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_add (
    .clk(clk), .rst_n(rst_n),
    .mode(fp_pkg::OP_ADD), .rm(fp_pkg::RM_RNE),
    .a(add_a), .b(add_b), .c('0),
    .y(add_y), .cmp(/*unused*/), .flags(/*unused*/),
    .in_valid(1'b1), .out_valid(unused_out_valid_add)
  );

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_div (
    .clk(clk), .rst_n(rst_n),
    .mode(fp_pkg::OP_DIV), .rm(fp_pkg::RM_RNE),
    .a(div_a), .b(div_b), .c('0),
    .y(div_y), .cmp(/*unused*/), .flags(/*unused*/),
    .in_valid(1'b1), .out_valid(unused_out_valid_div)
  );
  /* verilator lint_on PINCONNECTEMPTY */

  logic [31:0] exp_x, exp_y;
  logic        exp_start, exp_done;

  fp_exp_seq u_exp (
    .clk(clk), .rst_n(rst_n),
    .start(exp_start), .x(exp_x), .y(exp_y), .done(exp_done)
  );

  logic [15:0] p_bf;
  fp32_to_bf16_round u_round_p (.x(div_y), .y(p_bf));

  logic [31:0] max_a, max_b, max_y;
  logic [31:0] sub_a, sub_b, sub_y;
  logic [31:0] add_a, add_b, add_y;
  logic [31:0] div_a, div_b, div_y;

  // ------------------------------------------------------- FSM
  localparam logic [4:0] S_IDLE      = 5'd0,
                         S_LOAD      = 5'd1,
                         S_MAX_INIT  = 5'd2,
                         S_MAX_SETUP = 5'd3,
                         S_MAX_WAIT  = 5'd4,
                         S_MAX_READ  = 5'd5,
                         S_SUB_SETUP = 5'd6,
                         S_SUB_WAIT  = 5'd7,
                         S_EXP_START = 5'd8,
                         S_EXP_WAIT  = 5'd9,
                         S_STORE     = 5'd10,
                         S_SUM_INIT  = 5'd11,
                         S_SUM_SETUP = 5'd12,
                         S_SUM_WAIT  = 5'd13,
                         S_SUM_READ  = 5'd14,
                         S_NORM_DIV  = 5'd15,
                         S_NORM_WAIT = 5'd16,
                         S_NORM_OUT  = 5'd17,
                         S_DONE      = 5'd18;

  logic [4:0]       state;
  logic [CNT_W-1:0] cnt;
  logic [31:0]      running_max;
  logic [31:0]      sum;

  assign exp_start = (state == S_EXP_START);

  always_comb begin
    max_a = '0; max_b = '0;
    sub_a = '0; sub_b = '0;
    add_a = '0; add_b = '0;
    div_a = '0; div_b = '0;
    exp_x = '0;

    // Operands are held through the wait state so the (1-cycle latency) fp_unit
    // output is still valid in the state that reads it.
    case (state)
      S_MAX_SETUP, S_MAX_WAIT: begin
        max_a = s_buf[cnt];
        max_b = running_max;
      end
      S_SUB_SETUP, S_SUB_WAIT, S_EXP_START: begin
        sub_a = s_buf[cnt];
        sub_b = running_max;
        if (state == S_EXP_START) exp_x = sub_y;
      end
      S_SUM_SETUP, S_SUM_WAIT: begin
        if (cnt < row_len) begin
          add_a = sum;
          add_b = e_buf[cnt];
        end
      end
      S_NORM_DIV, S_NORM_WAIT: begin
        div_a = e_buf[cnt];
        div_b = sum;
      end
      default: ;
    endcase
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state       <= S_IDLE;
      cnt         <= '0;
      row_len     <= '0;
      done        <= 1'b0;
      running_max <= '0;
      sum         <= '0;
    end else begin
      done <= 1'b0;

      case (state)
        S_IDLE: begin
          if (valid_in) begin
            s_buf[0] <= data_in;
            if (last_in) begin
              row_len <= CNT_W'(1);
              state   <= S_MAX_INIT;
            end else begin
              cnt   <= CNT_W'(1);
              state <= S_LOAD;
            end
          end
        end

        S_LOAD: begin
          if (valid_in) begin
            s_buf[cnt] <= data_in;
            if (last_in) begin
              row_len <= cnt + 1'b1;
              cnt     <= '0;
              state   <= S_MAX_INIT;
            end else begin
              cnt <= cnt + 1'b1;
            end
          end
        end

        S_MAX_INIT: begin
          running_max <= s_buf[0];
          if (row_len == CNT_W'(1)) begin
            cnt   <= '0;
            state <= S_SUB_SETUP;   // single-element row: max is s_buf[0]
          end else begin
            cnt   <= CNT_W'(1);
            state <= S_MAX_SETUP;
          end
        end

        S_MAX_SETUP: state <= S_MAX_WAIT;
        S_MAX_WAIT:  state <= S_MAX_READ;

        S_MAX_READ: begin
          running_max <= max_y;
          if (cnt == row_len - 1'b1) begin
            cnt   <= '0;
            state <= S_SUB_SETUP;
          end else begin
            cnt   <= cnt + 1'b1;
            state <= S_MAX_SETUP;
          end
        end

        S_SUB_SETUP: state <= S_SUB_WAIT;
        S_SUB_WAIT:  state <= S_EXP_START;
        S_EXP_START: state <= S_EXP_WAIT;

        S_EXP_WAIT: begin
          if (exp_done) state <= S_STORE;
        end

        S_STORE: begin
          e_buf[cnt] <= exp_y;
          if (cnt == row_len - 1'b1) begin
            cnt   <= '0;
            state <= S_SUM_INIT;
          end else begin
            cnt   <= cnt + 1'b1;
            state <= S_SUB_SETUP;
          end
        end

        S_SUM_INIT: begin
          sum   <= e_buf[0];
          cnt   <= CNT_W'(1);
          state <= S_SUM_SETUP;
        end

        S_SUM_SETUP: begin
          if (cnt < row_len) begin
            state <= S_SUM_WAIT;
          end else begin
            cnt   <= '0;
            state <= S_NORM_DIV;
          end
        end

        S_SUM_WAIT: state <= S_SUM_READ;

        S_SUM_READ: begin
          sum   <= add_y;
          cnt   <= cnt + 1'b1;
          state <= S_SUM_SETUP;
        end

        S_NORM_DIV: state <= S_NORM_WAIT;
        S_NORM_WAIT: state <= S_NORM_OUT;

        S_NORM_OUT: begin
          p_buf[cnt] <= p_bf;
          if (cnt == row_len - 1'b1) begin
            state <= S_DONE;
          end else begin
            cnt   <= cnt + 1'b1;
            state <= S_NORM_DIV;
          end
        end

        S_DONE: begin
          done  <= 1'b1;
          state <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
