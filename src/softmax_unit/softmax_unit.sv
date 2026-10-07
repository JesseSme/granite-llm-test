// Numerically stable softmax: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
//
// fp_unit has 1-cycle registered output:
//   Present inputs at T (NBA) -> fp computes at T+1 -> result valid at T+2
// Pattern: SETUP -> WAIT -> READ

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off SYNCASYNCNET */
/* verilator lint_off PINCONNECTEMPTY */

module softmax_unit #(
  parameter int N = 32
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        valid_in,
  input  logic [31:0] data_in,
  input  logic        last_in,
  output logic [31:0] data_out,
  output logic        valid_out,
  output logic        done
);

  localparam int CNT_W = $clog2(N + 1);

  logic [31:0] input_buf [0:N-1];
  logic [CNT_W-1:0] row_len;
  logic [31:0] exp_buf [0:N-1];

  logic go_max, go_sub, go_add, go_div;   // 1-cycle in_valid pulses
  logic max_ov, sub_ov, add_ov, div_ov;   // fp_unit out_valid

  logic [31:0] max_a, max_b, max_y;
  fp_unit u_fp_max (.clk(clk), .rst_n(rst_n), .mode(fp_pkg::OP_MAX),
    .rm(fp_pkg::RM_RNE), .a(max_a), .b(max_b), .c('0), .y(max_y), .cmp(), .flags(),
    .in_valid(go_max), .out_valid(max_ov));

  logic [31:0] sub_r;
  logic [31:0] p_r;
  logic [31:0] sub_a, sub_b, sub_y;
  fp_unit u_fp_sub (.clk(clk), .rst_n(rst_n), .mode(fp_pkg::OP_SUB),
    .rm(fp_pkg::RM_RNE), .a(sub_a), .b(sub_b), .c('0), .y(sub_y), .cmp(), .flags(),
    .in_valid(go_sub), .out_valid(sub_ov));

  logic [31:0] add_a, add_b, add_y;
  fp_unit u_fp_add (.clk(clk), .rst_n(rst_n), .mode(fp_pkg::OP_ADD),
    .rm(fp_pkg::RM_RNE), .a(add_a), .b(add_b), .c('0), .y(add_y), .cmp(), .flags(),
    .in_valid(go_add), .out_valid(add_ov));

  logic [31:0] div_a, div_b, div_y;
  fp_unit u_fp_div (.clk(clk), .rst_n(rst_n), .mode(fp_pkg::OP_DIV),
    .rm(fp_pkg::RM_RNE), .a(div_a), .b(div_b), .c('0), .y(div_y), .cmp(), .flags(),
    .in_valid(go_div), .out_valid(div_ov));

  logic [31:0] exp_x, exp_y;
  fp_exp u_fp_exp (.clk(clk), .rst_n(rst_n), .x(exp_x), .y(exp_y));

  localparam logic [4:0] S_IDLE      = 5'd0,
                         S_LOAD      = 5'd1,
                         S_MAX_INIT  = 5'd2,
                         S_MAX_SETUP = 5'd3,
                         S_MAX_WAIT  = 5'd4,
                         S_MAX_READ  = 5'd5,
                         S_SUB_SETUP = 5'd6,
                         S_SUB_WAIT  = 5'd7,
                         S_EXP_SETUP = 5'd8,
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

  logic [4:0] state;
  logic [CNT_W-1:0] cnt;
  logic [31:0] running_max;
  logic [31:0] sum;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state   <= S_IDLE;
      cnt     <= '0;
      row_len <= '0;
      done    <= 1'b0;
      valid_out <= 1'b0;
      go_max  <= 1'b0;
      go_sub  <= 1'b0;
      go_add  <= 1'b0;
      go_div  <= 1'b0;
      sub_r   <= '0;
      p_r     <= '0;
    end else begin
      done <= 1'b0;
      valid_out <= 1'b0;

      case (state)
        S_IDLE: begin
          if (valid_in) begin
            input_buf[0] <= data_in;
            cnt <= (CNT_W)'(1);
            state <= S_LOAD;
          end
        end

        S_LOAD: begin
          if (valid_in) begin
            input_buf[cnt] <= data_in;
            if (last_in) begin
              row_len <= cnt + 1;
              state <= S_MAX_INIT;
              cnt <= '0;
            end else begin
              cnt <= cnt + 1;
            end
          end
        end

        S_MAX_INIT: begin
          running_max <= input_buf[0];
          cnt <= (CNT_W)'(1);
          state <= S_MAX_SETUP;
        end

        S_MAX_SETUP: begin
          max_a  <= input_buf[cnt];
          max_b  <= running_max;
          go_max <= 1'b1;
          state  <= S_MAX_WAIT;
        end

        S_MAX_WAIT: begin
          if (go_max) begin
            go_max <= 1'b0;
          end else if (max_ov) begin
            // fp_unit outputs hold only one cycle past out_valid: latch here.
            running_max <= max_y;
            if (cnt == row_len - 1) begin
              state <= S_SUB_SETUP;
              cnt   <= '0;
            end else begin
              cnt   <= cnt + 1;
              state <= S_MAX_SETUP;
            end
          end
        end

        S_MAX_READ: begin
          running_max <= max_y;
          if (cnt == row_len - 1) begin
            state <= S_SUB_SETUP;
            cnt <= '0;
          end else begin
            cnt <= cnt + 1;
            state <= S_MAX_SETUP;
          end
        end

        S_SUB_SETUP: begin
          sub_a  <= input_buf[cnt];
          sub_b  <= running_max;
          go_sub <= 1'b1;
          state  <= S_SUB_WAIT;
        end

        S_SUB_WAIT: begin
          if (go_sub) go_sub <= 1'b0;
          else if (sub_ov) begin sub_r <= sub_y; state <= S_EXP_SETUP; end
        end

        S_EXP_SETUP: begin
          exp_x <= sub_r;
          state <= S_EXP_WAIT;
        end

        S_EXP_WAIT: begin
          state <= S_STORE;
        end

        S_STORE: begin
          exp_buf[cnt] <= exp_y;
          if (cnt == row_len - 1) begin
            state <= S_SUM_INIT;
            cnt <= '0;
          end else begin
            cnt <= cnt + 1;
            state <= S_SUB_SETUP;
          end
        end

        S_SUM_INIT: begin
          sum <= exp_buf[0];
          cnt <= (CNT_W)'(1);
          state <= S_SUM_SETUP;
        end

        S_SUM_SETUP: begin
          if (cnt < row_len) begin
            add_a  <= sum;
            add_b  <= exp_buf[cnt];
            go_add <= 1'b1;
            state  <= S_SUM_WAIT;
          end else begin
            state <= S_NORM_DIV;
            cnt <= '0;
          end
        end

        S_SUM_WAIT: begin
          if (go_add) begin
            go_add <= 1'b0;
          end else if (add_ov) begin
            sum   <= add_y;
            cnt   <= cnt + 1;
            state <= S_SUM_SETUP;
          end
        end

        S_SUM_READ: begin
          sum <= add_y;
          cnt <= cnt + 1;
          state <= S_SUM_SETUP;
        end

        S_NORM_DIV: begin
          div_a  <= exp_buf[0];
          div_b  <= sum;
          go_div <= 1'b1;
          cnt    <= (CNT_W)'(1);
          state  <= S_NORM_WAIT;
        end

        S_NORM_WAIT: begin
          if (go_div) go_div <= 1'b0;
          else if (div_ov) begin
            p_r    <= div_y;
            state  <= S_NORM_OUT;
          end
        end

        S_NORM_OUT: begin
          data_out <= p_r;
          valid_out <= 1'b1;
          if (cnt < row_len) begin
            div_a  <= exp_buf[cnt];
            div_b  <= sum;
            go_div <= 1'b1;
            cnt    <= cnt + 1;
            state  <= S_NORM_WAIT;
          end else begin
            state <= S_DONE;
            done  <= 1'b1;
          end
        end

        S_DONE: begin
          state <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
