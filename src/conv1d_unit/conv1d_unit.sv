// Causal depthwise 1D convolution (4-tap FIR per channel).
//
// Implements torch.nn.Conv1d(in_channels=1536, out_channels=1536,
// kernel_size=4, groups=1536, bias=True, padding=3) for Mamba2 SSM
// preprocessing.
//
// Operation (per channel c, per position t):
//   output[c,t] = weight[c,0]*input[c,t] + weight[c,1]*input[c,t-1]
//               + weight[c,2]*input[c,t-2] + weight[c,3]*input[c,t-3] + bias[c]
//
// Streaming architecture: each valid_in brings one element for one channel.
// The unit computes using a single shared fp_unit. ready_o goes low during
// computation; backpressure-aware upstream must wait for ready before
// presenting the next element.
//
// Weight/bias loaded via load interface before processing starts.
// Circular buffer (3 x CHANNELS) stores previous timesteps.
//
// fp_unit protocol: each operation is started by a 1-cycle in_valid pulse and
// its result is latched on out_valid (MUL 4, ADD 3 cycles). The FSM steps
// through the per-element chain: MUL0, MUL1, ADD01, MUL2, ADD012, MUL3,
// ADD_BIAS, ADD_OUT; each step latches the operand registers, pulses `go`
// (in_valid) for one cycle, then waits for out_valid before the next step.
// The accumulation order and operands are unchanged (fp32 accumulation, one
// final bf16 rounding), so results are bit-identical.

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */

module conv1d_unit #(
  parameter int CHANNELS = 1536,
  parameter int KERNEL   = 4,
  parameter int PADDING  = 3,
  parameter int W_DATA   = 16
) (
  input  logic        clk,
  input  logic        rst_n,

  // Stream input (active when ready_o && valid_i)
  input  logic        valid_i,
  input  logic [W_DATA-1:0] data_i,
  output logic        ready_o,

  // Stream output
  output logic        valid_o,
  output logic [W_DATA-1:0] data_o,

  // Weight/bias load interface
  input  logic        load_en,
  input  logic [$clog2(CHANNELS)-1:0] load_ch,
  input  logic [$clog2(KERNEL)-1:0] load_tap,
  input  logic [W_DATA-1:0] load_wdata,
  input  logic        load_is_bias  // 0=weight, 1=bias
);

  // Package names are fully qualified (no `import fp_pkg::*;`) because
  // Yosys/SymbiYosys do not support module-scope imports.

  // ----------------------------------------------------------------
  // Parameters
  // ----------------------------------------------------------------
  localparam int W_EXP    = 8;
  localparam int W_MANT   = 7;   // bfloat16 storage format
  localparam int W_FP     = 1 + W_EXP + W_MANT;
  localparam int W_MANT32 = 23;  // fp32 arithmetic format
  localparam int W_FP32   = 1 + W_EXP + W_MANT32;
  localparam int BUF_DEPTH = PADDING;

  // ----------------------------------------------------------------
  // Weight and bias storage
  // ----------------------------------------------------------------
  logic [W_FP-1:0] weight_mem [0:CHANNELS-1][0:KERNEL-1];
  logic [W_FP-1:0] bias_mem [0:CHANNELS-1];

  always_ff @(posedge clk) begin
    if (load_en) begin
      if (load_is_bias)
        bias_mem[load_ch] <= load_wdata;
      else
        weight_mem[load_ch][load_tap] <= load_wdata;
    end
  end

  // ----------------------------------------------------------------
  // Circular history buffer: buf_mem[slot][channel]
  //   slot 0 = most recent stored input (t-1)
  //   slot 1 = two steps back (t-2)
  //   slot 2 = three steps back (t-3)
  // ----------------------------------------------------------------
  logic [W_FP-1:0] buf_mem [0:BUF_DEPTH-1][0:CHANNELS-1];
  logic [$clog2(BUF_DEPTH)-1:0] buf_wr_ptr;

  // ----------------------------------------------------------------
  // Channel counter
  // ----------------------------------------------------------------
  logic [$clog2(CHANNELS)-1:0] ch_cnt;

  // ----------------------------------------------------------------
  // FPU instance (fp32: W_EXP=8, W_MANT=23). The bfloat16 operands are
  // widened to fp32 (exact: the 16 bits go in the high half), so all four
  // products, their sum and the bias are computed in fp32 and only the final
  // result is rounded to bfloat16 - matching the model's causal_conv1d_fn
  // kernel (fp32 accumulation, a single rounding).
  // ----------------------------------------------------------------
  fp_pkg::op_t  fpu_mode;
  fp_pkg::rounding_t fpu_rm;
  logic [W_FP32-1:0] fpu_a, fpu_b, fpu_c;
  logic [W_FP32-1:0] fpu_y;
  logic              fpu_start, fpu_out_valid;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0]  fpu_cmp;
  logic [4:0]  fpu_flags;
  /* verilator lint_on UNUSEDSIGNAL */

  fp_unit #(.W_EXP(W_EXP), .W_MANT(W_MANT32)) u_fpu (
    .clk      (clk),
    .rst_n    (rst_n),
    .in_valid (fpu_start),
    .mode     (fpu_mode),
    .rm       (fpu_rm),
    .a        (fpu_a),
    .b        (fpu_b),
    .c        (fpu_c),
    .y        (fpu_y),
    .cmp      (fpu_cmp),
    .flags    (fpu_flags),
    .out_valid(fpu_out_valid)
  );

  // ----------------------------------------------------------------
  // Pipeline registers
  // ----------------------------------------------------------------
  logic [W_FP32-1:0] acc;
  logic              go;   // in_valid pulse for the current step

  function automatic logic [W_FP32-1:0] bf16_to_fp32(input logic [W_FP-1:0] v);
    bf16_to_fp32 = {v, {(W_FP32 - W_FP){1'b0}}};
  endfunction

  logic [W_FP-1:0] out_bf16;
  fp32_to_bf16_round u_round_out (.x(fpu_y), .y(out_bf16));
  logic [W_FP-1:0] cur_input;
  logic [$clog2(CHANNELS)-1:0] cur_channel;

  // Buffer read data for current channel
  // buf_rd[0] = t-1 (most recent), buf_rd[1] = t-2, buf_rd[2] = t-3
  // The circular buffer writes at buf_wr_ptr, so the most recent entry
  // is at (buf_wr_ptr - 1), two back at (buf_wr_ptr - 2), etc.
  logic [W_FP-1:0] buf_rd [0:BUF_DEPTH-1];

  genvar gt;
  generate
    for (gt = 0; gt < BUF_DEPTH; gt++) begin : g_buf_rd
      assign buf_rd[gt] = buf_mem[(buf_wr_ptr - 1 - gt + BUF_DEPTH) % BUF_DEPTH][cur_channel];
    end
  endgenerate

  // ----------------------------------------------------------------
  // FSM
  //
  // Sequence per element: MUL0, MUL1, ADD01, MUL2, ADD012, MUL3, ADD_BIAS,
  // ADD_OUT. Each step: latch the operand registers, pulse `go` for one cycle
  // (in_valid), then wait for out_valid (fp_mul_pipe/fp_add_pipe accept a new
  // start every cycle, but this unit issues one operation at a time).
  // ----------------------------------------------------------------
  typedef enum logic [3:0] {
    S_IDLE,
    S_WAIT_MUL0,
    S_WAIT_MUL1,
    S_WAIT_ADD01,
    S_WAIT_MUL2,
    S_WAIT_ADD012,
    S_WAIT_MUL3,
    S_WAIT_BIAS,
    S_WAIT_OUT
  } state_t;

  state_t state;

  // ----------------------------------------------------------------
  // Ready: can accept input only when idle
  // ----------------------------------------------------------------
  assign ready_o   = (state == S_IDLE);
  assign fpu_start = go;

  // ----------------------------------------------------------------
  // Sequential: main FSM
  // ----------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state       <= S_IDLE;
      ch_cnt      <= '0;
      buf_wr_ptr  <= '0;
      acc         <= '0;
      valid_o     <= 1'b0;
      data_o      <= '0;
      cur_input   <= '0;
      cur_channel <= '0;
      fpu_mode    <= fp_pkg::OP_MUL;
      fpu_rm      <= fp_pkg::RM_RNE;
      fpu_a       <= '0;
      fpu_b       <= '0;
      fpu_c       <= '0;
      go          <= 1'b0;
    end else begin
      valid_o <= 1'b0;

      case (state)
        // ----------------------------------------------------------
        S_IDLE: begin
          if (valid_i) begin
            cur_input   <= data_i;
            cur_channel <= ch_cnt;
            // Start MUL(weight[c,0], data_i)
            fpu_mode <= fp_pkg::OP_MUL;
            fpu_a    <= bf16_to_fp32(weight_mem[ch_cnt][0]);
            fpu_b    <= bf16_to_fp32(data_i);
            fpu_c    <= '0;
            go       <= 1'b1;
            state    <= S_WAIT_MUL0;
          end
        end

        // ----------------------------------------------------------
        // MUL0 result -> acc; start MUL(weight[c,1], buf[c][0])  (t-1)
        // ----------------------------------------------------------
        S_WAIT_MUL0: begin
          if (go) begin
            go <= 1'b0;
          end else if (fpu_out_valid) begin
            acc      <= fpu_y;
            fpu_mode <= fp_pkg::OP_MUL;
            fpu_a    <= bf16_to_fp32(weight_mem[cur_channel][1]);
            fpu_b    <= bf16_to_fp32(buf_rd[0]);
            fpu_c    <= '0;
            go       <= 1'b1;
            state    <= S_WAIT_MUL1;
          end
        end

        // ----------------------------------------------------------
        // Start ADD(acc, product1)
        // ----------------------------------------------------------
        S_WAIT_MUL1: begin
          if (go) begin
            go <= 1'b0;
          end else if (fpu_out_valid) begin
            fpu_mode <= fp_pkg::OP_ADD;
            fpu_a    <= acc;
            fpu_b    <= fpu_y;
            fpu_c    <= '0;
            go       <= 1'b1;
            state    <= S_WAIT_ADD01;
          end
        end

        // ----------------------------------------------------------
        // ADD01 result -> acc; start MUL(weight[c,2], buf[c][1])  (t-2)
        // ----------------------------------------------------------
        S_WAIT_ADD01: begin
          if (go) begin
            go <= 1'b0;
          end else if (fpu_out_valid) begin
            acc      <= fpu_y;
            fpu_mode <= fp_pkg::OP_MUL;
            fpu_a    <= bf16_to_fp32(weight_mem[cur_channel][2]);
            fpu_b    <= bf16_to_fp32(buf_rd[1]);
            fpu_c    <= '0;
            go       <= 1'b1;
            state    <= S_WAIT_MUL2;
          end
        end

        // ----------------------------------------------------------
        // Start ADD(acc, product2)
        // ----------------------------------------------------------
        S_WAIT_MUL2: begin
          if (go) begin
            go <= 1'b0;
          end else if (fpu_out_valid) begin
            fpu_mode <= fp_pkg::OP_ADD;
            fpu_a    <= acc;
            fpu_b    <= fpu_y;
            fpu_c    <= '0;
            go       <= 1'b1;
            state    <= S_WAIT_ADD012;
          end
        end

        // ----------------------------------------------------------
        // ADD012 result -> acc; start MUL(weight[c,3], buf[c][2])  (t-3)
        // ----------------------------------------------------------
        S_WAIT_ADD012: begin
          if (go) begin
            go <= 1'b0;
          end else if (fpu_out_valid) begin
            acc      <= fpu_y;
            fpu_mode <= fp_pkg::OP_MUL;
            fpu_a    <= bf16_to_fp32(weight_mem[cur_channel][3]);
            fpu_b    <= bf16_to_fp32(buf_rd[2]);
            fpu_c    <= '0;
            go       <= 1'b1;
            state    <= S_WAIT_MUL3;
          end
        end

        // ----------------------------------------------------------
        // Start ADD(acc, product3); store the current input in the history
        // ----------------------------------------------------------
        S_WAIT_MUL3: begin
          if (go) begin
            go <= 1'b0;
          end else if (fpu_out_valid) begin
            fpu_mode <= fp_pkg::OP_ADD;
            fpu_a    <= acc;
            fpu_b    <= fpu_y;
            fpu_c    <= '0;
            go       <= 1'b1;
            state    <= S_WAIT_BIAS;
          end
        end

        // ----------------------------------------------------------
        // sum_of_products -> acc; start ADD(sum_of_products, bias)
        // ----------------------------------------------------------
        S_WAIT_BIAS: begin
          if (go) begin
            go <= 1'b0;
          end else if (fpu_out_valid) begin
            acc <= fpu_y;
            // Store current input into history buffer
            buf_mem[buf_wr_ptr][cur_channel] <= cur_input;
            fpu_mode <= fp_pkg::OP_ADD;
            fpu_a    <= fpu_y;
            fpu_b    <= bf16_to_fp32(bias_mem[cur_channel]);
            fpu_c    <= '0;
            go       <= 1'b1;
            state    <= S_WAIT_OUT;
          end
        end

        // ----------------------------------------------------------
        // Final result (sum + bias): output and advance
        // ----------------------------------------------------------
        S_WAIT_OUT: begin
          if (go) begin
            go <= 1'b0;
          end else if (fpu_out_valid) begin
            data_o  <= out_bf16;
            valid_o <= 1'b1;

            // Advance channel counter
            if (ch_cnt == CHANNELS - 1) begin
              ch_cnt     <= '0;
              buf_wr_ptr <= (buf_wr_ptr == BUF_DEPTH - 1)
                            ? '0 : buf_wr_ptr + 1'b1;
            end else begin
              ch_cnt <= ch_cnt + 1'b1;
            end

            state <= S_IDLE;
          end
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
