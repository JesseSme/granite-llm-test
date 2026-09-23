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
// The unit computes using a single shared FPU.
// ready_o goes low during computation; backpressure-aware upstream must
// wait for ready before presenting the next element.
//
// Weight/bias loaded via load interface before processing starts.
// Circular buffer (3 x CHANNELS) stores previous timesteps.
//
// FPU timing: the fp_unit has a 1-stage registered output.
// Inputs presented at posedge N produce a result at posedge N+1.
// Due to non-blocking assignment semantics within a single always_ff block,
// we cannot read fpu_y in the same cycle we present new inputs. The FSM
// uses explicit WAIT states to ensure fpu_y is stable before reading.

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
  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0]  fpu_cmp;
  logic [4:0]  fpu_flags;

  fp_unit #(.W_EXP(W_EXP), .W_MANT(W_MANT32)) u_fpu (
    .clk   (clk),
    .rst_n (rst_n),
    .mode  (fpu_mode),
    .rm    (fpu_rm),
    .a     (fpu_a),
    .b     (fpu_b),
    .c     (fpu_c),
    .y     (fpu_y),
    .cmp   (fpu_cmp),
    .flags (fpu_flags)
  );

  // ----------------------------------------------------------------
  // Pipeline registers
  // ----------------------------------------------------------------
  logic [W_FP32-1:0] acc;

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
  // FPU timing: inputs presented at posedge N → result at posedge N+1.
  // Due to non-blocking assignments, we CANNOT read fpu_y in the same
  // cycle we present new inputs. Each state that presents FPU inputs
  // is followed by a WAIT state where the result becomes available.
  //
  // Sequence per element (16 cycles):
  //   S_IDLE      : present MUL(weight[0], input)
  //   S_WAIT_MUL0 : (FPU computes)
  //   S_MUL0      : read MUL0 result → acc. present MUL(weight[1], buf[0])
  //   S_WAIT_MUL1 : (FPU computes)
  //   S_MUL1      : read MUL1 result. present ADD(acc, product1)
  //   S_WAIT_ADD01: (FPU computes)
  //   S_ADD01     : read ADD result → acc. present MUL(weight[2], buf[1])
  //   S_WAIT_MUL2 : (FPU computes)
  //   S_MUL2      : read MUL2 result. present ADD(acc, product2)
  //   S_WAIT_ADD012:(FPU computes)
  //   S_ADD012    : read ADD result → acc. present MUL(weight[3], buf[2])
  //   S_WAIT_MUL3 : (FPU computes)
  //   S_MUL3      : read MUL3 result. present ADD(acc, product3)
  //   S_WAIT_BIAS : (FPU computes)
  //   S_ADD_BIAS  : read ADD result → acc. present ADD(acc, bias)
  //   S_WAIT_OUT  : (FPU computes)
  //   S_OUTPUT    : read ADD result. output data. advance buffer.
  // ----------------------------------------------------------------
  typedef enum logic [4:0] {
    S_IDLE,
    S_WAIT_MUL0,
    S_MUL0,
    S_WAIT_MUL1,
    S_MUL1,
    S_WAIT_ADD01,
    S_ADD01,
    S_WAIT_MUL2,
    S_MUL2,
    S_WAIT_ADD012,
    S_ADD012,
    S_WAIT_MUL3,
    S_MUL3,
    S_WAIT_BIAS,
    S_ADD_BIAS,
    S_WAIT_OUT,
    S_OUTPUT
  } state_t;

  state_t state;

  // ----------------------------------------------------------------
  // Ready: can accept input only when idle
  // ----------------------------------------------------------------
  assign ready_o = (state == S_IDLE);

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
    end else begin
      valid_o <= 1'b0;

      case (state)
        // ----------------------------------------------------------
        S_IDLE: begin
          if (valid_i) begin
            cur_input   <= data_i;
            cur_channel <= ch_cnt;
            // Present MUL(weight[c,0], data_i)
            fpu_mode <= fp_pkg::OP_MUL;
            fpu_a    <= bf16_to_fp32(weight_mem[ch_cnt][0]);
            fpu_b    <= bf16_to_fp32(data_i);
            fpu_c    <= '0;
            state    <= S_WAIT_MUL0;
          end
        end

        // ----------------------------------------------------------
        // WAIT states: FPU is computing, result not yet available
        // ----------------------------------------------------------
        S_WAIT_MUL0: state <= S_MUL0;
        S_WAIT_MUL1: state <= S_MUL1;
        S_WAIT_ADD01: state <= S_ADD01;
        S_WAIT_MUL2: state <= S_MUL2;
        S_WAIT_ADD012: state <= S_ADD012;
        S_WAIT_MUL3: state <= S_MUL3;
        S_WAIT_BIAS: state <= S_ADD_BIAS;
        S_WAIT_OUT:  state <= S_OUTPUT;

        // ----------------------------------------------------------
        // MUL0: fpu_y = weight[c,0] * data_i (from S_IDLE)
        // Read result, present MUL1 inputs
        // ----------------------------------------------------------
        S_MUL0: begin
          acc <= fpu_y;
          // Present MUL(weight[c,1], buf[c][0])  — t-1
          fpu_mode <= fp_pkg::OP_MUL;
          fpu_a    <= bf16_to_fp32(weight_mem[cur_channel][1]);
          fpu_b    <= bf16_to_fp32(buf_rd[0]);
          fpu_c    <= '0;
          state    <= S_WAIT_MUL1;
        end

        // ----------------------------------------------------------
        // MUL1: fpu_y = weight[c,1] * buf[c][0]
        // Read result, present ADD01 inputs
        // ----------------------------------------------------------
        S_MUL1: begin
          // Present ADD(acc, product1)
          fpu_mode <= fp_pkg::OP_ADD;
          fpu_a    <= acc;
          fpu_b    <= fpu_y;
          fpu_c    <= '0;
          state    <= S_WAIT_ADD01;
        end

        // ----------------------------------------------------------
        // ADD01: fpu_y = acc + product1
        // Read result, present MUL2 inputs
        // ----------------------------------------------------------
        S_ADD01: begin
          acc <= fpu_y;
          // Present MUL(weight[c,2], buf[c][1])  — t-2
          fpu_mode <= fp_pkg::OP_MUL;
          fpu_a    <= bf16_to_fp32(weight_mem[cur_channel][2]);
          fpu_b    <= bf16_to_fp32(buf_rd[1]);
          fpu_c    <= '0;
          state    <= S_WAIT_MUL2;
        end

        // ----------------------------------------------------------
        // MUL2: fpu_y = weight[c,2] * buf[c][1]
        // Read result, present ADD012 inputs
        // ----------------------------------------------------------
        S_MUL2: begin
          // Present ADD(acc, product2)
          fpu_mode <= fp_pkg::OP_ADD;
          fpu_a    <= acc;
          fpu_b    <= fpu_y;
          fpu_c    <= '0;
          state    <= S_WAIT_ADD012;
        end

        // ----------------------------------------------------------
        // ADD012: fpu_y = sum(0,1) + product2
        // Read result, present MUL3 inputs
        // ----------------------------------------------------------
        S_ADD012: begin
          acc <= fpu_y;
          // Present MUL(weight[c,3], buf[c][2])  — t-3
          fpu_mode <= fp_pkg::OP_MUL;
          fpu_a    <= bf16_to_fp32(weight_mem[cur_channel][3]);
          fpu_b    <= bf16_to_fp32(buf_rd[2]);
          fpu_c    <= '0;
          state    <= S_WAIT_MUL3;
        end

        // ----------------------------------------------------------
        // MUL3: fpu_y = weight[c,3] * buf[c][2]
        // Read result, present ADD_BIAS inputs
        // ----------------------------------------------------------
        S_MUL3: begin
          // Present ADD(acc, product3)
          fpu_mode <= fp_pkg::OP_ADD;
          fpu_a    <= acc;
          fpu_b    <= fpu_y;
          fpu_c    <= '0;
          state    <= S_WAIT_BIAS;
        end

        // ----------------------------------------------------------
        // ADD_BIAS: fpu_y = sum_of_products
        // Read result, present ADD(acc, bias) inputs
        // Store current input into history buffer
        // ----------------------------------------------------------
        S_ADD_BIAS: begin
          acc <= fpu_y;
          // Store current input into history buffer
          buf_mem[buf_wr_ptr][cur_channel] <= cur_input;
          // Present ADD(sum_of_products, bias)
          fpu_mode <= fp_pkg::OP_ADD;
          fpu_a    <= fpu_y;
          fpu_b    <= bf16_to_fp32(bias_mem[cur_channel]);
          fpu_c    <= '0;
          state    <= S_WAIT_OUT;
        end

        // ----------------------------------------------------------
        // OUTPUT: fpu_y = final result (sum + bias)
        // ----------------------------------------------------------
        S_OUTPUT: begin
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

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
