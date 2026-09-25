// RMSNorm (Root Mean Square Normalization) hardware unit.
//
// Normalizes each vector independently:
//   output = input / sqrt(mean(input²) + eps) * weight
//
// Sequential FSM processing one vector of WIDTH elements at a time.
// Uses the fp_unit for all floating-point operations (MUL, ADD, DIV, SQRT).
//
// Phases:
//   1. LOAD_WEIGHT: Buffer weight vector (WIDTH elements via weight_in)
//   2. BUFFER_INPUT: Accept input vector (WIDTH elements via data_in)
//   3. ACCUM_SQ: Compute sum of squares from buffered input
//   4. MEAN_EPS: Divide sum by WIDTH, add eps
//   5. SQRT_RMS: Compute sqrt(mean + eps)
//   6. NORMALIZE: Read input, divide by rms, multiply by weight, output
//
// bfloat16 I/O (W_DATA=16), parameterizable vector width (default 768).
// fp_unit latency: 1 cycle (registered output).

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off UNUSED */

module rmsnorm_unit #(
  parameter int WIDTH = 768,
  parameter int W_DATA = 16,
  parameter int W_CNT = $clog2(WIDTH + 1)
) (
  input  logic                  clk,
  input  logic                  rst_n,
  input  logic                  valid_in,
  input  logic [W_DATA-1:0]    data_in,
  input  logic                  weight_valid,
  input  logic [W_DATA-1:0]    weight_in,
  output logic [W_DATA-1:0]    data_out,
  output logic                  valid_out,
  output logic                  busy
);

  // --------------------------------------------------------------- states
  typedef enum logic [2:0] {
    IDLE,
    LOAD_WEIGHT,
    BUFFER_INPUT,
    ACCUM_SQ,
    MEAN_EPS,
    SQRT_RMS,
    NORMALIZE,
    DONE
  } state_t;

  state_t state;

  // -------------------------------------------------------- counters
  logic [W_CNT-1:0] cnt;

  // -------------------------------------------------------- data regs
  logic [31:0] sum_sq;
  logic [W_DATA-1:0] mean_eps_val;
  logic [31:0] rms_val;
  logic [31:0] fp_result;

  // --------------------------------------------------- storage
  logic [W_DATA-1:0] input_mem  [0:WIDTH-1];
  logic [W_DATA-1:0] weight_mem [0:WIDTH-1];
  logic              weight_loaded;

  // -------------------------------------------------- FPU interface
  logic [31:0]   fpu_a, fpu_b, fpu_c;
  fp_pkg::op_t         fpu_mode;
  fp_pkg::rounding_t   fpu_rm;
  // Package names are fully qualified (no `import fp_pkg::*;`) because
  // Yosys/SymbiYosys do not support module-scope imports.
  logic [31:0]   fpu_y;
  logic [1:0]          fpu_cmp;
  logic [4:0]          fpu_flags;
  /* verilator lint_off UNUSEDSIGNAL */
  logic                unused_out_valid_fp;
  /* verilator lint_on UNUSEDSIGNAL */

  function automatic logic [31:0] bf16_to_fp32(input logic [W_DATA-1:0] v);
    bf16_to_fp32 = {v, 16'h0};
  endfunction

  logic [W_DATA-1:0] out_bf16;
  fp32_to_bf16_round u_round_out (.x(fpu_y), .y(out_bf16));

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_fp (
    .clk(clk), .rst_n(rst_n),
    .mode(fpu_mode), .rm(fpu_rm),
    .a(fpu_a), .b(fpu_b), .c(fpu_c),
    .y(fpu_y), .cmp(fpu_cmp), .flags(fpu_flags),
    .in_valid(1'b1), .out_valid(unused_out_valid_fp)
  );

  // -------------------------------------------------- pipeline tracking
  typedef enum logic [2:0] {
    FP_IDLE,
    FP_SQUARE,    // input * input
    FP_ACCUM,     // sum + square
    FP_DIV_MEAN,  // sum / WIDTH
    FP_ADD_EPS,   // mean + eps
    FP_SQRT_OP,   // sqrt(val)
    FP_DIV_NORM,  // input / rms
    FP_MUL_WT     // normalized * weight
  } fp_phase_t;

  fp_phase_t fp_phase;

  // -------------------------------------------------- pre-computed constants
  // eps = ~1e-5 in bfloat16: 0x3780 = 2^-17 ≈ 7.6e-6
  localparam logic [31:0] EPS = 32'h3727C5AC;  // 1e-5 (config.rms_norm_eps)
  // WIDTH = 768 in bfloat16: 1.5 * 2^9 = 0x4440
  localparam logic [31:0] WIDTH_FP32 = 32'h44400000;  // (float)WIDTH

  // -------------------------------------------------- FPU drive
  always_comb begin
    fpu_rm  = fp_pkg::RM_RNE;
    fpu_mode = fp_pkg::OP_ADD;
    fpu_a = '0;
    fpu_b = '0;
    fpu_c = '0;

    if (state == ACCUM_SQ) begin
      if (fp_phase == FP_IDLE) begin
        fpu_mode = fp_pkg::OP_MUL;
        fpu_a = bf16_to_fp32(input_mem[cnt]);
        fpu_b = bf16_to_fp32(input_mem[cnt]);
      end else if (fp_phase == FP_SQUARE) begin
        fpu_mode = fp_pkg::OP_ADD;
        fpu_a = sum_sq;
        fpu_b = fpu_y;
      end
    end else if (state == MEAN_EPS) begin
      if (fp_phase == FP_IDLE) begin
        fpu_mode = fp_pkg::OP_DIV;
        fpu_a = sum_sq;
        fpu_b = WIDTH_FP32;
      end else if (fp_phase == FP_DIV_MEAN) begin
        fpu_mode = fp_pkg::OP_ADD;
        fpu_a = fpu_y;  // DIV result from previous cycle
        fpu_b = EPS;
      end
    end else if (state == SQRT_RMS && fp_phase == FP_IDLE) begin
      fpu_mode = fp_pkg::OP_SQRT;
      fpu_a = fp_result;  // mean+eps captured from MEAN_EPS
    end else if (state == NORMALIZE) begin
      if (fp_phase == FP_IDLE) begin
        fpu_mode = fp_pkg::OP_DIV;
        fpu_a = bf16_to_fp32(input_mem[cnt]);
        fpu_b = rms_val;
      end else if (fp_phase == FP_DIV_NORM) begin
        fpu_mode = fp_pkg::OP_MUL;
        fpu_a = fpu_y;  // DIV result from previous cycle's registered output
        fpu_b = bf16_to_fp32(weight_mem[cnt]);
      end
    end
  end

  // -------------------------------------------------- FSM
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state         <= IDLE;
      cnt           <= '0;
      sum_sq        <= '0;
      mean_eps_val  <= '0;
      rms_val       <= '0;
      fp_result     <= '0;
      fp_phase      <= FP_IDLE;
      weight_loaded <= 1'b0;
      data_out      <= '0;
      valid_out     <= 1'b0;
      busy          <= 1'b0;
    end else begin
      valid_out <= 1'b0;

      case (state)
        // -------------------------------------------------- IDLE
        IDLE: begin
          busy <= 1'b0;
          if (weight_valid && !weight_loaded) begin
            weight_mem[cnt] <= weight_in;
            cnt <= cnt + 1'b1;
            if (cnt == WIDTH - 1) begin
              weight_loaded <= 1'b1;
              cnt <= '0;
            end
          end else if (valid_in && weight_loaded) begin
            state     <= BUFFER_INPUT;
            cnt       <= 1;  // element 0 stored below, BUFFER_INPUT starts at 1
            input_mem[0] <= data_in;
            busy      <= 1'b1;
          end
        end

        // -------------------------------------------------- BUFFER_INPUT
        BUFFER_INPUT: begin
          if (valid_in) begin
            input_mem[cnt] <= data_in;
            cnt <= cnt + 1'b1;
            if (cnt == WIDTH - 1) begin
              state  <= ACCUM_SQ;
              cnt    <= '0;
              sum_sq <= '0;
            end
          end
        end

        // -------------------------------------------------- ACCUM_SQ
        // 2 FPU cycles per element: MUL (square), then ADD (accumulate)
        ACCUM_SQ: begin
          case (fp_phase)
            FP_IDLE: begin
              // Drive MUL inputs this cycle; result ready next cycle
              fp_phase <= FP_SQUARE;
            end
            FP_SQUARE: begin
              // fpu_y = square. Drive ADD inputs this cycle
              fp_phase <= FP_ACCUM;
            end
            FP_ACCUM: begin
              // fpu_y = sum + square. Store and advance
              sum_sq <= fpu_y;
              cnt    <= cnt + 1'b1;
              fp_phase <= FP_IDLE;
              if (cnt == WIDTH - 1) begin
                state <= MEAN_EPS;
                cnt   <= '0;
              end
            end
            default: fp_phase <= FP_IDLE;
          endcase
        end

        // -------------------------------------------------- MEAN_EPS
        // DIV (sum/width), then ADD (+eps)
        MEAN_EPS: begin
          case (fp_phase)
            FP_IDLE: begin
              fp_phase <= FP_DIV_MEAN;
            end
            FP_DIV_MEAN: begin
              // fpu_y = mean. Drive ADD for eps
              fp_result <= fpu_y;
              fp_phase  <= FP_ADD_EPS;
            end
            FP_ADD_EPS: begin
              // fpu_y = mean + eps
              fp_result <= fpu_y;
              fp_phase  <= FP_IDLE;
              state     <= SQRT_RMS;
            end
            default: fp_phase <= FP_IDLE;
          endcase
        end

        // -------------------------------------------------- SQRT_RMS
        SQRT_RMS: begin
          case (fp_phase)
            FP_IDLE: begin
              fp_phase <= FP_SQRT_OP;
            end
            FP_SQRT_OP: begin
              // fpu_y = sqrt(mean + eps) = rms
              rms_val  <= fpu_y;
              fp_phase <= FP_IDLE;
              state    <= NORMALIZE;
              cnt      <= '0;
            end
            default: fp_phase <= FP_IDLE;
          endcase
        end

        // -------------------------------------------------- NORMALIZE
        // DIV (input / rms), then MUL (* weight)
        NORMALIZE: begin
          case (fp_phase)
            FP_IDLE: begin
              fp_phase <= FP_DIV_NORM;
            end
            FP_DIV_NORM: begin
              // fpu_y = input / rms. Drive MUL for weight
              fp_result <= fpu_y;
              fp_phase  <= FP_MUL_WT;
            end
            FP_MUL_WT: begin
              // fpu_y = normalized * weight
              data_out  <= out_bf16;
              valid_out <= 1'b1;
              cnt       <= cnt + 1'b1;
              fp_phase  <= FP_IDLE;
              if (cnt == WIDTH - 1) begin
                state <= DONE;
              end
            end
            default: fp_phase <= FP_IDLE;
          endcase
        end

        // -------------------------------------------------- DONE
        DONE: begin
          state <= IDLE;
          busy  <= 1'b0;
        end

        default: state <= IDLE;
      endcase
    end
  end

endmodule
