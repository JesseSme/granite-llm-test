// RMSNorm (Root Mean Square Normalization) hardware unit.
//
// Normalizes each vector independently:
//   output = input / sqrt(mean(input²) + eps) * weight
//
// Sequential FSM processing one vector of WIDTH elements at a time.
// Uses one pipelined fp_unit for all floating-point operations
// (MUL, ADD, DIV, SQRT): one transaction at a time, started by a 1-cycle
// `fpu_start` pulse, with `fpu_y` valid on the `fpu_out_valid` pulse. Each
// operation therefore runs through a REQ phase (start pulse) and a WAIT phase
// (latch on out_valid), instead of the previous 1-cycle-latency schedule.
// The operation order and operands are unchanged, so results are bit-identical.
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
  logic [31:0] mean_eps_val;
  logic [31:0] rms_val;
  logic [31:0] square_val;
  logic [31:0] div_val;

  // --------------------------------------------------- storage
  logic [W_DATA-1:0] input_mem  [0:WIDTH-1];
  logic [W_DATA-1:0] weight_mem [0:WIDTH-1];
  logic              weight_loaded;

  // -------------------------------------------------- FPU interface
  logic [31:0]   fpu_a, fpu_b, fpu_c;
  fp_pkg::op_t         fpu_mode;
  fp_pkg::rounding_t   fpu_rm;
  logic                fpu_start;
  // Package names are fully qualified (no `import fp_pkg::*;`) because
  // Yosys/SymbiYosys do not support module-scope imports.
  logic [31:0]   fpu_y;
  logic [1:0]          fpu_cmp;
  logic [4:0]          fpu_flags;
  logic                fpu_out_valid;

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
    .in_valid(fpu_start), .out_valid(fpu_out_valid)
  );

  // -------------------------------------------------- op phase tracking
  // One REQ phase (1-cycle start pulse) and one WAIT phase (latch on
  // out_valid) per operation.
  typedef enum logic [3:0] {
    FP_IDLE,
    SQ_REQ, SQ_WAIT,   // square  = input * input
    AC_REQ, AC_WAIT,   // sum_sq  = sum_sq + square
    DM_REQ, DM_WAIT,   // mean    = sum_sq / WIDTH
    AE_REQ, AE_WAIT,   // mean+eps= mean + eps
    SR_REQ, SR_WAIT,   // rms     = sqrt(mean + eps)
    DN_REQ, DN_WAIT,   // norm    = input / rms
    MW_REQ, MW_WAIT    // out     = norm * weight
  } fp_phase_t;

  fp_phase_t fp_phase;

  // -------------------------------------------------- pre-computed constants
  localparam logic [31:0] EPS = 32'h3727C5AC;  // 1e-5 (config.rms_norm_eps)
  localparam logic [31:0] WIDTH_FP32 = 32'h44400000;  // (float)WIDTH

  // -------------------------------------------------- FPU drive
  always_comb begin
    fpu_rm   = fp_pkg::RM_RNE;
    fpu_mode = fp_pkg::OP_ADD;
    fpu_a    = '0;
    fpu_b    = '0;
    fpu_c    = '0;
    fpu_start = 1'b0;

    // The operation context (mode/operands) is held stable through the
    // REQ and WAIT phases: the iterative div/sqrt engine uses mode live for
    // the whole transaction, so a SQRT must keep mode = OP_SQRT until done.
    case (fp_phase)
      SQ_REQ, SQ_WAIT: begin
        fpu_start = (fp_phase == SQ_REQ);
        fpu_mode  = fp_pkg::OP_MUL;
        fpu_a     = bf16_to_fp32(input_mem[cnt]);
        fpu_b     = bf16_to_fp32(input_mem[cnt]);
      end
      AC_REQ, AC_WAIT: begin
        fpu_start = (fp_phase == AC_REQ);
        fpu_mode  = fp_pkg::OP_ADD;
        fpu_a     = sum_sq;
        fpu_b     = square_val;
      end
      DM_REQ, DM_WAIT: begin
        fpu_start = (fp_phase == DM_REQ);
        fpu_mode  = fp_pkg::OP_DIV;
        fpu_a     = sum_sq;
        fpu_b     = WIDTH_FP32;
      end
      AE_REQ, AE_WAIT: begin
        fpu_start = (fp_phase == AE_REQ);
        fpu_mode  = fp_pkg::OP_ADD;
        fpu_a     = div_val;
        fpu_b     = EPS;
      end
      SR_REQ, SR_WAIT: begin
        fpu_start = (fp_phase == SR_REQ);
        fpu_mode  = fp_pkg::OP_SQRT;
        fpu_a     = mean_eps_val;
      end
      DN_REQ, DN_WAIT: begin
        fpu_start = (fp_phase == DN_REQ);
        fpu_mode  = fp_pkg::OP_DIV;
        fpu_a     = bf16_to_fp32(input_mem[cnt]);
        fpu_b     = rms_val;
      end
      MW_REQ, MW_WAIT: begin
        fpu_start = (fp_phase == MW_REQ);
        fpu_mode  = fp_pkg::OP_MUL;
        fpu_a     = div_val;
        fpu_b     = bf16_to_fp32(weight_mem[cnt]);
      end
      default: ;
    endcase
  end

  // -------------------------------------------------- FSM
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state         <= IDLE;
      cnt           <= '0;
      sum_sq        <= '0;
      mean_eps_val  <= '0;
      rms_val       <= '0;
      square_val    <= '0;
      div_val       <= '0;
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
              fp_phase <= FP_IDLE;
            end
          end
        end

        // -------------------------------------------------- ACCUM_SQ
        ACCUM_SQ: begin
          case (fp_phase)
            FP_IDLE: fp_phase <= SQ_REQ;
            SQ_REQ:  fp_phase <= SQ_WAIT;
            SQ_WAIT: begin
              if (fpu_out_valid) begin
                square_val <= fpu_y;
                fp_phase   <= AC_REQ;
              end
            end
            AC_REQ: fp_phase <= AC_WAIT;
            AC_WAIT: begin
              if (fpu_out_valid) begin
                sum_sq   <= fpu_y;
                fp_phase <= FP_IDLE;
                cnt      <= cnt + 1'b1;
                if (cnt == WIDTH - 1) begin
                  state    <= MEAN_EPS;
                  cnt      <= '0;
                  fp_phase <= FP_IDLE;
                end
              end
            end
            default: fp_phase <= FP_IDLE;
          endcase
        end

        // -------------------------------------------------- MEAN_EPS
        MEAN_EPS: begin
          case (fp_phase)
            FP_IDLE: fp_phase <= DM_REQ;
            DM_REQ:  fp_phase <= DM_WAIT;
            DM_WAIT: begin
              if (fpu_out_valid) begin
                div_val  <= fpu_y;
                fp_phase <= AE_REQ;
              end
            end
            AE_REQ: fp_phase <= AE_WAIT;
            AE_WAIT: begin
              if (fpu_out_valid) begin
                mean_eps_val <= fpu_y;
                fp_phase     <= FP_IDLE;
                state        <= SQRT_RMS;
              end
            end
            default: fp_phase <= FP_IDLE;
          endcase
        end

        // -------------------------------------------------- SQRT_RMS
        SQRT_RMS: begin
          case (fp_phase)
            FP_IDLE: fp_phase <= SR_REQ;
            SR_REQ:  fp_phase <= SR_WAIT;
            SR_WAIT: begin
              if (fpu_out_valid) begin
                rms_val  <= fpu_y;
                fp_phase <= FP_IDLE;
                state    <= NORMALIZE;
                cnt      <= '0;
              end
            end
            default: fp_phase <= FP_IDLE;
          endcase
        end

        // -------------------------------------------------- NORMALIZE
        NORMALIZE: begin
          case (fp_phase)
            FP_IDLE: fp_phase <= DN_REQ;
            DN_REQ:  fp_phase <= DN_WAIT;
            DN_WAIT: begin
              if (fpu_out_valid) begin
                div_val  <= fpu_y;
                fp_phase <= MW_REQ;
              end
            end
            MW_REQ: fp_phase <= MW_WAIT;
            MW_WAIT: begin
              if (fpu_out_valid) begin
                data_out  <= out_bf16;
                valid_out <= 1'b1;
                fp_phase  <= FP_IDLE;
                cnt       <= cnt + 1'b1;
                if (cnt == WIDTH - 1) begin
                  state <= DONE;
                end
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
