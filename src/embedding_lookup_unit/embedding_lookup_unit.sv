// Embedding lookup (nn.Embedding) hardware unit.
//
// Maps a token ID to its dense hidden-state vector and scales it by
// embedding_multiplier = 12 (GraniteMoeHybridModel.forward:
// `inputs_embeds = self.embed_tokens(input_ids) * self.embedding_multiplier`):
//
//   data_out[i] = embed_table[token_id][i] * 12.0   for i in [0, DIM)
//
// Storage: parameterized (VOCAB x DIM) bfloat16 table RAM, written through the
// load port before lookups start (weights are tied to lm_head; see
// LLM_LAYER_DESCRIPTION.md §1).
//
// Streaming: one token in (valid_in, accepted when idle), DIM elements out
// (valid_out, one per cycle after a 4-cycle pipeline fill). The shared fp_unit
// performs the bfloat16 multiply by 12.0 (exactly representable, so the result
// matches PyTorch's bfloat16 scalar multiply).
//
// fp_unit protocol: one MUL transaction per output element, started by a
// 1-cycle in_valid pulse; fp_mul_pipe accepts a start every cycle and this
// unit issues element i+1 while element i's result travels through the
// 4-cycle pipeline, so the element-per-cycle output throughput is preserved.
// The scaling is exactly representable, so results are unchanged.
//
// bfloat16 format note: this unit instantiates fp_unit at W_MANT = 7 (BF16);
// the pipelined datapaths share the same latencies as fp32 (MUL 4, ADD 3).
//
// FSM: IDLE -> OUT (issue ahead + emit) -> DONE -> IDLE.

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */

module embedding_lookup_unit #(
  parameter int VOCAB  = 100352,
  parameter int DIM    = 768,
  parameter int W_DATA = 16,
  parameter int T_W    = $clog2(VOCAB),
  parameter int D_W    = $clog2(DIM)
) (
  input  logic              clk,
  input  logic              rst_n,

  // Table load interface (one element per cycle, before lookups start)
  input  logic              load_en,
  input  logic [T_W-1:0]    load_token,
  input  logic [D_W-1:0]    load_idx,
  input  logic [W_DATA-1:0] load_wdata,

  // Lookup interface
  input  logic              valid_in,
  input  logic [T_W-1:0]    token_id,
  output logic [W_DATA-1:0] data_out,
  output logic              valid_out,
  output logic              busy
);

  // ---------------------------------------------------------------- storage
  logic [W_DATA-1:0] embed_mem [0:VOCAB-1][0:DIM-1];

  always_ff @(posedge clk) begin
    if (load_en)
      embed_mem[load_token][load_idx] <= load_wdata;
  end

  // ---------------------------------------------------------------- constants
  // 12.0 in bfloat16: 1.5 * 2^3 -> exponent 130 (0x82), mantissa 0x40.
  localparam logic [W_DATA-1:0] SCALE_12 = 16'h4140;

  // ---------------------------------------------------------------- FPU
  fp_pkg::op_t       fpu_mode;
  fp_pkg::rounding_t fpu_rm;
  logic [W_DATA-1:0] fpu_a, fpu_b, fpu_c;
  logic [W_DATA-1:0] fpu_y;
  logic              fpu_start;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0]        fpu_cmp;
  logic [4:0]        fpu_flags;
  logic              fpu_out_valid;
  /* verilator lint_on UNUSEDSIGNAL */

  fp_unit #(.W_EXP(8), .W_MANT(7)) u_fp (
    .clk(clk), .rst_n(rst_n),
    .mode(fpu_mode), .rm(fpu_rm),
    .a(fpu_a), .b(fpu_b), .c(fpu_c),
    .y(fpu_y), .cmp(fpu_cmp), .flags(fpu_flags),
    .in_valid(fpu_start), .out_valid(fpu_out_valid)
  );

  // ---------------------------------------------------------------- FSM
  typedef enum logic [1:0] {
    IDLE,
    OUT,
    DONE
  } state_t;

  state_t state;
  logic [T_W-1:0] token;
  logic [D_W-1:0] cnt;    // element being emitted
  logic [D_W-1:0] iss;    // element being issued (cnt + pipeline depth)

  // MUL issue: element 0 while accepting the token, then element iss while
  // streaming out; the multiply is exactly representable, so the result is
  // unchanged. Only DIM elements are issued.
  wire issue_now = (state == IDLE && valid_in) || (state == OUT && iss < D_W'(DIM));

  always_comb begin
    fpu_rm   = fp_pkg::RM_RNE;
    fpu_mode = fp_pkg::OP_MUL;
    fpu_b    = SCALE_12;
    fpu_c    = '0;
    fpu_a    = '0;

    if (state == IDLE && valid_in)
      fpu_a = embed_mem[token_id][0];
    else if (state == OUT && iss < D_W'(DIM))
      fpu_a = embed_mem[token][iss];
  end

  assign fpu_start = issue_now;

  // Result valid pipeline: fp_mul_pipe's binary32/BF16 latency is 4 cycles
  // from the start cycle to the result cycle.
  logic [3:0] dv;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state     <= IDLE;
      token     <= '0;
      cnt       <= '0;
      iss       <= '0;
      data_out  <= '0;
      valid_out <= 1'b0;
      busy      <= 1'b0;
      dv        <= 4'b0;
    end else begin
      valid_out <= 1'b0;
      dv[0]     <= issue_now;
      dv[1]     <= dv[0];
      dv[2]     <= dv[1];
      dv[3]     <= dv[2];

      case (state)
        // ------------------------------------------------------------ IDLE
        IDLE: begin
          busy <= 1'b0;
          if (valid_in) begin
            token <= token_id;
            cnt   <= '0;
            iss   <= D_W'(1);
            state <= OUT;
            busy  <= 1'b1;
          end
        end

        // ------------------------------------------------------------- OUT
        OUT: begin
          if (issue_now)
            iss <= iss + 1'b1;
          if (dv[3]) begin
            data_out  <= fpu_y;
            valid_out <= 1'b1;
            if (cnt == D_W'(DIM - 1))
              state <= DONE;
            else
              cnt <= cnt + 1'b1;
          end
        end

        // ------------------------------------------------------------ DONE
        DONE: begin
          state <= IDLE;
          busy  <= 1'b0;
        end

        default: state <= IDLE;
      endcase
    end
  end

endmodule
