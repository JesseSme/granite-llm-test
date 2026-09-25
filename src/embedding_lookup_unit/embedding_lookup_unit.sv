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
// (valid_out, one per cycle). The shared fp_unit performs the bfloat16
// multiply by 12.0 (exactly representable, so the result matches PyTorch's
// bfloat16 scalar multiply).
//
// FSM: IDLE -> OUT (DIM cycles) -> DONE -> IDLE.
//   IDLE: accept token, issue MUL for element 0
//   OUT : emit element cnt (registered fp_unit result), issue MUL for cnt+1
//   DONE: final element is emitted (valid_out held), busy deasserts

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
  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0]        fpu_cmp;
  logic [4:0]        fpu_flags;
  logic              unused_out_valid_fp;
  /* verilator lint_on UNUSEDSIGNAL */

  fp_unit #(.W_EXP(8), .W_MANT(7)) u_fp (
    .clk(clk), .rst_n(rst_n),
    .mode(fpu_mode), .rm(fpu_rm),
    .a(fpu_a), .b(fpu_b), .c(fpu_c),
    .y(fpu_y), .cmp(fpu_cmp), .flags(fpu_flags),
    .in_valid(1'b1), .out_valid(unused_out_valid_fp)
  );

  // ---------------------------------------------------------------- FSM
  typedef enum logic [1:0] {
    IDLE,
    OUT,
    DONE
  } state_t;

  state_t state;
  logic [T_W-1:0] token;
  logic [D_W-1:0] cnt;

  always_comb begin
    fpu_rm   = fp_pkg::RM_RNE;
    fpu_mode = fp_pkg::OP_MUL;
    fpu_a    = '0;
    fpu_b    = SCALE_12;
    fpu_c    = '0;

    if (state == IDLE && valid_in) begin
      fpu_a = embed_mem[token_id][0];
    end else if (state == OUT && cnt != D_W'(DIM - 1)) begin
      fpu_a = embed_mem[token][cnt + 1'b1];
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state     <= IDLE;
      token     <= '0;
      cnt       <= '0;
      data_out  <= '0;
      valid_out <= 1'b0;
      busy      <= 1'b0;
    end else begin
      valid_out <= 1'b0;

      case (state)
        // ------------------------------------------------------------ IDLE
        IDLE: begin
          busy <= 1'b0;
          if (valid_in) begin
            token <= token_id;
            cnt   <= '0;
            state <= OUT;
            busy  <= 1'b1;
          end
        end

        // ------------------------------------------------------------- OUT
        // Emit the fp_unit result issued in the previous cycle, and issue the
        // multiply for the next element.
        OUT: begin
          data_out  <= fpu_y;
          valid_out <= 1'b1;
          if (cnt == D_W'(DIM - 1))
            state <= DONE;
          else
            cnt <= cnt + 1'b1;
        end

        // ------------------------------------------------------------ DONE
        // Final element is on data_out/valid_out this cycle.
        DONE: begin
          state <= IDLE;
          busy  <= 1'b0;
        end

        default: state <= IDLE;
      endcase
    end
  end

endmodule
