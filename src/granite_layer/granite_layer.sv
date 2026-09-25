// One Granite 4.0-H-350M decoder layer (mamba layer type), single token.
//
//   residual = x
//   h = input_layernorm(x)
//   h = mamba(h)                     (mamba2_unit)
//   r = residual + h * 0.246         (residual_multiplier, in residual_adder_unit)
//   residual = r
//   h = post_attention_layernorm(r)
//   h = mlp(h)                       (mlp_unit / SwiGLU)
//   y = residual + h * 0.246
//
// Sub-units are the individually verified units. The norms and the residual
// adder have valid-only handshakes (no tready); the mixer and MLP are
// AXI-Stream. Weight loads are routed by load_sel:
//   0..7 = mamba2_unit sub-selects (in_proj, out_proj, conv w/b, norm w, SSM)
//   8    = input_layernorm weight
//   9    = post_attention_layernorm weight
//   10   = mlp gate+up projection
//   11   = mlp down projection

module granite_layer #(
  parameter int HIDDEN    = 768,
  parameter int INTER     = 1536,
  parameter int NUM_HEADS = 48,
  parameter int HEAD_DIM  = 32,
  parameter int D_STATE   = 128,
  parameter int MLP_INTER = 2048,
  parameter int W_DATA    = 16,
  parameter int PW_W      = $clog2(INTER + INTER + 2 * D_STATE + NUM_HEADS),
  parameter int MLP_LO_W  = $clog2(2 * MLP_INTER),
  parameter int MLP_LI_W  = $clog2(INTER),
  parameter int LO_W      = (PW_W > MLP_LO_W) ? PW_W : MLP_LO_W,
  parameter int LI_W      = (MLP_LI_W > $clog2(HIDDEN)) ? MLP_LI_W : $clog2(HIDDEN),
  parameter int HW        = $clog2(HIDDEN)
) (
  input  logic              clk,
  input  logic              rst_n,

  input  logic              load_en,
  input  logic [3:0]        load_sel,
  input  logic [LO_W-1:0]   load_out_idx,
  input  logic [LI_W-1:0]   load_in_idx,
  input  logic [W_DATA-1:0] load_wdata,

  input  logic              s_axis_tvalid,
  output logic              s_axis_tready,
  input  logic [W_DATA-1:0] s_axis_tdata,
  input  logic              s_axis_tlast,

  output logic              m_axis_tvalid,
  input  logic              m_axis_tready,
  output logic [W_DATA-1:0] m_axis_tdata,
  output logic              m_axis_tlast,

  output logic              busy
);
  logic [W_DATA-1:0] x_buf  [0:HIDDEN-1];
  logic [W_DATA-1:0] h1_buf [0:HIDDEN-1];
  logic [W_DATA-1:0] m_buf  [0:HIDDEN-1];
  logic [W_DATA-1:0] r1_buf [0:HIDDEN-1];
  logic [W_DATA-1:0] h2_buf [0:HIDDEN-1];
  logic [W_DATA-1:0] m2_buf [0:HIDDEN-1];
  logic [W_DATA-1:0] y_buf  [0:HIDDEN-1];

  // ---------------- input_layernorm ----------------
  logic n1_win;  logic [W_DATA-1:0] n1_wdata;
  logic n1_vin;  logic [W_DATA-1:0] n1_din;
  logic n1_vout; logic [W_DATA-1:0] n1_dout; logic n1_busy;
  rmsnorm_unit #(.WIDTH(HIDDEN)) u_norm1 (
    .clk(clk), .rst_n(rst_n),
    .valid_in(n1_vin), .data_in(n1_din),
    .weight_valid(n1_win), .weight_in(n1_wdata),
    .data_out(n1_dout), .valid_out(n1_vout), .busy(n1_busy)
  );

  // ---------------- mamba2 mixer ----------------
  /* verilator lint_off UNUSEDSIGNAL */
  logic mx_svalid, mx_stready, mx_stlast, mx_mvalid, mx_mready, mx_mlast, mx_busy;
  logic [W_DATA-1:0] mx_stdata, mx_mtdata;
  logic [2:0] mx_load_sel;
  mamba2_unit #(.HIDDEN(HIDDEN), .INTER(INTER), .NUM_HEADS(NUM_HEADS),
                .HEAD_DIM(HEAD_DIM), .D_STATE(D_STATE)) u_mixer (
    .clk(clk), .rst_n(rst_n),
    .load_en(load_en && (load_sel[3] == 1'b0)),
    .load_sel(mx_load_sel),
    .load_out_idx(load_out_idx[PW_W-1:0]),
    .load_in_idx(load_in_idx[$clog2((HIDDEN > INTER) ? HIDDEN : INTER)-1:0]),
    .load_wdata(load_wdata),
    .s_axis_tvalid(mx_svalid), .s_axis_tready(mx_stready),
    .s_axis_tdata(mx_stdata), .s_axis_tlast(mx_stlast),
    .m_axis_tvalid(mx_mvalid), .m_axis_tready(mx_mready),
    .m_axis_tdata(mx_mtdata), .m_axis_tlast(mx_mlast),
    .busy(mx_busy)
  );
  assign mx_load_sel = load_sel[2:0];

  // ---------------- residual 1 ----------------
  logic r1_vin; logic [W_DATA-1:0] r1_din, r1_rin; logic r1_vout; logic [W_DATA-1:0] r1_dout;
  residual_adder_unit u_res1 (
    .clk(clk), .rst_n(rst_n),
    .valid_in(r1_vin), .data_in(r1_din), .residual_in(r1_rin),
    .data_out(r1_dout), .valid_out(r1_vout)
  );

  // ---------------- post_attention_layernorm ----------------
  logic n2_win;  logic [W_DATA-1:0] n2_wdata;
  logic n2_vin;  logic [W_DATA-1:0] n2_din;
  logic n2_vout; logic [W_DATA-1:0] n2_dout; logic n2_busy;
  rmsnorm_unit #(.WIDTH(HIDDEN)) u_norm2 (
    .clk(clk), .rst_n(rst_n),
    .valid_in(n2_vin), .data_in(n2_din),
    .weight_valid(n2_win), .weight_in(n2_wdata),
    .data_out(n2_dout), .valid_out(n2_vout), .busy(n2_busy)
  );

  // ---------------- MLP (SwiGLU) ----------------
  logic mlp_svalid, mlp_stready, mlp_stlast, mlp_mvalid, mlp_mready, mlp_mlast, mlp_busy;
  /* verilator lint_on UNUSEDSIGNAL */
  logic [W_DATA-1:0] mlp_stdata, mlp_mtdata;
  mlp_unit #(.HIDDEN(HIDDEN), .INTER(MLP_INTER)) u_mlp (
    .clk(clk), .rst_n(rst_n),
    .load_en(load_en && (load_sel[3:2] == 2'b10)),
    .load_sel(load_sel[0]),
    .load_out_idx(load_out_idx[MLP_LO_W-1:0]),
    .load_in_idx(load_in_idx[MLP_LI_W-1:0]),
    .load_wdata(load_wdata),
    .s_axis_tvalid(mlp_svalid), .s_axis_tready(mlp_stready),
    .s_axis_tdata(mlp_stdata), .s_axis_tlast(mlp_stlast),
    .m_axis_tvalid(mlp_mvalid), .m_axis_tready(mlp_mready),
    .m_axis_tdata(mlp_mtdata), .m_axis_tlast(mlp_mlast),
    .busy(mlp_busy)
  );

  // ---------------- residual 2 ----------------
  logic r2_vin; logic [W_DATA-1:0] r2_din, r2_rin; logic r2_vout; logic [W_DATA-1:0] r2_dout;
  residual_adder_unit u_res2 (
    .clk(clk), .rst_n(rst_n),
    .valid_in(r2_vin), .data_in(r2_din), .residual_in(r2_rin),
    .data_out(r2_dout), .valid_out(r2_vout)
  );

  // ---------------- norm weight routing ----------------
  assign n1_win   = load_en && (load_sel == 4'd8);
  assign n1_wdata = load_wdata;
  assign n2_win   = load_en && (load_sel == 4'd9);
  assign n2_wdata = load_wdata;

  // ---------------- FSM ----------------
  // Feed counters saturate at HIDDEN; capture counters count the unit's
  // valid outputs, so units with latency (residual adder: 3 cycles; norms:
  // buffered) are handled by advancing on the capture count.
  typedef enum logic [3:0] {
    L_IDLE, L_N1F, L_N1C, L_MIX, L_R1F, L_N2F, L_N2C, L_MLP, L_R2F, L_OUT
  } st_t;
  localparam int CW = HW + 1;
  st_t st;
  logic [HW:0] fcnt, ccnt;

  assign s_axis_tready = (st == L_IDLE);
  assign busy = (st != L_IDLE) || n1_busy || n2_busy || mx_busy || mlp_busy;

  // mamba2 / mlp AXI-Stream feeds
  assign mx_svalid  = (st == L_MIX) && (fcnt < CW'(HIDDEN));
  assign mx_stdata  = h1_buf[fcnt[HW-1:0]];
  assign mx_stlast  = (st == L_MIX) && (fcnt == CW'(HIDDEN - 1));
  assign mx_mready  = 1'b1;
  assign mlp_svalid = (st == L_MLP) && (fcnt < CW'(HIDDEN));
  assign mlp_stdata = h2_buf[fcnt[HW-1:0]];
  assign mlp_stlast = (st == L_MLP) && (fcnt == CW'(HIDDEN - 1));
  assign mlp_mready = 1'b1;

  // norm / residual feeds
  assign n1_vin = (st == L_N1F) && (fcnt < CW'(HIDDEN));
  assign n1_din = x_buf[fcnt[HW-1:0]];
  assign n2_vin = (st == L_N2F) && (fcnt < CW'(HIDDEN));
  assign n2_din = r1_buf[fcnt[HW-1:0]];
  assign r1_vin = (st == L_R1F) && (fcnt < CW'(HIDDEN));
  assign r1_din = m_buf[fcnt[HW-1:0]];
  assign r1_rin = x_buf[fcnt[HW-1:0]];
  assign r2_vin = (st == L_R2F) && (fcnt < CW'(HIDDEN));
  assign r2_din = m2_buf[fcnt[HW-1:0]];
  assign r2_rin = r1_buf[fcnt[HW-1:0]];

  assign m_axis_tvalid = (st == L_OUT) && (fcnt < CW'(HIDDEN));
  assign m_axis_tdata  = y_buf[fcnt[HW-1:0]];
  assign m_axis_tlast  = (st == L_OUT) && (fcnt == CW'(HIDDEN - 1));

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st <= L_IDLE; fcnt <= '0; ccnt <= '0;
    end else begin
      case (st)
        L_IDLE: begin
          if (s_axis_tvalid) begin
            x_buf[fcnt[HW-1:0]] <= s_axis_tdata;
            if (s_axis_tlast || (fcnt == CW'(HIDDEN - 1))) begin fcnt <= '0; ccnt <= '0; st <= L_N1F; end
            else fcnt <= fcnt + 1'b1;
          end
        end
        L_N1F: begin
          if (fcnt == CW'(HIDDEN - 1)) begin fcnt <= '0; ccnt <= '0; st <= L_N1C; end
          else fcnt <= fcnt + 1'b1;
        end
        L_N1C: begin
          if (n1_vout) begin
            h1_buf[ccnt[HW-1:0]] <= n1_dout;
            if (ccnt == CW'(HIDDEN - 1)) begin fcnt <= '0; ccnt <= '0; st <= L_MIX; end
            else ccnt <= ccnt + 1'b1;
          end
        end
        L_MIX: begin
          if (mx_svalid && mx_stready) fcnt <= fcnt + 1'b1;
          if (mx_mvalid) begin
            m_buf[ccnt[HW-1:0]] <= mx_mtdata;
            if (ccnt == CW'(HIDDEN - 1)) begin fcnt <= '0; ccnt <= '0; st <= L_R1F; end
            else ccnt <= ccnt + 1'b1;
          end
        end
        L_R1F: begin
          if (r1_vin) fcnt <= fcnt + 1'b1;
          if (r1_vout) begin
            r1_buf[ccnt[HW-1:0]] <= r1_dout;
            if (ccnt == CW'(HIDDEN - 1)) begin fcnt <= '0; ccnt <= '0; st <= L_N2F; end
            else ccnt <= ccnt + 1'b1;
          end
        end
        L_N2F: begin
          if (fcnt == CW'(HIDDEN - 1)) begin fcnt <= '0; ccnt <= '0; st <= L_N2C; end
          else fcnt <= fcnt + 1'b1;
        end
        L_N2C: begin
          if (n2_vout) begin
            h2_buf[ccnt[HW-1:0]] <= n2_dout;
            if (ccnt == CW'(HIDDEN - 1)) begin fcnt <= '0; ccnt <= '0; st <= L_MLP; end
            else ccnt <= ccnt + 1'b1;
          end
        end
        L_MLP: begin
          if (mlp_svalid && mlp_stready) fcnt <= fcnt + 1'b1;
          if (mlp_mvalid) begin
            m2_buf[ccnt[HW-1:0]] <= mlp_mtdata;
            if (ccnt == CW'(HIDDEN - 1)) begin fcnt <= '0; ccnt <= '0; st <= L_R2F; end
            else ccnt <= ccnt + 1'b1;
          end
        end
        L_R2F: begin
          if (r2_vin) fcnt <= fcnt + 1'b1;
          if (r2_vout) begin
            y_buf[ccnt[HW-1:0]] <= r2_dout;
            if (ccnt == CW'(HIDDEN - 1)) begin fcnt <= '0; ccnt <= '0; st <= L_OUT; end
            else ccnt <= ccnt + 1'b1;
          end
        end
        L_OUT: begin
          if (m_axis_tvalid && m_axis_tready) begin
            if (fcnt == CW'(HIDDEN - 1)) begin fcnt <= '0; ccnt <= '0; st <= L_IDLE; end
            else fcnt <= fcnt + 1'b1;
          end
        end
        default: st <= L_IDLE;
      endcase
    end
  end
endmodule
