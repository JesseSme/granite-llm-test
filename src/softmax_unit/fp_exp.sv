// IEEE 754 float32 exponential: y = exp(x)
//
// Two 256-entry LUTs:
//   pos_lut: exp(i/32) for i in [0, 255] — covers x in [0, 8)
//   neg_lut: exp(-i/32) for i in [0, 255] — covers x in (-8, 0]
//
// Address computation from float32 |x|:
//   scaled_exp = x_exp - 122
//   if scaled_exp < 0: addr = 0
//   elif scaled_exp >= 8: addr = 255
//   else: addr = 2^scaled_exp + x_mant >> (23 - scaled_exp)
//
// Special cases: NaN→NaN, +Inf→+Inf, -Inf→0, overflow→+Inf, underflow→0.
// Latency: 1 cycle (registered output).

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */

module fp_exp (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [31:0] x,
  output logic [31:0] y
);

  localparam int EXP_ALL = 8'hFF;

  // ----------------------------------------------------------------
  // Unpack
  // ----------------------------------------------------------------
  logic        sign;
  logic [7:0]  x_exp;
  logic [22:0] x_mant;
  assign sign   = x[31];
  assign x_exp  = x[30:23];
  assign x_mant = x[22:0];

  logic is_nan, is_pos_inf, is_neg_inf, is_zero;
  assign is_nan     = (x_exp == EXP_ALL) && (x_mant != 0);
  assign is_pos_inf = (x_exp == EXP_ALL) && (x_mant == 0) && !sign;
  assign is_neg_inf = (x_exp == EXP_ALL) && (x_mant == 0) && sign;
  assign is_zero    = (x_exp == 0) && (x_mant == 0);

  // ----------------------------------------------------------------
  // LUT address: addr = floor(|x| * 32), clamped to [0, 255]
  //
  // |x| = 2^(e-127) * (1 + m/2^23)
  // |x| * 32 = 2^(e-122) * (1 + m/2^23)
  //
  // For scaled_exp = e - 122 in [0, 7]:
  //   addr = 2^scaled_exp + m >> (23 - scaled_exp)
  //
  // For scaled_exp < 0: addr = 0
  // For scaled_exp >= 8: addr = 255
  // ----------------------------------------------------------------
  logic [7:0]  lut_addr;
  logic signed [8:0] scaled_exp;

  assign scaled_exp = {1'b0, x_exp} - 9'd122;

  always_comb begin
    if (is_zero || x_exp == 0) begin
      // |x| ≈ 0: exp(0) = 1.0 → addr = 0
      lut_addr = 8'd0;
    end else if (scaled_exp < 0) begin
      // |x| < 2^-5 = 0.03125: addr = 0
      lut_addr = 8'd0;
    end else if (scaled_exp >= 8) begin
      // |x| >= 256: clamp to max entry
      lut_addr = 8'd255;
    end else begin
      // scaled_exp in [0, 7]: addr = 2^scaled_exp + top bits of mantissa
      // 2^scaled_exp: 1, 2, 4, 8, 16, 32, 64, 128
      // m >> (23 - scaled_exp): top scaled_exp bits of mantissa
      case (scaled_exp[2:0])
        3'd0: lut_addr = 8'd1;                                        // 2^0=1, no mantissa bits
        3'd1: lut_addr = 8'd2   + {7'd0, x_mant[22]};                // 2^1=2, m>>22
        3'd2: lut_addr = 8'd4   + {6'd0, x_mant[22:21]};             // 2^2=4, m>>21
        3'd3: lut_addr = 8'd8   + {5'd0, x_mant[22:20]};             // 2^3=8, m>>20
        3'd4: lut_addr = 8'd16  + {4'd0, x_mant[22:19]};             // 2^4=16, m>>19
        3'd5: lut_addr = 8'd32  + {3'd0, x_mant[22:18]};             // 2^5=32, m>>18
        3'd6: lut_addr = 8'd64  + {2'd0, x_mant[22:17]};             // 2^6=64, m>>17
        3'd7: lut_addr = 8'd128 + {1'd0, x_mant[22:16]};             // 2^7=128, m>>16
        default: lut_addr = 8'd0;
      endcase
    end
  end

  // ----------------------------------------------------------------
  // pos_lut: exp(i/32) for i in [0, 255]
  // ----------------------------------------------------------------
  logic [31:0] pos_lut_val;
  always_comb begin
    unique case (lut_addr)
      8'h00: pos_lut_val = 32'h3F800000;
      8'h01: pos_lut_val = 32'h3F84102B;
      8'h02: pos_lut_val = 32'h3F88415B;
      8'h03: pos_lut_val = 32'h3F8C949C;
      8'h04: pos_lut_val = 32'h3F910B02;
      8'h05: pos_lut_val = 32'h3F95A5AC;
      8'h06: pos_lut_val = 32'h3F9A65C1;
      8'h07: pos_lut_val = 32'h3F9F4C6F;
      8'h08: pos_lut_val = 32'h3FA45AF2;
      8'h09: pos_lut_val = 32'h3FA9928C;
      8'h0A: pos_lut_val = 32'h3FAEF48C;
      8'h0B: pos_lut_val = 32'h3FB48249;
      8'h0C: pos_lut_val = 32'h3FBA3D29;
      8'h0D: pos_lut_val = 32'h3FC02698;
      8'h0E: pos_lut_val = 32'h3FC64012;
      8'h0F: pos_lut_val = 32'h3FCC8B1D;
      8'h10: pos_lut_val = 32'h3FD3094C;
      8'h11: pos_lut_val = 32'h3FD9BC3F;
      8'h12: pos_lut_val = 32'h3FE0A5A2;
      8'h13: pos_lut_val = 32'h3FE7C72F;
      8'h14: pos_lut_val = 32'h3FEF22AF;
      8'h15: pos_lut_val = 32'h3FF6B9F9;
      8'h16: pos_lut_val = 32'h3FFE8EF3;
      8'h17: pos_lut_val = 32'h400351C9;
      8'h18: pos_lut_val = 32'h40077CEE;
      8'h19: pos_lut_val = 32'h400BC9F2;
      8'h1A: pos_lut_val = 32'h401039EA;
      8'h1B: pos_lut_val = 32'h4014CDF1;
      8'h1C: pos_lut_val = 32'h4019872C;
      8'h1D: pos_lut_val = 32'h401E66CA;
      8'h1E: pos_lut_val = 32'h40236E02;
      8'h1F: pos_lut_val = 32'h40289E17;
      8'h20: pos_lut_val = 32'h402DF854;
      8'h21: pos_lut_val = 32'h40337E10;
      8'h22: pos_lut_val = 32'h403930AD;
      8'h23: pos_lut_val = 32'h403F1197;
      8'h24: pos_lut_val = 32'h40452246;
      8'h25: pos_lut_val = 32'h404B643F;
      8'h26: pos_lut_val = 32'h4051D911;
      8'h27: pos_lut_val = 32'h4058825C;
      8'h28: pos_lut_val = 32'h405F61C7;
      8'h29: pos_lut_val = 32'h4066790D;
      8'h2A: pos_lut_val = 32'h406DC9F2;
      8'h2B: pos_lut_val = 32'h4075564B;
      8'h2C: pos_lut_val = 32'h407D1FFA;
      8'h2D: pos_lut_val = 32'h40829479;
      8'h2E: pos_lut_val = 32'h4086B99C;
      8'h2F: pos_lut_val = 32'h408B006D;
      8'h30: pos_lut_val = 32'h408F69FF;
      8'h31: pos_lut_val = 32'h4093F76D;
      8'h32: pos_lut_val = 32'h4098A9D9;
      8'h33: pos_lut_val = 32'h409D8270;
      8'h34: pos_lut_val = 32'h40A28269;
      8'h35: pos_lut_val = 32'h40A7AB03;
      8'h36: pos_lut_val = 32'h40ACFD89;
      8'h37: pos_lut_val = 32'h40B27B4F;
      8'h38: pos_lut_val = 32'h40B825B5;
      8'h39: pos_lut_val = 32'h40BDFE25;
      8'h3A: pos_lut_val = 32'h40C40615;
      8'h3B: pos_lut_val = 32'h40CA3F09;
      8'h3C: pos_lut_val = 32'h40D0AA8D;
      8'h3D: pos_lut_val = 32'h40D74A3D;
      8'h3E: pos_lut_val = 32'h40DE1FC0;
      8'h3F: pos_lut_val = 32'h40E52CCD;
      8'h40: pos_lut_val = 32'h40EC7326;
      8'h41: pos_lut_val = 32'h40F3F49D;
      8'h42: pos_lut_val = 32'h40FBB312;
      8'h43: pos_lut_val = 32'h4101D83B;
      8'h44: pos_lut_val = 32'h4105F763;
      8'h45: pos_lut_val = 32'h410A380A;
      8'h46: pos_lut_val = 32'h410E9B40;
      8'h47: pos_lut_val = 32'h4113221D;
      8'h48: pos_lut_val = 32'h4117CDC4;
      8'h49: pos_lut_val = 32'h411C9F5F;
      8'h4A: pos_lut_val = 32'h41219822;
      8'h4B: pos_lut_val = 32'h4126B94D;
      8'h4C: pos_lut_val = 32'h412C0426;
      8'h4D: pos_lut_val = 32'h41317A02;
      8'h4E: pos_lut_val = 32'h41371C3D;
      8'h4F: pos_lut_val = 32'h413CEC40;
      8'h50: pos_lut_val = 32'h4142EB7F;
      8'h51: pos_lut_val = 32'h41491B7A;
      8'h52: pos_lut_val = 32'h414F7DBC;
      8'h53: pos_lut_val = 32'h415613E0;
      8'h54: pos_lut_val = 32'h415CDF89;
      8'h55: pos_lut_val = 32'h4163E26C;
      8'h56: pos_lut_val = 32'h416B1E48;
      8'h57: pos_lut_val = 32'h417294ED;
      8'h58: pos_lut_val = 32'h417A4838;
      8'h59: pos_lut_val = 32'h41811D0C;
      8'h5A: pos_lut_val = 32'h41853643;
      8'h5B: pos_lut_val = 32'h418970C9;
      8'h5C: pos_lut_val = 32'h418DCDAB;
      8'h5D: pos_lut_val = 32'h41924E02;
      8'h5E: pos_lut_val = 32'h4196F2ED;
      8'h5F: pos_lut_val = 32'h419BBD95;
      8'h60: pos_lut_val = 32'h41A0AF2E;
      8'h61: pos_lut_val = 32'h41A5C8F3;
      8'h62: pos_lut_val = 32'h41AB0C2C;
      8'h63: pos_lut_val = 32'h41B07A28;
      8'h64: pos_lut_val = 32'h41B61444;
      8'h65: pos_lut_val = 32'h41BBDBE6;
      8'h66: pos_lut_val = 32'h41C1D27F;
      8'h67: pos_lut_val = 32'h41C7F98F;
      8'h68: pos_lut_val = 32'h41CE529E;
      8'h69: pos_lut_val = 32'h41D4DF42;
      8'h6A: pos_lut_val = 32'h41DBA120;
      8'h6B: pos_lut_val = 32'h41E299E7;
      8'h6C: pos_lut_val = 32'h41E9CB55;
      8'h6D: pos_lut_val = 32'h41F13738;
      8'h6E: pos_lut_val = 32'h41F8DF6A;
      8'h6F: pos_lut_val = 32'h420062EA;
      8'h70: pos_lut_val = 32'h42047639;
      8'h71: pos_lut_val = 32'h4208AAA6;
      8'h72: pos_lut_val = 32'h420D013F;
      8'h73: pos_lut_val = 32'h42117B18;
      8'h74: pos_lut_val = 32'h42161951;
      8'h75: pos_lut_val = 32'h421ADD11;
      8'h76: pos_lut_val = 32'h421FC789;
      8'h77: pos_lut_val = 32'h4224D9F4;
      8'h78: pos_lut_val = 32'h422A1597;
      8'h79: pos_lut_val = 32'h422F7BBF;
      8'h7A: pos_lut_val = 32'h42350DC7;
      8'h7B: pos_lut_val = 32'h423ACD14;
      8'h7C: pos_lut_val = 32'h4240BB15;
      8'h7D: pos_lut_val = 32'h4246D946;
      8'h7E: pos_lut_val = 32'h424D292E;
      8'h7F: pos_lut_val = 32'h4253AC62;
      8'h80: pos_lut_val = 32'h425A6481;
      8'h81: pos_lut_val = 32'h4261533B;
      8'h82: pos_lut_val = 32'h42687A4B;
      8'h83: pos_lut_val = 32'h426FDB7B;
      8'h84: pos_lut_val = 32'h427778A3;
      8'h85: pos_lut_val = 32'h427F53AA;
      8'h86: pos_lut_val = 32'h4283B744;
      8'h87: pos_lut_val = 32'h4287E5A1;
      8'h88: pos_lut_val = 32'h428C35F9;
      8'h89: pos_lut_val = 32'h4290A95E;
      8'h8A: pos_lut_val = 32'h429540EF;
      8'h8B: pos_lut_val = 32'h4299FDD1;
      8'h8C: pos_lut_val = 32'h429EE133;
      8'h8D: pos_lut_val = 32'h42A3EC4E;
      8'h8E: pos_lut_val = 32'h42A92065;
      8'h8F: pos_lut_val = 32'h42AE7EC5;
      8'h90: pos_lut_val = 32'h42B408C5;
      8'h91: pos_lut_val = 32'h42B9BFC9;
      8'h92: pos_lut_val = 32'h42BFA53E;
      8'h93: pos_lut_val = 32'h42C5BA9D;
      8'h94: pos_lut_val = 32'h42CC016B;
      8'h95: pos_lut_val = 32'h42D27B3C;
      8'h96: pos_lut_val = 32'h42D929AC;
      8'h97: pos_lut_val = 32'h42E00E67;
      8'h98: pos_lut_val = 32'h42E72B27;
      8'h99: pos_lut_val = 32'h42EE81B4;
      8'h9A: pos_lut_val = 32'h42F613E2;
      8'h9B: pos_lut_val = 32'h42FDE396;
      8'h9C: pos_lut_val = 32'h4302F962;
      8'h9D: pos_lut_val = 32'h430721B8;
      8'h9E: pos_lut_val = 32'h430B6BD8;
      8'h9F: pos_lut_val = 32'h430FD8D3;
      8'hA0: pos_lut_val = 32'h431469C5;
      8'hA1: pos_lut_val = 32'h43191FD2;
      8'hA2: pos_lut_val = 32'h431DFC28;
      8'hA3: pos_lut_val = 32'h4322FFFE;
      8'hA4: pos_lut_val = 32'h43282C95;
      8'hA5: pos_lut_val = 32'h432D8337;
      8'hA6: pos_lut_val = 32'h4333053C;
      8'hA7: pos_lut_val = 32'h4338B402;
      8'hA8: pos_lut_val = 32'h433E90F7;
      8'hA9: pos_lut_val = 32'h43449D91;
      8'hAA: pos_lut_val = 32'h434ADB53;
      8'hAB: pos_lut_val = 32'h43514BCD;
      8'hAC: pos_lut_val = 32'h4357F09B;
      8'hAD: pos_lut_val = 32'h435ECB67;
      8'hAE: pos_lut_val = 32'h4365DDE6;
      8'hAF: pos_lut_val = 32'h436D29DF;
      8'hB0: pos_lut_val = 32'h4374B122;
      8'hB1: pos_lut_val = 32'h437C7594;
      8'hB2: pos_lut_val = 32'h43823C92;
      8'hB3: pos_lut_val = 32'h43865EEA;
      8'hB4: pos_lut_val = 32'h438AA2DA;
      8'hB5: pos_lut_val = 32'h438F0974;
      8'hB6: pos_lut_val = 32'h439393D1;
      8'hB7: pos_lut_val = 32'h43984313;
      8'hB8: pos_lut_val = 32'h439D1868;
      8'hB9: pos_lut_val = 32'h43A21503;
      8'hBA: pos_lut_val = 32'h43A73A24;
      8'hBB: pos_lut_val = 32'h43AC8914;
      8'hBC: pos_lut_val = 32'h43B20328;
      8'hBD: pos_lut_val = 32'h43B7A9BE;
      8'hBE: pos_lut_val = 32'h43BD7E3E;
      8'hBF: pos_lut_val = 32'h43C38220;
      8'hC0: pos_lut_val = 32'h43C9B6E3;
      8'hC1: pos_lut_val = 32'h43D01E14;
      8'hC2: pos_lut_val = 32'h43D6B94F;
      8'hC3: pos_lut_val = 32'h43DD8A38;
      8'hC4: pos_lut_val = 32'h43E49286;
      8'hC5: pos_lut_val = 32'h43EBD3F9;
      8'hC6: pos_lut_val = 32'h43F35063;
      8'hC7: pos_lut_val = 32'h43FB09A2;
      8'hC8: pos_lut_val = 32'h440180D2;
      8'hC9: pos_lut_val = 32'h44059D34;
      8'hCA: pos_lut_val = 32'h4409DAFE;
      8'hCB: pos_lut_val = 32'h440E3B40;
      8'hCC: pos_lut_val = 32'h4412BF11;
      8'hCD: pos_lut_val = 32'h44176793;
      8'hCE: pos_lut_val = 32'h441C35EF;
      8'hCF: pos_lut_val = 32'h44212B5A;
      8'hD0: pos_lut_val = 32'h44264911;
      8'hD1: pos_lut_val = 32'h442B905A;
      8'hD2: pos_lut_val = 32'h44310289;
      8'hD3: pos_lut_val = 32'h4436A0F9;
      8'hD4: pos_lut_val = 32'h443C6D12;
      8'hD5: pos_lut_val = 32'h44426847;
      8'hD6: pos_lut_val = 32'h44489418;
      8'hD7: pos_lut_val = 32'h444EF20F;
      8'hD8: pos_lut_val = 32'h445583C3;
      8'hD9: pos_lut_val = 32'h445C4AD9;
      8'hDA: pos_lut_val = 32'h44634903;
      8'hDB: pos_lut_val = 32'h446A8001;
      8'hDC: pos_lut_val = 32'h4471F1A0;
      8'hDD: pos_lut_val = 32'h44799FBC;
      8'hDE: pos_lut_val = 32'h4480C621;
      8'hDF: pos_lut_val = 32'h4484DC96;
      8'hE0: pos_lut_val = 32'h44891443;
      8'hE1: pos_lut_val = 32'h448D6E36;
      8'hE2: pos_lut_val = 32'h4491EB84;
      8'hE3: pos_lut_val = 32'h44968D4F;
      8'hE4: pos_lut_val = 32'h449B54BE;
      8'hE5: pos_lut_val = 32'h44A04302;
      8'hE6: pos_lut_val = 32'h44A55959;
      8'hE7: pos_lut_val = 32'h44AA9906;
      8'hE8: pos_lut_val = 32'h44B0035B;
      8'hE9: pos_lut_val = 32'h44B599B1;
      8'hEA: pos_lut_val = 32'h44BB5D6F;
      8'hEB: pos_lut_val = 32'h44C15005;
      8'hEC: pos_lut_val = 32'h44C772F0;
      8'hED: pos_lut_val = 32'h44CDC7B9;
      8'hEE: pos_lut_val = 32'h44D44FF5;
      8'hEF: pos_lut_val = 32'h44DB0D46;
      8'hF0: pos_lut_val = 32'h44E2015B;
      8'hF1: pos_lut_val = 32'h44E92DF2;
      8'hF2: pos_lut_val = 32'h44F094D6;
      8'hF3: pos_lut_val = 32'h44F837E0;
      8'hF4: pos_lut_val = 32'h45000C7D;
      8'hF5: pos_lut_val = 32'h45041D0D;
      8'hF6: pos_lut_val = 32'h45084EA6;
      8'hF7: pos_lut_val = 32'h450CA252;
      8'hF8: pos_lut_val = 32'h45111929;
      8'hF9: pos_lut_val = 32'h4515B446;
      8'hFA: pos_lut_val = 32'h451A74D1;
      8'hFB: pos_lut_val = 32'h451F5BFA;
      8'hFC: pos_lut_val = 32'h45246AFB;
      8'hFD: pos_lut_val = 32'h4529A317;
      8'hFE: pos_lut_val = 32'h452F059D;
      8'hFF: pos_lut_val = 32'h453493E6;
      default: pos_lut_val = 32'h7F800000;
    endcase
  end

  // ----------------------------------------------------------------
  // neg_lut: exp(-i/32) for i in [0, 255]
  // ----------------------------------------------------------------
  logic [31:0] neg_lut_val;
  always_comb begin
    unique case (lut_addr)
      8'h00: neg_lut_val = 32'h3F800000;
      8'h01: neg_lut_val = 32'h3F781FAB;
      8'h02: neg_lut_val = 32'h3F707D60;
      8'h03: neg_lut_val = 32'h3F691735;
      8'h04: neg_lut_val = 32'h3F61EB51;
      8'h05: neg_lut_val = 32'h3F5AF7E9;
      8'h06: neg_lut_val = 32'h3F543B41;
      8'h07: neg_lut_val = 32'h3F4DB3A8;
      8'h08: neg_lut_val = 32'h3F475F7D;
      8'h09: neg_lut_val = 32'h3F413D2B;
      8'h0A: neg_lut_val = 32'h3F3B4B29;
      8'h0B: neg_lut_val = 32'h3F3587FC;
      8'h0C: neg_lut_val = 32'h3F2FF231;
      8'h0D: neg_lut_val = 32'h3F2A8863;
      8'h0E: neg_lut_val = 32'h3F254939;
      8'h0F: neg_lut_val = 32'h3F203361;
      8'h10: neg_lut_val = 32'h3F1B4598;
      8'h11: neg_lut_val = 32'h3F167EA0;
      8'h12: neg_lut_val = 32'h3F11DD4A;
      8'h13: neg_lut_val = 32'h3F0D606B;
      8'h14: neg_lut_val = 32'h3F0906E5;
      8'h15: neg_lut_val = 32'h3F04CFA1;
      8'h16: neg_lut_val = 32'h3F00B992;
      8'h17: neg_lut_val = 32'h3EF98764;
      8'h18: neg_lut_val = 32'h3EF1DA07;
      8'h19: neg_lut_val = 32'h3EEA6922;
      8'h1A: neg_lut_val = 32'h3EE332D9;
      8'h1B: neg_lut_val = 32'h3EDC355D;
      8'h1C: neg_lut_val = 32'h3ED56EF0;
      8'h1D: neg_lut_val = 32'h3ECEDDE0;
      8'h1E: neg_lut_val = 32'h3EC88088;
      8'h1F: neg_lut_val = 32'h3EC25552;
      8'h20: neg_lut_val = 32'h3EBC5AB2;
      8'h21: neg_lut_val = 32'h3EB68F29;
      8'h22: neg_lut_val = 32'h3EB0F145;
      8'h23: neg_lut_val = 32'h3EAB7F9F;
      8'h24: neg_lut_val = 32'h3EA638D9;
      8'h25: neg_lut_val = 32'h3EA11BA2;
      8'h26: neg_lut_val = 32'h3E9C26B4;
      8'h27: neg_lut_val = 32'h3E9758CF;
      8'h28: neg_lut_val = 32'h3E92B0C2;
      8'h29: neg_lut_val = 32'h3E8E2D61;
      8'h2A: neg_lut_val = 32'h3E89CD8D;
      8'h2B: neg_lut_val = 32'h3E85902D;
      8'h2C: neg_lut_val = 32'h3E817431;
      8'h2D: neg_lut_val = 32'h3E7AF126;
      8'h2E: neg_lut_val = 32'h3E7338A8;
      8'h2F: neg_lut_val = 32'h3E6BBCFA;
      8'h30: neg_lut_val = 32'h3E647C3C;
      8'h31: neg_lut_val = 32'h3E5D749E;
      8'h32: neg_lut_val = 32'h3E56A45E;
      8'h33: neg_lut_val = 32'h3E5009C9;
      8'h34: neg_lut_val = 32'h3E49A337;
      8'h35: neg_lut_val = 32'h3E436F0F;
      8'h36: neg_lut_val = 32'h3E3D6BC4;
      8'h37: neg_lut_val = 32'h3E3797D4;
      8'h38: neg_lut_val = 32'h3E31F1CC;
      8'h39: neg_lut_val = 32'h3E2C7841;
      8'h3A: neg_lut_val = 32'h3E2729D5;
      8'h3B: neg_lut_val = 32'h3E220534;
      8'h3C: neg_lut_val = 32'h3E1D0916;
      8'h3D: neg_lut_val = 32'h3E18343A;
      8'h3E: neg_lut_val = 32'h3E13856D;
      8'h3F: neg_lut_val = 32'h3E0EFB81;
      8'h40: neg_lut_val = 32'h3E0A9555;
      8'h41: neg_lut_val = 32'h3E0651CF;
      8'h42: neg_lut_val = 32'h3E022FDF;
      8'h43: neg_lut_val = 32'h3DFC5CF5;
      8'h44: neg_lut_val = 32'h3DF49946;
      8'h45: neg_lut_val = 32'h3DED12BE;
      8'h46: neg_lut_val = 32'h3DE5C77C;
      8'h47: neg_lut_val = 32'h3DDEB5AD;
      8'h48: neg_lut_val = 32'h3DD7DB8C;
      8'h49: neg_lut_val = 32'h3DD13764;
      8'h4A: neg_lut_val = 32'h3DCAC78B;
      8'h4B: neg_lut_val = 32'h3DC48A64;
      8'h4C: neg_lut_val = 32'h3DBE7E61;
      8'h4D: neg_lut_val = 32'h3DB8A1FF;
      8'h4E: neg_lut_val = 32'h3DB2F3C6;
      8'h4F: neg_lut_val = 32'h3DAD724B;
      8'h50: neg_lut_val = 32'h3DA81C2E;
      8'h51: neg_lut_val = 32'h3DA2F019;
      8'h52: neg_lut_val = 32'h3D9DECC0;
      8'h53: neg_lut_val = 32'h3D9910E3;
      8'h54: neg_lut_val = 32'h3D945B4C;
      8'h55: neg_lut_val = 32'h3D8FCACC;
      8'h56: neg_lut_val = 32'h3D8B5E3F;
      8'h57: neg_lut_val = 32'h3D87148B;
      8'h58: neg_lut_val = 32'h3D82EC9C;
      8'h59: neg_lut_val = 32'h3D7DCAD3;
      8'h5A: neg_lut_val = 32'h3D75FBE2;
      8'h5B: neg_lut_val = 32'h3D6E6A71;
      8'h5C: neg_lut_val = 32'h3D67149C;
      8'h5D: neg_lut_val = 32'h3D5FF88D;
      8'h5E: neg_lut_val = 32'h3D59147E;
      8'h5F: neg_lut_val = 32'h3D5266B5;
      8'h60: neg_lut_val = 32'h3D4BED86;
      8'h61: neg_lut_val = 32'h3D45A754;
      8'h62: neg_lut_val = 32'h3D3F928D;
      8'h63: neg_lut_val = 32'h3D39ADAC;
      8'h64: neg_lut_val = 32'h3D33F737;
      8'h65: neg_lut_val = 32'h3D2E6DC0;
      8'h66: neg_lut_val = 32'h3D290FE6;
      8'h67: neg_lut_val = 32'h3D23DC51;
      8'h68: neg_lut_val = 32'h3D1ED1B4;
      8'h69: neg_lut_val = 32'h3D19EECC;
      8'h6A: neg_lut_val = 32'h3D153261;
      8'h6B: neg_lut_val = 32'h3D109B43;
      8'h6C: neg_lut_val = 32'h3D0C284C;
      8'h6D: neg_lut_val = 32'h3D07D860;
      8'h6E: neg_lut_val = 32'h3D03AA6C;
      8'h6F: neg_lut_val = 32'h3CFF3AC4;
      8'h70: neg_lut_val = 32'h3CF76081;
      8'h71: neg_lut_val = 32'h3CEFC417;
      8'h72: neg_lut_val = 32'h3CE8639F;
      8'h73: neg_lut_val = 32'h3CE13D42;
      8'h74: neg_lut_val = 32'h3CDA4F35;
      8'h75: neg_lut_val = 32'h3CD397BD;
      8'h76: neg_lut_val = 32'h3CCD152C;
      8'h77: neg_lut_val = 32'h3CC6C5E2;
      8'h78: neg_lut_val = 32'h3CC0A84A;
      8'h79: neg_lut_val = 32'h3CBABADD;
      8'h7A: neg_lut_val = 32'h3CB4FC1F;
      8'h7B: neg_lut_val = 32'h3CAF6AA2;
      8'h7C: neg_lut_val = 32'h3CAA0500;
      8'h7D: neg_lut_val = 32'h3CA4C9E1;
      8'h7E: neg_lut_val = 32'h3C9FB7F4;
      8'h7F: neg_lut_val = 32'h3C9ACDF7;
      8'h80: neg_lut_val = 32'h3C960AAE;
      8'h81: neg_lut_val = 32'h3C916CE8;
      8'h82: neg_lut_val = 32'h3C8CF37E;
      8'h83: neg_lut_val = 32'h3C889D52;
      8'h84: neg_lut_val = 32'h3C84694E;
      8'h85: neg_lut_val = 32'h3C805665;
      8'h86: neg_lut_val = 32'h3C78C724;
      8'h87: neg_lut_val = 32'h3C711FB2;
      8'h88: neg_lut_val = 32'h3C69B489;
      8'h89: neg_lut_val = 32'h3C6283CE;
      8'h8A: neg_lut_val = 32'h3C5B8BB5;
      8'h8B: neg_lut_val = 32'h3C54CA80;
      8'h8C: neg_lut_val = 32'h3C4E3E7F;
      8'h8D: neg_lut_val = 32'h3C47E60E;
      8'h8E: neg_lut_val = 32'h3C41BF99;
      8'h8F: neg_lut_val = 32'h3C3BC994;
      8'h90: neg_lut_val = 32'h3C360282;
      8'h91: neg_lut_val = 32'h3C3068F2;
      8'h92: neg_lut_val = 32'h3C2AFB7D;
      8'h93: neg_lut_val = 32'h3C25B8C8;
      8'h94: neg_lut_val = 32'h3C209F82;
      8'h95: neg_lut_val = 32'h3C1BAE65;
      8'h96: neg_lut_val = 32'h3C16E434;
      8'h97: neg_lut_val = 32'h3C123FBD;
      8'h98: neg_lut_val = 32'h3C0DBFD7;
      8'h99: neg_lut_val = 32'h3C096361;
      8'h9A: neg_lut_val = 32'h3C052945;
      8'h9B: neg_lut_val = 32'h3C011074;
      8'h9C: neg_lut_val = 32'h3BFA2FD0;
      8'h9D: neg_lut_val = 32'h3BF27D45;
      8'h9E: neg_lut_val = 32'h3BEB075A;
      8'h9F: neg_lut_val = 32'h3BE3CC32;
      8'hA0: neg_lut_val = 32'h3BDCC9FF;
      8'hA1: neg_lut_val = 32'h3BD5FEFF;
      8'hA2: neg_lut_val = 32'h3BCF6980;
      8'hA3: neg_lut_val = 32'h3BC907DD;
      8'hA4: neg_lut_val = 32'h3BC2D87D;
      8'hA5: neg_lut_val = 32'h3BBCD9D3;
      8'hA6: neg_lut_val = 32'h3BB70A61;
      8'hA7: neg_lut_val = 32'h3BB168B3;
      8'hA8: neg_lut_val = 32'h3BABF360;
      8'hA9: neg_lut_val = 32'h3BA6A90B;
      8'hAA: neg_lut_val = 32'h3BA18860;
      8'hAB: neg_lut_val = 32'h3B9C9019;
      8'hAC: neg_lut_val = 32'h3B97BEF6;
      8'hAD: neg_lut_val = 32'h3B9313C4;
      8'hAE: neg_lut_val = 32'h3B8E8D58;
      8'hAF: neg_lut_val = 32'h3B8A2A90;
      8'hB0: neg_lut_val = 32'h3B85EA53;
      8'hB1: neg_lut_val = 32'h3B81CB91;
      8'hB2: neg_lut_val = 32'h3B7B9A86;
      8'hB3: neg_lut_val = 32'h3B73DCD2;
      8'hB4: neg_lut_val = 32'h3B6C5C17;
      8'hB5: neg_lut_val = 32'h3B651673;
      8'hB6: neg_lut_val = 32'h3B5E0A17;
      8'hB7: neg_lut_val = 32'h3B57353E;
      8'hB8: neg_lut_val = 32'h3B509633;
      8'hB9: neg_lut_val = 32'h3B4A2B50;
      8'hBA: neg_lut_val = 32'h3B43F2F8;
      8'hBB: neg_lut_val = 32'h3B3DEB9D;
      8'hBC: neg_lut_val = 32'h3B3813BF;
      8'hBD: neg_lut_val = 32'h3B3269E7;
      8'hBE: neg_lut_val = 32'h3B2CECAA;
      8'hBF: neg_lut_val = 32'h3B279AA9;
      8'hC0: neg_lut_val = 32'h3B227290;
      8'hC1: neg_lut_val = 32'h3B1D7314;
      8'hC2: neg_lut_val = 32'h3B189AF5;
      8'hC3: neg_lut_val = 32'h3B13E8FF;
      8'hC4: neg_lut_val = 32'h3B0F5C03;
      8'hC5: neg_lut_val = 32'h3B0AF2DF;
      8'hC6: neg_lut_val = 32'h3B06AC78;
      8'hC7: neg_lut_val = 32'h3B0287BE;
      8'hC8: neg_lut_val = 32'h3AFD074B;
      8'hC9: neg_lut_val = 32'h3AF53E5E;
      8'hCA: neg_lut_val = 32'h3AEDB2C1;
      8'hCB: neg_lut_val = 32'h3AE66293;
      8'hCC: neg_lut_val = 32'h3ADF4BFF;
      8'hCD: neg_lut_val = 32'h3AD86D3E;
      8'hCE: neg_lut_val = 32'h3AD1C49A;
      8'hCF: neg_lut_val = 32'h3ACB5069;
      8'hD0: neg_lut_val = 32'h3AC50F0C;
      8'hD1: neg_lut_val = 32'h3ABEFEF5;
      8'hD2: neg_lut_val = 32'h3AB91E9E;
      8'hD3: neg_lut_val = 32'h3AB36C8F;
      8'hD4: neg_lut_val = 32'h3AADE75D;
      8'hD5: neg_lut_val = 32'h3AA88DA6;
      8'hD6: neg_lut_val = 32'h3AA35E12;
      8'hD7: neg_lut_val = 32'h3A9E5758;
      8'hD8: neg_lut_val = 32'h3A997833;
      8'hD9: neg_lut_val = 32'h3A94BF6E;
      8'hDA: neg_lut_val = 32'h3A902BD9;
      8'hDB: neg_lut_val = 32'h3A8BBC50;
      8'hDC: neg_lut_val = 32'h3A876FB7;
      8'hDD: neg_lut_val = 32'h3A8344FB;
      8'hDE: neg_lut_val = 32'h3A7E7620;
      8'hDF: neg_lut_val = 32'h3A76A1EA;
      8'hE0: neg_lut_val = 32'h3A6F0B5D;
      8'hE1: neg_lut_val = 32'h3A67B094;
      8'hE2: neg_lut_val = 32'h3A608FB9;
      8'hE3: neg_lut_val = 32'h3A59A703;
      8'hE4: neg_lut_val = 32'h3A52F4B8;
      8'hE5: neg_lut_val = 32'h3A4C772B;
      8'hE6: neg_lut_val = 32'h3A462CBD;
      8'hE7: neg_lut_val = 32'h3A4013DB;
      8'hE8: neg_lut_val = 32'h3A3A2AFF;
      8'hE9: neg_lut_val = 32'h3A3470AF;
      8'hEA: neg_lut_val = 32'h3A2EE37C;
      8'hEB: neg_lut_val = 32'h3A298203;
      8'hEC: neg_lut_val = 32'h3A244AEB;
      8'hED: neg_lut_val = 32'h3A1F3CE6;
      8'hEE: neg_lut_val = 32'h3A1A56B2;
      8'hEF: neg_lut_val = 32'h3A159714;
      8'hF0: neg_lut_val = 32'h3A10FCDD;
      8'hF1: neg_lut_val = 32'h3A0C86E6;
      8'hF2: neg_lut_val = 32'h3A083411;
      8'hF3: neg_lut_val = 32'h3A04034A;
      8'hF4: neg_lut_val = 32'h39FFE709;
      8'hF5: neg_lut_val = 32'h39F80779;
      8'hF6: neg_lut_val = 32'h39F065EC;
      8'hF7: neg_lut_val = 32'h39E9007A;
      8'hF8: neg_lut_val = 32'h39E1D549;
      8'hF9: neg_lut_val = 32'h39DAE28F;
      8'hFA: neg_lut_val = 32'h39D4268E;
      8'hFB: neg_lut_val = 32'h39CD9F98;
      8'hFC: neg_lut_val = 32'h39C74C0C;
      8'hFD: neg_lut_val = 32'h39C12A53;
      8'hFE: neg_lut_val = 32'h39BB38E6;
      8'hFF: neg_lut_val = 32'h39B57648;
      default: neg_lut_val = 32'h00000000;
    endcase
  end

  // ----------------------------------------------------------------
  // Select LUT value and handle special cases
  // ----------------------------------------------------------------
  logic [31:0] lut_result;
  assign lut_result = sign ? neg_lut_val : pos_lut_val;

  logic [31:0] result;
  always_comb begin
    if (is_nan) begin
      result = x;
    end else if (is_pos_inf) begin
      result = 32'h7F800000;
    end else if (is_neg_inf) begin
      result = 32'h00000000;
    end else if (!sign && x_exp >= 8'd135) begin
      result = 32'h7F800000;
    end else if (sign && x_exp >= 8'd135) begin
      result = 32'h00000000;
    end else begin
      result = lut_result;
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      y <= '0;
    end else begin
      y <= result;
    end
  end

endmodule
