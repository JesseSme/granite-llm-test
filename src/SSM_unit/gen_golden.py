"""Generate a golden sample for the ssm_unit cocotb test.

Emulates the RTL datapath in float32:
  z    = bf16(dt + dt_bias)
  dtp  = bf16(softplus(z))                  (matches torch bf16 softplus)
  A_h  = -exp(A_log_h)
  dA_h = exp(A_h * dtp_h)                   (fp32; HW exp ~2e-5 relative error)
  w_s  = dtp_h * B_s
  h    = dA_h * h + w_s * x_hd              (fp32, sequential in s)
  y_hd = D_h * x_hd + sum_s h * C_s         (fp32)

The hardware exp/softplus are polynomial approximations (~1e-5 relative
error), so the unit test uses a tolerance rather than bit-exact comparison.
"""

import argparse
import struct
from pathlib import Path

import torch
import torch.nn.functional as F


def bf16_hex(t):
    return f"{t.view(torch.uint16).item():04x}"


def f32_hex(v):
    return f"{struct.unpack('I', struct.pack('f', float(v)))[0]:08x}"


def reference(A_log, D, dt_bias, xs, Bs, Cs, dts):
    NH = A_log.shape[0]
    HD = xs.shape[1] // NH
    DS = Bs.shape[1]
    seq = xs.shape[0]

    h = torch.zeros(NH, HD, DS, dtype=torch.float32)
    outs = []
    for t in range(seq):
        z = (dts[t] + dt_bias).to(torch.bfloat16)
        dtp = F.softplus(z).float()
        A = -torch.exp(A_log.float())
        dA = torch.exp(A * dtp)
        xv = xs[t].view(NH, HD).float()
        Bv = Bs[t].float()
        Cv = Cs[t].float()
        w = dtp.unsqueeze(1) * Bv.unsqueeze(0)  # (NH, DS)
        for hh in range(NH):
            for dd in range(HD):
                acc = D[hh].float() * xv[hh, dd]
                for ss in range(DS):
                    newh = dA[hh] * h[hh, dd, ss] + w[hh, ss] * xv[hh, dd]
                    h[hh, dd, ss] = newh
                    acc = acc + newh * Cv[ss]
                outs.append(acc)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=2)
    ap.add_argument("--d-state", type=int, default=4)
    ap.add_argument("--seq", type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(__file__).parent
    torch.manual_seed(42)

    NH, HD, DS = args.heads, args.head_dim, args.d_state
    seq = args.seq

    A_log = torch.randn(NH, dtype=torch.bfloat16)
    D = torch.randn(NH, dtype=torch.bfloat16)
    dt_bias = torch.randn(NH, dtype=torch.bfloat16)
    xs = torch.randn(seq, NH * HD, dtype=torch.bfloat16)
    Bs = torch.randn(seq, DS, dtype=torch.bfloat16)
    Cs = torch.randn(seq, DS, dtype=torch.bfloat16)
    dts = torch.randn(seq, NH, dtype=torch.bfloat16)

    outs = reference(A_log, D, dt_bias, xs, Bs, Cs, dts)

    with open(out_dir / "golden_a_log.hex", "w") as f:
        for v in A_log:
            f.write(f"{bf16_hex(v)}\n")
    with open(out_dir / "golden_d.hex", "w") as f:
        for v in D:
            f.write(f"{bf16_hex(v)}\n")
    with open(out_dir / "golden_dt_bias.hex", "w") as f:
        for v in dt_bias:
            f.write(f"{bf16_hex(v)}\n")

    with open(out_dir / "golden_inputs.hex", "w") as f:
        for t in range(seq):
            for i in range(NH * HD):
                f.write(f"{bf16_hex(xs[t, i])}\n")
            for i in range(DS):
                f.write(f"{bf16_hex(Bs[t, i])}\n")
            for i in range(DS):
                f.write(f"{bf16_hex(Cs[t, i])}\n")
            for i in range(NH):
                f.write(f"{bf16_hex(dts[t, i])}\n")

    with open(out_dir / "golden_outputs.hex", "w") as f:
        for v in outs:
            f.write(f"{f32_hex(v)}\n")

    print(f"heads={NH} head_dim={HD} d_state={DS} seq={seq}")
    print(f"  A_log={A_log.float().tolist()}")
    print(f"  D={D.float().tolist()}")
    print(f"  dt_bias={dt_bias.float().tolist()}")
    print(f"  y[0,:4]={[round(float(v),6) for v in outs[:4]]}")


if __name__ == "__main__":
    main()
