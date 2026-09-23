"""Generate a golden sample for the mamba2_unit cocotb test.

Emulates GraniteMoeHybridMambaLayer in float32 with the model's rounding points:

  in_proj    : bf16 in/weights, sequential fp32 accumulation, one bf16 rounding
  conv1d     : causal depthwise 4-tap FIR over B_C (weights in the RTL tap
               order: tap 0 = current input), fp32 accumulation + bias, one
               bf16 rounding; then silu in fp32 rounded to bf16 (the model's
               causal_conv1d_fn casts the activation result back to bf16)
  SSM        : the RTL recurrence (z = bf16(dt+dt_bias), dtp =
               bf16(softplus(z)), dA = exp(A*dtp), h = dA*h + (dtp*B)*x,
               y = D*x + sum_s h*C) in fp32 with torch's exp/softplus; the
               hardware exp is a polynomial (~1e-5 relative), so the unit test
               uses a tolerance
  gated norm : xn = ssm_out * silu(gate) in fp32, y = xn/sqrt(mean(xn^2)+1e-5)
               * weight, rounded once to bf16 (matching .to(dtype))
  out_proj   : same as in_proj

Tokens are processed in order: the conv history and the SSM state carry over.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def bf16_hex(t):
    return f"{t.view(torch.uint16).item():04x}"


def linear_seq(x, W):
    """(N, IN) bf16 x (OUT, IN) bf16 -> (N, OUT) bf16, RTL accumulation order."""
    outs = torch.empty(x.shape[0], W.shape[0], dtype=torch.bfloat16)
    for v in range(x.shape[0]):
        acc = torch.zeros(W.shape[0], dtype=torch.float32)
        for i in range(x.shape[1]):
            acc = acc + x[v, i].float() * W[:, i].float()
        outs[v] = acc.bfloat16()
    return outs


def conv_seq(bc, Wc, bias):
    """Causal depthwise conv over (T, C); Wc is in RTL tap order (tap 0 =
    current input), so the taps are reversed for PyTorch's cross-correlation.

    ATen's bf16 conv (fp32 accumulation, bf16 output) is used because the
    conv1d_unit was verified bit-exact against it; an explicit Python sum
    differs by up to 1 bf16 ULP at rounding boundaries, which the recurrence
    then amplifies.
    """
    T, C = bc.shape
    x = bc.transpose(0, 1).unsqueeze(0)                 # (1, C, T)
    w = Wc.flip(-1).unsqueeze(1)                        # (C, 1, KERNEL)
    out = F.conv1d(x, w, bias, padding=3, groups=C)[:, :, :T]   # (1, C, T)
    return out[0].transpose(0, 1).contiguous().to(torch.bfloat16)  # (T, C)


def ssm_seq(A_log, D, dt_bias, xs, Bs, Cs, dts, heads, head_dim, d_state):
    """RTL SSM recurrence (fp32) with torch exp/softplus."""
    A = -torch.exp(A_log.float())
    h = torch.zeros(heads, head_dim, d_state, dtype=torch.float32)
    outs = []
    for t in range(xs.shape[0]):
        z = (dts[t] + dt_bias).to(torch.bfloat16)
        dtp = F.softplus(z).float()
        dA = torch.exp(A * dtp)
        xv = xs[t].view(heads, head_dim).float()
        Bv = Bs[t].float()
        Cv = Cs[t].float()
        w = dtp.unsqueeze(1) * Bv.unsqueeze(0)
        for hh in range(heads):
            for dd in range(head_dim):
                acc = D[hh].float() * xv[hh, dd]
                for ss in range(d_state):
                    newh = dA[hh] * h[hh, dd, ss] + w[hh, ss] * xv[hh, dd]
                    h[hh, dd, ss] = newh
                    acc = acc + newh * Cv[ss]
                outs.append(acc)
    return torch.stack(outs).reshape(xs.shape[0], heads * head_dim)


def reference(x, W_in, Wc, cb, W_out, norm_w, A_log, D, dt_bias,
              heads, head_dim, d_state):
    inter = heads * head_dim
    conv_ch = inter + 2 * d_state
    proj = linear_seq(x, W_in)
    gate = proj[:, :inter]
    bc = proj[:, inter:inter + conv_ch]
    dt = proj[:, inter + conv_ch:]

    conv = conv_seq(bc, Wc, cb)
    act = F.silu(conv.float()).bfloat16()          # conv activation (bf16)
    x_ssm = act[:, :inter]
    B = act[:, inter:inter + d_state]
    C = act[:, inter + d_state:inter + 2 * d_state]

    ssm_out = ssm_seq(A_log, D, dt_bias, x_ssm, B, C, dt,
                      heads, head_dim, d_state)

    g = F.silu(gate.float())                        # gate branch stays fp32
    xn = ssm_out * g
    var = xn.pow(2).mean(-1, keepdim=True)
    y = xn / torch.sqrt(var + 1e-5) * norm_w.float()
    return linear_seq(y.bfloat16(), W_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=2)
    ap.add_argument("--d-state", type=int, default=2)
    ap.add_argument("--seq", type=int, default=4)
    args = ap.parse_args()

    hidden, heads, head_dim, d_state = (args.hidden, args.heads,
                                        args.head_dim, args.d_state)
    inter = heads * head_dim
    conv_ch = inter + 2 * d_state
    proj = inter + conv_ch + heads

    out_dir = Path(__file__).parent
    torch.manual_seed(42)

    x = torch.randn(args.seq, hidden, dtype=torch.bfloat16)
    W_in = torch.randn(proj, hidden, dtype=torch.bfloat16)
    Wc = torch.randn(conv_ch, 4, dtype=torch.bfloat16)
    cb = torch.randn(conv_ch, dtype=torch.bfloat16)
    W_out = torch.randn(hidden, inter, dtype=torch.bfloat16)
    norm_w = torch.randn(inter, dtype=torch.bfloat16)
    A_log = torch.randn(heads, dtype=torch.bfloat16)
    D = torch.randn(heads, dtype=torch.bfloat16)
    dt_bias = torch.randn(heads, dtype=torch.bfloat16)

    y = reference(x, W_in, Wc, cb, W_out, norm_w, A_log, D, dt_bias,
                  heads, head_dim, d_state)

    def write(name, t):
        with open(out_dir / name, "w") as f:
            for v in t.flatten():
                f.write(f"{bf16_hex(v)}\n")

    write("golden_in_proj_w.hex", W_in)
    write("golden_out_proj_w.hex", W_out)
    write("golden_conv_w.hex", Wc)
    write("golden_conv_b.hex", cb)
    write("golden_norm_w.hex", norm_w)
    write("golden_a_log.hex", A_log)
    write("golden_d.hex", D)
    write("golden_dt_bias.hex", dt_bias)
    write("golden_inputs.hex", x)
    write("golden_outputs.hex", y)

    print(f"hidden={hidden} heads={heads} head_dim={head_dim} "
          f"d_state={d_state} inter={inter} conv_ch={conv_ch} proj={proj} "
          f"seq={args.seq}")
    print(f"  x[0,:4] = {x[0, :4].float().tolist()}")
    print(f"  y[0,:4] = {y[0, :4].float().tolist()}")


if __name__ == "__main__":
    main()
