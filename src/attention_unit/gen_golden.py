"""Generate a golden sample for the attention_unit cocotb test.

Emulates the RTL datapath in float32 with the same rounding points:

  Q/K/V/O projections : bf16 in/weights widened to fp32, sequential fp32
                        accumulation over the input features, one bf16 rounding
                        (identical to matrix_unit / torch F.linear)
  score_j             : sequential fp32 dot product, rounded to bf16, then
                        multiplied by attention_multiplier (0.015625) in fp32
                        and rounded to bf16 again (torch: bf16 matmul -> bf16,
                        then bf16 tensor * python float -> bf16)
  softmax             : fp32 (torch's exact exp; the RTL exp is a ~1e-5
                        relative-error polynomial, hence the unit tolerance)
  context_d           : sum_j bf16(p_j) * bf16(v_j[d]) in fp32, one bf16
                        rounding
  tokens are processed in order, K/V cached as bf16 (causal attention)

Dimensions come from the CLI (must match the DUT build parameters).
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

SCALE = 0.015625


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


def reference(x, Wq, Wk, Wv, Wo, heads, kv_heads, head_dim):
    seq = x.shape[0]
    hidden = heads * head_dim
    kn = kv_heads * head_dim
    groups = heads // kv_heads

    kc = torch.zeros(seq, kn, dtype=torch.bfloat16)
    vc = torch.zeros(seq, kn, dtype=torch.bfloat16)
    outs = torch.empty(seq, hidden, dtype=torch.bfloat16)

    for t in range(seq):
        q = linear_seq(x[t:t + 1], Wq)[0]
        k = linear_seq(x[t:t + 1], Wk)[0]
        v = linear_seq(x[t:t + 1], Wv)[0]
        kc[t] = k
        vc[t] = v

        ctx = torch.empty(hidden, dtype=torch.bfloat16)
        for h in range(heads):
            kv = h // groups
            scores = torch.empty(t + 1, dtype=torch.bfloat16)
            for j in range(t + 1):
                acc = torch.zeros((), dtype=torch.float32)
                for d in range(head_dim):
                    acc = acc + (q[h * head_dim + d].float() *
                                 kc[j, kv * head_dim + d].float())
                s_bf = acc.bfloat16()
                scores[j] = (s_bf.float() * SCALE).bfloat16()
            p = F.softmax(scores.float(), dim=-1).bfloat16()
            for d in range(head_dim):
                acc = torch.zeros((), dtype=torch.float32)
                for j in range(t + 1):
                    acc = acc + p[j].float() * vc[j, kv * head_dim + d].float()
                ctx[h * head_dim + d] = acc.bfloat16()
        outs[t] = linear_seq(ctx.unsqueeze(0), Wo)[0]
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--kv-heads", type=int, default=1)
    ap.add_argument("--head-dim", type=int, default=8)
    ap.add_argument("--seq", type=int, default=4)
    args = ap.parse_args()

    heads, kv_heads, head_dim = args.heads, args.kv_heads, args.head_dim
    hidden, kn, seq = heads * head_dim, kv_heads * head_dim, args.seq
    assert hidden == args.hidden, "hidden must equal heads*head_dim"

    out_dir = Path(__file__).parent
    torch.manual_seed(42)

    x = torch.randn(seq, hidden, dtype=torch.bfloat16)
    Wq = torch.randn(hidden, hidden, dtype=torch.bfloat16)
    Wk = torch.randn(kn, hidden, dtype=torch.bfloat16)
    Wv = torch.randn(kn, hidden, dtype=torch.bfloat16)
    Wo = torch.randn(hidden, hidden, dtype=torch.bfloat16)

    y = reference(x, Wq, Wk, Wv, Wo, heads, kv_heads, head_dim)

    def write(name, t):
        with open(out_dir / name, "w") as f:
            for v in t.flatten():
                f.write(f"{bf16_hex(v)}\n")

    write("golden_q_w.hex", Wq)
    write("golden_k_w.hex", Wk)
    write("golden_v_w.hex", Wv)
    write("golden_o_w.hex", Wo)
    write("golden_inputs.hex", x)
    write("golden_outputs.hex", y)

    print(f"hidden={hidden} heads={heads} kv_heads={kv_heads} "
          f"head_dim={head_dim} seq={seq}")
    print(f"  x[0,:4] = {x[0, :4].float().tolist()}")
    print(f"  y[0,:4] = {y[0, :4].float().tolist()}")


if __name__ == "__main__":
    main()
