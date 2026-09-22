"""Generate a golden sample for the mlp_unit cocotb test.

Emulates GraniteMoeHybridMLP in float32 with the model's rounding points:

  combined = x @ W_gu^T          bf16 in/weights, sequential fp32 accumulation,
                                 one bf16 rounding (matrix_unit semantics)
  gate, up = combined.chunk(2)   gate = first INTER, up = last INTER
  gated    = silu(gate) * up     silu computed in fp32 from the bf16 gate and
                                 rounded to bf16, product in fp32 rounded to bf16
  y        = gated @ W_dn^T      same as the first projection

The hardware reuses the LUT-based silu_unit (sigmoid LUT accuracy < 0.5%), so
the unit test compares with a tolerance rather than bit-exactly.
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


def reference(x, W_gu, W_dn, inter):
    combined = linear_seq(x, W_gu)
    gate = combined[:, :inter]
    up = combined[:, inter:]
    act = F.silu(gate.float()).bfloat16()
    gated = (act.float() * up.float()).bfloat16()
    return linear_seq(gated, W_dn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--inter", type=int, default=8)
    ap.add_argument("--seq", type=int, default=4)
    args = ap.parse_args()

    hidden, inter, seq = args.hidden, args.inter, args.seq
    out_dir = Path(__file__).parent
    torch.manual_seed(42)

    x = torch.randn(seq, hidden, dtype=torch.bfloat16)
    W_gu = torch.randn(2 * inter, hidden, dtype=torch.bfloat16)
    W_dn = torch.randn(hidden, inter, dtype=torch.bfloat16)

    y = reference(x, W_gu, W_dn, inter)

    def write(name, t):
        with open(out_dir / name, "w") as f:
            for v in t.flatten():
                f.write(f"{bf16_hex(v)}\n")

    write("golden_gateup_w.hex", W_gu)
    write("golden_down_w.hex", W_dn)
    write("golden_inputs.hex", x)
    write("golden_outputs.hex", y)

    print(f"hidden={hidden} inter={inter} seq={seq}")
    print(f"  x[0,:4] = {x[0, :4].float().tolist()}")
    print(f"  y[0,:4] = {y[0, :4].float().tolist()}")


if __name__ == "__main__":
    main()
