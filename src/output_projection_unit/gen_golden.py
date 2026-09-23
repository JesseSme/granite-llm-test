#!/usr/bin/env python3
"""Generate golden samples for output_projection_unit.

Reference: the model's LM head, `GraniteMoeHybridForCausalLM.forward`:
    logits = self.lm_head(hidden_states)          # Linear(HIDDEN, VOCAB, bias=False)
    logits = logits / self.config.logits_scaling  # 3
The weight is tied to the input embedding table. Computed with torch in
bfloat16 so the golden follows the same rounding as the model (one rounding
per op from the fp32 internal precision).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent


def bf16_hex(t: torch.Tensor) -> str:
    return f"{t.view(torch.uint16).item():04x}"


def reference(x: torch.Tensor, W: torch.Tensor, scaling: float) -> torch.Tensor:
    return (F.linear(x, W) / scaling).to(torch.bfloat16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--seq", type=int, default=4)
    ap.add_argument("--scaling", type=float, default=3.0)
    args = ap.parse_args()
    torch.manual_seed(0)

    x = torch.randn(args.seq, args.hidden, dtype=torch.bfloat16)
    W = torch.randn(args.vocab, args.hidden, dtype=torch.bfloat16)
    y = reference(x, W, args.scaling)

    def write(name: str, t: torch.Tensor) -> None:
        with open(ROOT / name, "w") as f:
            for v in t.reshape(-1):
                f.write(bf16_hex(v) + "\n")

    write("golden_weights.hex", W)
    write("golden_inputs.hex", x)
    write("golden_outputs.hex", y)
    print(f"hidden={args.hidden} vocab={args.vocab} seq={args.seq} scaling={args.scaling}")
    print("  y[0,:4] =", [float(v) for v in y[0, :4]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
