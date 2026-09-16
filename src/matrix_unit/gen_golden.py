"""Generate a golden sample for the matrix_unit cocotb test.

Reference semantics (matches torch bf16 F.linear to within rare 1-ULP due to
ATen GEMM ordering): bf16 inputs/weights widened to fp32, sequential fp32
accumulation, bias added last in fp32, one rounding to bf16.

Also used to generate the in-loop weights/inputs when running with real layer
dimensions (see run_test.py).
"""

import argparse
from pathlib import Path

import torch


def reference(x, W, b):
    """Sequential fp32 accumulation, bias last, rounded once to bf16."""
    outs = torch.empty(x.shape[0], W.shape[0], dtype=torch.bfloat16)
    for v in range(x.shape[0]):
        acc = torch.zeros(W.shape[0], dtype=torch.float32)
        for i in range(x.shape[1]):
            acc = acc + x[v, i].float() * W[:, i].float()
        outs[v] = (acc + b.float()).bfloat16()
    return outs


def bf16_hex(t):
    return f"{t.view(torch.uint16).item():04x}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="IN", type=int, default=32)
    ap.add_argument("--out", dest="OUT", type=int, default=16)
    ap.add_argument("--vectors", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(__file__).parent
    torch.manual_seed(42)

    x = torch.randn(args.vectors, args.IN, dtype=torch.bfloat16)
    W = torch.randn(args.OUT, args.IN, dtype=torch.bfloat16)
    b = torch.randn(args.OUT, dtype=torch.bfloat16)
    y = reference(x, W, b)

    with open(out_dir / "golden_inputs.hex", "w") as f:
        for v in range(args.vectors):
            for i in range(args.IN):
                f.write(f"{bf16_hex(x[v, i])}\n")

    with open(out_dir / "golden_weights.hex", "w") as f:
        for o in range(args.OUT):
            for i in range(args.IN):
                f.write(f"{bf16_hex(W[o, i])}\n")

    with open(out_dir / "golden_biases.hex", "w") as f:
        for o in range(args.OUT):
            f.write(f"{bf16_hex(b[o])}\n")

    with open(out_dir / "golden_outputs.hex", "w") as f:
        for v in range(args.vectors):
            for o in range(args.OUT):
                f.write(f"{bf16_hex(y[v, o])}\n")

    print(f"IN={args.IN} OUT={args.OUT} vectors={args.vectors}")
    print(f"  x[0,:4] = {x[0, :4].float().tolist()}")
    print(f"  y[0,:4] = {y[0, :4].float().tolist()}")


if __name__ == "__main__":
    main()
