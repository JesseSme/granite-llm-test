"""Generate a golden sample for the residual_adder_unit cocotb test.

Computes the model expression in bfloat16 exactly as torch does:
`residual + branch * 0.246` with the Python float scalar at full precision and
each operation rounded once to bfloat16. The RTL widens to binary32, uses the
fp32 constant and rounds each op once, which reproduces this bit-exactly.
"""

from pathlib import Path

import torch

WIDTH = 768
N_VECTORS = 8
MULT = 0.246


def bf16_hex(t):
    return f"{t.view(torch.uint16).item():04x}"


def main():
    out_dir = Path(__file__).parent
    torch.manual_seed(42)

    branch = torch.randn(N_VECTORS, WIDTH, dtype=torch.bfloat16)
    residual = torch.randn(N_VECTORS, WIDTH, dtype=torch.bfloat16)
    outputs = residual + branch * MULT

    with open(out_dir / "golden_inputs.hex", "w") as f:
        for v in range(N_VECTORS):
            for i in range(WIDTH):
                f.write(f"{bf16_hex(branch[v, i])}\n")

    with open(out_dir / "golden_residuals.hex", "w") as f:
        for v in range(N_VECTORS):
            for i in range(WIDTH):
                f.write(f"{bf16_hex(residual[v, i])}\n")

    with open(out_dir / "golden_outputs.hex", "w") as f:
        for v in range(N_VECTORS):
            for i in range(WIDTH):
                f.write(f"{bf16_hex(outputs[v, i])}\n")

    print(f"Vectors: {N_VECTORS} x {WIDTH}")
    print("Sample (vector 0):")
    print(f"  branch[:4]   = {branch[0, :4].float().tolist()}")
    print(f"  residual[:4] = {residual[0, :4].float().tolist()}")
    print(f"  output[:4]   = {outputs[0, :4].float().tolist()}")


if __name__ == "__main__":
    main()
