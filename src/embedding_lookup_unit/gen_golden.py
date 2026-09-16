"""Generate a golden sample for the embedding_lookup_unit cocotb test.

Creates random bfloat16 rows for a small set of token IDs, computes the
expected output as `row * 12.0` in bfloat16 (matching the model's
`embed_tokens(input_ids) * embedding_multiplier`), and saves hex files.

Only the rows used by the test are generated/loaded; the DUT table itself is
parameterized to the full vocab.
"""

import os
from pathlib import Path

import torch

VOCAB = 100352
DIM = 768
N_TOKENS = 16
SCALE = 12.0


def bf16_hex(t):
    return f"{t.view(torch.uint16).item():04x}"


def main():
    out_dir = Path(__file__).parent
    torch.manual_seed(42)

    token_ids = torch.randperm(VOCAB)[:N_TOKENS].tolist()
    weights = torch.randn(N_TOKENS, DIM, dtype=torch.bfloat16)
    outputs = weights * SCALE

    with open(out_dir / "golden_tokens.hex", "w") as f:
        for tok in token_ids:
            f.write(f"{tok:06x}\n")

    with open(out_dir / "golden_weights.hex", "w") as f:
        for i in range(N_TOKENS):
            for j in range(DIM):
                f.write(f"{bf16_hex(weights[i, j])}\n")

    with open(out_dir / "golden_outputs.hex", "w") as f:
        for i in range(N_TOKENS):
            for j in range(DIM):
                f.write(f"{bf16_hex(outputs[i, j])}\n")

    print(f"Tokens: {token_ids}")
    print(f"Weights: {N_TOKENS} x {DIM} -> golden_weights.hex")
    print(f"Outputs: {N_TOKENS} x {DIM} -> golden_outputs.hex")
    print("Sample (token 0):")
    print(f"  row[:4]    = {weights[0, :4].float().tolist()}")
    print(f"  output[:4] = {outputs[0, :4].float().tolist()}")


if __name__ == "__main__":
    main()
