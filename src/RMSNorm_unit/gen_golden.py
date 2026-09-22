"""Generate golden samples for RMSNorm unit testing.

Creates random bfloat16 input vectors and weight vectors,
computes RMSNorm using bfloat16 at every step (matching RTL behavior),
and saves as hex files.
"""

import torch
import numpy as np
from pathlib import Path


def rms_norm_bf16_only(x, weight, eps=1e-5):
    """Compute RMSNorm entirely in bfloat16 precision (matching RTL).
    
    The RTL computes:
    1. sum_sq = Σ(x_i²)  -- accumulated in bfloat16
    2. mean = sum_sq / 768  -- bfloat16 division
    3. mean_eps = mean + eps  -- bfloat16 addition
    4. rms = sqrt(mean_eps)  -- bfloat16 sqrt
    5. normalized = x / rms  -- bfloat16 division
    6. output = normalized * weight  -- bfloat16 multiplication
    """
    # Cast to bfloat16 to match RTL precision at every step
    x_bf16 = x.bfloat16()
    weight_bf16 = weight.bfloat16()
    width_bf16 = torch.tensor(768.0, dtype=torch.bfloat16)
    eps_bf16 = torch.tensor(float(eps), dtype=torch.bfloat16)
    
    # Step 1: Sum of squares (accumulated in bfloat16)
    sum_sq = torch.tensor(0.0, dtype=torch.bfloat16)
    for i in range(len(x_bf16)):
        square = x_bf16[i] * x_bf16[i]
        sum_sq = sum_sq + square
    
    # Step 2: Mean (bfloat16 division)
    mean = sum_sq / width_bf16
    
    # Step 3: Add epsilon (bfloat16 addition)
    mean_eps = mean + eps_bf16
    
    # Step 4: Square root (bfloat16 sqrt)
    rms = torch.sqrt(mean_eps)
    
    # Step 5: Normalize (bfloat16 division)
    normalized = x_bf16 / rms
    
    # Step 6: Apply weight (bfloat16 multiplication)
    output = normalized * weight_bf16
    
    return output


def float_to_bf16_hex(val):
    """Convert a float value to bfloat16 hex string."""
    bf16 = torch.tensor([val], dtype=torch.bfloat16)
    bits = bf16.view(torch.uint16).item()
    return f"{bits:04x}"


def main():
    output_dir = Path(__file__).parent
    WIDTH = 768
    NUM_VECTORS = 10
    EPS = 1e-5
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate weight vector (shared across all vectors)
    weight = torch.randn(WIDTH, dtype=torch.bfloat16)
    
    # Save weight
    with open(output_dir / "weight.hex", "w") as f:
        for w in weight:
            f.write(f"{float_to_bf16_hex(w.item())}\n")
    
    print(f"Weight saved to {output_dir / 'weight.hex'}")
    print(f"Weight sample (first 8): {weight[:8].float().tolist()}")
    
    # Generate input vectors and compute golden outputs
    for i in range(NUM_VECTORS):
        x = torch.randn(WIDTH, dtype=torch.bfloat16)
        
        # Compute RMSNorm in bfloat16-only mode (matching RTL)
        y = rms_norm_bf16_only(x, weight, eps=EPS)
        
        # Save input
        with open(output_dir / f"input_{i}.hex", "w") as f:
            for val in x:
                f.write(f"{float_to_bf16_hex(val.item())}\n")
        
        # Save output
        with open(output_dir / f"output_{i}.hex", "w") as f:
            for val in y:
                f.write(f"{float_to_bf16_hex(val.item())}\n")
    
    print(f"Generated {NUM_VECTORS} test vectors")
    
    # Print verification
    x = torch.randn(WIDTH, dtype=torch.bfloat16)
    y = rms_norm_bf16_only(x, weight, eps=EPS)
    print(f"\nVerification vector sample (first 8):")
    print(f"  Input:  {x[:8].float().tolist()}")
    print(f"  Output: {y[:8].float().tolist()}")


if __name__ == "__main__":
    main()
