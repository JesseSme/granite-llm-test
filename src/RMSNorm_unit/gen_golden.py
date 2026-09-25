"""Generate golden samples for RMSNorm unit testing.

Creates random bfloat16 input vectors and weight vectors,
computes RMSNorm using bfloat16 at every step (matching RTL behavior),
and saves as hex files.
"""

import torch
import numpy as np
from pathlib import Path


def rms_norm_bf16_only(x, weight, eps=1e-5):
    """fp32 datapath mirror of rmsnorm_unit: sequential fp32 sum of squares,
    fp32 mean/eps/sqrt/divide/weight multiply, one bfloat16 rounding."""
    n = int(x.numel())
    xf = x.float()
    sumsq = torch.zeros((), dtype=torch.float32)
    for i in range(n):
        sumsq = sumsq + xf[i] * xf[i]
    mean = sumsq / torch.tensor(float(n), dtype=torch.float32)
    rms = torch.sqrt(mean + torch.tensor(float(eps), dtype=torch.float32))
    return ((xf / rms) * weight.float()).bfloat16()


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
