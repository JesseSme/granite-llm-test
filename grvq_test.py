import os
import math
import time
import torch
import numpy as np
from safetensors import safe_open
from sklearn.cluster import MiniBatchKMeans

# ============================================================
# Configuration
# ============================================================

# Script is in ~/tinyllm
# Model is in ~/tinyllm/granite-4.0-h-350m
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "granite-4.0-h-350m"
)

NUM_GROUPS = 8           # 768 -> 8 groups of 96 dimensions
CODEBOOK_SIZE = 256      # 256 codewords = 1 byte index
RESIDUAL_LEVELS_LIST = [1, 2, 3]

TRAINING_SAMPLES = 30000
BATCH_SIZE = 1024

torch.set_grad_enabled(False)

# ============================================================
# Load embedding matrix
# ============================================================

def load_embedding():
    print("Searching model directory:")
    print(MODEL_DIR)

    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(f"Directory not found: {MODEL_DIR}")

    files = os.listdir(MODEL_DIR)
    print("Files found:")
    for f in files:
        print("  ", f)

    for filename in files:
        if filename.endswith(".safetensors"):
            path = os.path.join(MODEL_DIR, filename)
            print("\nOpening:", path)

            with safe_open(path, framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    if key.endswith("embed_tokens.weight"):
                        weight = sf.get_tensor(key)
                        print("Loaded:", key)
                        print("Shape:", tuple(weight.shape))
                        print("Dtype:", weight.dtype)
                        return weight.float()

    raise RuntimeError("embed_tokens.weight not found.")

# ============================================================
# Train one residual codebook
# ============================================================

def train_codebook(vectors):
    km = MiniBatchKMeans(
        n_clusters=CODEBOOK_SIZE,
        batch_size=2048,
        max_iter=100,
        n_init=1,
        random_state=42,
    )

    km.fit(vectors)

    centers = torch.tensor(
        km.cluster_centers_,
        dtype=torch.float32,
    )

    return centers

# ============================================================
# Find nearest codeword
# ============================================================

def nearest(vectors, codebook):
    distances = (
        (vectors[:, None, :] - codebook[None, :, :]) ** 2
    ).sum(dim=2)

    idx = distances.argmin(dim=1)

    return idx, codebook[idx]

# ============================================================
# Train grouped residual VQ
# ============================================================

def train_grvq(weight, residual_levels):

    tokens, dims = weight.shape
    group_dim = dims // NUM_GROUPS

    sample_ids = np.random.default_rng(42).choice(
        tokens,
        size=min(TRAINING_SAMPLES, tokens),
        replace=False,
    )

    sample = weight[sample_ids]

    codebooks = []

    print(f"\nTraining GRVQ ({residual_levels} residual levels)")

    for g in range(NUM_GROUPS):
        print(f"  Group {g+1}/{NUM_GROUPS}")

        residual = sample[
            :, g * group_dim:(g + 1) * group_dim
        ].clone()

        group_books = []

        for level in range(residual_levels):
            print(f"    Residual level {level+1}")

            cb = train_codebook(residual.numpy())
            group_books.append(cb)

            idx, approx = nearest(residual, cb)
            residual = residual - approx

        codebooks.append(group_books)

    return codebooks

# ============================================================
# Encode + reconstruct
# ============================================================

def reconstruct(weight, codebooks):

    tokens, dims = weight.shape
    group_dim = dims // NUM_GROUPS

    reconstructed = torch.zeros_like(weight)

    for g in range(NUM_GROUPS):

        start = g * group_dim
        end = (g + 1) * group_dim

        residual = weight[:, start:end].clone()
        output = torch.zeros_like(residual)

        for cb in codebooks[g]:
            idx, approx = nearest(residual, cb)
            output += approx
            residual -= approx

        reconstructed[:, start:end] = output

    return reconstructed

# ============================================================
# Metrics
# ============================================================

def metrics(original, reconstructed):

    error = reconstructed - original

    mse = (error ** 2).mean().item()
    rmse = math.sqrt(mse)
    mae = error.abs().mean().item()
    max_err = error.abs().max().item()

    cosine = torch.nn.functional.cosine_similarity(
        original,
        reconstructed,
        dim=1,
    ).mean().item()

    exact = (
        reconstructed.to(torch.bfloat16)
        == original.to(torch.bfloat16)
    ).float().mean().item()

    return rmse, mae, cosine, exact, max_err

# ============================================================
# Storage calculation
# ============================================================

def storage(tokens, dims, residual_levels):

    original_bytes = tokens * dims * 2

    # Code indices
    token_bytes = NUM_GROUPS * residual_levels

    code_bytes = token_bytes * tokens

    group_dim = dims // NUM_GROUPS

    codebook_bytes = (
        NUM_GROUPS
        * residual_levels
        * CODEBOOK_SIZE
        * group_dim
        * 2
    )

    total = code_bytes + codebook_bytes

    return {
        "token_bytes": token_bytes,
        "total_bytes": total,
        "ratio": original_bytes / total,
        "original_mib": original_bytes / 1024**2,
        "compressed_mib": total / 1024**2,
    }

# ============================================================
# Main experiment
# ============================================================

def main():

    print("=" * 70)
    print("Granite H-350M Embedding Compression Test")
    print("Grouped Residual Vector Quantization")
    print("=" * 70)

    weight = load_embedding()

    tokens, dims = weight.shape

    print(f"\nEmbedding matrix: {tokens:,} x {dims}")

    summary = []

    for levels in RESIDUAL_LEVELS_LIST:

        start = time.time()

        codebooks = train_grvq(weight, levels)

        print("  Reconstructing full embedding table...")
        reconstructed = reconstruct(weight, codebooks)

        rmse, mae, cosine, exact, max_err = metrics(
            weight,
            reconstructed,
        )

        s = storage(tokens, dims, levels)

        elapsed = time.time() - start

        summary.append({
            "levels": levels,
            "rmse": rmse,
            "mae": mae,
            "cosine": cosine,
            "exact": exact,
            "max_error": max_err,
            **s,
        })

        print("\nRESULT")
        print("-" * 50)
        print(f"Residual levels      : {levels}")
        print(f"Bytes/token          : {s['token_bytes']}")
        print(f"Compressed size      : {s['compressed_mib']:.2f} MiB")
        print(f"Compression ratio    : {s['ratio']:.2f}x")
        print(f"RMSE                 : {rmse:.6f}")
        print(f"MAE                  : {mae:.6f}")
        print(f"Cosine similarity    : {cosine:.6f}")
        print(f"Max error            : {max_err:.6f}")
        print(f"Exact BF16           : {exact*100:.4f}%")
        print(f"Elapsed              : {elapsed:.1f} sec")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"{'Levels':>6} {'Bytes':>8} {'MiB':>8} "
        f"{'Ratio':>8} {'RMSE':>10} {'Cosine':>10} {'Exact BF16':>12}"
    )

    for r in summary:
        print(
            f"{r['levels']:>6} "
            f"{r['token_bytes']:>8} "
            f"{r['compressed_mib']:>8.2f} "
            f"{r['ratio']:>7.2f}x "
            f"{r['rmse']:>10.6f} "
            f"{r['cosine']:>10.6f} "
            f"{r['exact']*100:>11.4f}%"
        )


if __name__ == "__main__":
    main()
