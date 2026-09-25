import os
import math
import json

import torch


MODEL_DIR = os.path.abspath(os.path.dirname(__file__))

# Test progressively smaller mathematical representations.
COEFFICIENT_COUNTS = [8, 16, 32, 64, 128, 256, 384]

BATCH_SIZE = 512


def load_embedding(model_dir):
    """
    Load only model.embed_tokens.weight from a local
    Hugging Face safetensors checkpoint.
    """

    from safetensors import safe_open

    safetensor_files = []

    for root, _, files in os.walk(model_dir):
        for filename in files:
            if filename.endswith(".safetensors"):
                safetensor_files.append(
                    os.path.join(root, filename)
                )

    if not safetensor_files:
        raise FileNotFoundError(
            f"No .safetensors files found in {model_dir}"
        )

    print("Safetensors files:")

    for path in safetensor_files:
        print("  ", path)

    # Search every shard until we find the embedding.
    for path in safetensor_files:

        with safe_open(
            path,
            framework="pt",
            device="cpu"
        ) as f:

            keys = list(f.keys())

            for key in keys:

                if key.endswith("embed_tokens.weight"):

                    print()
                    print("Found embedding:")
                    print(" ", key)
                    print(" ", path)

                    weight = f.get_tensor(key)

                    print()
                    print("Shape:", tuple(weight.shape))
                    print("Dtype:", weight.dtype)

                    return weight

    raise RuntimeError(
        "Could not find embed_tokens.weight in checkpoint."
    )


def create_dct_basis(n, k):
    """
    Create K orthonormal DCT-II basis functions.

    Shape:

        [n, k]

    The basis is mathematical and therefore does not
    need to be stored with the compressed model.
    """

    x = torch.arange(
        n,
        dtype=torch.float64
    ).unsqueeze(1)

    j = torch.arange(
        k,
        dtype=torch.float64
    ).unsqueeze(0)

    basis = torch.cos(
        math.pi / n *
        (x + 0.5) *
        j
    )

    basis[:, 0] *= 1.0 / math.sqrt(n)

    if k > 1:
        basis[:, 1:] *= math.sqrt(2.0 / n)

    return basis


def bf16_equal_fraction(original, reconstructed):
    """
    Convert reconstruction to BF16 and determine how many
    values become exactly equal to the original BF16 values.
    """

    reconstructed_bf16 = reconstructed.to(torch.bfloat16)

    equal = reconstructed_bf16 == original

    return equal.float().mean().item()


def evaluate(weight, k):

    num_tokens, dimensions = weight.shape

    print()
    print("=" * 70)
    print(f"Testing {k} mathematical coefficients per token")
    print("=" * 70)

    basis = create_dct_basis(
        dimensions,
        k
    )

    total_squared_error = 0.0
    total_absolute_error = 0.0

    total_values = 0
    exact_bf16_values = 0

    maximum_error = 0.0

    # Original storage:
    #
    # 100352 × 768 × 2 bytes
    #
    original_bytes = (
        num_tokens *
        dimensions *
        2
    )

    # We store K BF16 coefficients per token.
    compressed_bytes = (
        num_tokens *
        k *
        2
    )

    for start in range(
        0,
        num_tokens,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            num_tokens
        )

        original = weight[
            start:end
        ].to(torch.float64)

        # --------------------------------------------------
        # Fit mathematical coefficients
        #
        # E ≈ C Bᵀ
        #
        # C = E B
        # --------------------------------------------------

        coefficients = original @ basis

        # --------------------------------------------------
        # Reconstruct the embedding
        # --------------------------------------------------

        reconstructed = (
            coefficients @ basis.T
        )

        error = (
            reconstructed -
            original
        )

        squared_error = (
            error * error
        )

        absolute_error = torch.abs(error)

        total_squared_error += (
            squared_error.sum().item()
        )

        total_absolute_error += (
            absolute_error.sum().item()
        )

        total_values += original.numel()

        maximum_error = max(
            maximum_error,
            absolute_error.max().item()
        )

        # --------------------------------------------------
        # The important test:
        #
        # Does the mathematical reconstruction produce
        # the exact original BF16 value?
        # --------------------------------------------------

        reconstructed_bf16 = (
            reconstructed.to(torch.bfloat16)
        )

        exact_bf16_values += (
            (
                reconstructed_bf16 ==
                weight[start:end]
            )
            .sum()
            .item()
        )

    mse = (
        total_squared_error /
        total_values
    )

    rmse = math.sqrt(mse)

    mae = (
        total_absolute_error /
        total_values
    )

    exact_fraction = (
        exact_bf16_values /
        total_values
    )

    compression_ratio = (
        original_bytes /
        compressed_bytes
    )

    print()
    print(f"Bytes/token:              {k * 2}")
    print(
        f"Original size:             "
        f"{original_bytes / 1024**2:.2f} MiB"
    )
    print(
        f"Compressed size:           "
        f"{compressed_bytes / 1024**2:.2f} MiB"
    )
    print(
        f"Compression ratio:         "
        f"{compression_ratio:.2f}x"
    )
    print()
    print(f"MSE:                       {mse:.8g}")
    print(f"RMSE:                      {rmse:.8g}")
    print(f"Mean absolute error:       {mae:.8g}")
    print(f"Maximum absolute error:    {maximum_error:.8g}")
    print()
    print(
        f"Exact original BF16:       "
        f"{exact_fraction * 100:.4f}%"
    )

    return {
        "coefficients": k,
        "bytes_per_token": k * 2,
        "compressed_mib":
            compressed_bytes / 1024**2,
        "compression_ratio":
            compression_ratio,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "max_error": maximum_error,
        "exact_bf16_fraction":
            exact_fraction,
    }


def main():

    print("=" * 70)
    print("Granite Embedding Mathematical Compression Experiment")
    print("=" * 70)

    print()
    print("Model directory:")
    print(MODEL_DIR)

    weight = load_embedding(
        MODEL_DIR
    )

    if weight.ndim != 2:
        raise RuntimeError(
            f"Expected 2D matrix, got {weight.shape}"
        )

    num_tokens, dimensions = weight.shape

    print()
    print("Embedding matrix:")
    print(f"  Tokens:     {num_tokens}")
    print(f"  Dimensions: {dimensions}")
    print(f"  Dtype:      {weight.dtype}")

    original_bytes = (
        num_tokens *
        dimensions *
        2
    )

    print()
    print(
        f"Original storage: "
        f"{original_bytes:,} bytes "
        f"({original_bytes / 1024**2:.2f} MiB)"
    )

    results = []

    for k in COEFFICIENT_COUNTS:

        results.append(
            evaluate(
                weight,
                k
            )
        )

    output_file = os.path.join(
        MODEL_DIR,
        "compression_results.json"
    )

    with open(
        output_file,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print()

    print(
        f"{'K':>6} "
        f"{'Bytes':>8} "
        f"{'MiB':>10} "
        f"{'Ratio':>10} "
        f"{'RMSE':>14} "
        f"{'Exact BF16':>14}"
    )

    for r in results:

        print(
            f"{r['coefficients']:>6} "
            f"{r['bytes_per_token']:>8} "
            f"{r['compressed_mib']:>10.2f} "
            f"{r['compression_ratio']:>9.2f}x "
            f"{r['rmse']:>14.6g} "
            f"{r['exact_bf16_fraction'] * 100:>13.4f}%"
        )

    print()
    print(
        f"Results written to: {output_file}"
    )


if __name__ == "__main__":
    main()
