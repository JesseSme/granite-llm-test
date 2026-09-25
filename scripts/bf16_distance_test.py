import os
import torch
from safetensors import safe_open

# ============================================================
# Configuration
# ============================================================

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "granite-4.0-h-350m"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "model.safetensors"
)

CHUNK_SIZE = 4096


# ============================================================
# Find embedding
# ============================================================

def find_embedding():

    print("Opening:")
    print(MODEL_FILE)

    with safe_open(
        MODEL_FILE,
        framework="pt",
        device="cpu"
    ) as f:

        for key in f.keys():

            if key.endswith("embed_tokens.weight"):

                print("Found:", key)

                return key

    raise RuntimeError(
        "model.embed_tokens.weight not found"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Granite BF16 Embedding Precision Test")
    print("=" * 70)

    key = find_embedding()

    with safe_open(
        MODEL_FILE,
        framework="pt",
        device="cpu"
    ) as f:

        # ----------------------------------------------------
        # Read metadata
        # ----------------------------------------------------

        weight = f.get_slice(key)

        shape = weight.get_shape()

        tokens = shape[0]
        dimensions = shape[1]

        print()
        print("Shape:")
        print(f"  Tokens     : {tokens:,}")
        print(f"  Dimensions : {dimensions}")
        print("  Dtype      : bfloat16")

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_values = 0

        sum_abs = 0.0
        sum_squared = 0.0

        max_abs = 0.0

        min_value = float("inf")
        max_value = float("-inf")

        # BF16 -> FP32 rounding distance
        total_bf16_rounding = 0.0
        max_bf16_rounding = 0.0

        # FP16 comparison
        total_fp16_error = 0.0
        max_fp16_error = 0.0

        # FP32 is the reference representation
        # for numerical calculations.

        # ----------------------------------------------------
        # Process chunks
        # ----------------------------------------------------

        print()
        print("Processing embedding in chunks...")

        for start in range(
            0,
            tokens,
            CHUNK_SIZE
        ):

            end = min(
                start + CHUNK_SIZE,
                tokens
            )

            x_bf16 = weight[
                start:end
            ]

            x = x_bf16.float()

            # ------------------------------------------------
            # Basic statistics
            # ------------------------------------------------

            total_values += x.numel()

            min_value = min(
                min_value,
                x.min().item()
            )

            max_value = max(
                max_value,
                x.max().item()
            )

            # ------------------------------------------------
            # BF16 -> FP32
            #
            # This conversion itself is exact.
            #
            # BF16 values are exactly representable in FP32.
            # ------------------------------------------------

            reconstructed_bf16 = (
                x.to(torch.bfloat16)
                .float()
            )

            error = (
                reconstructed_bf16 - x
            ).abs()

            sum_abs += error.sum().item()

            sum_squared += (
                error * error
            ).sum().item()

            chunk_max = error.max().item()

            max_abs = max(
                max_abs,
                chunk_max
            )

            # ------------------------------------------------
            # Compare BF16 against FP16
            # ------------------------------------------------

            fp16 = x.to(torch.float16).float()

            fp16_error = (
                fp16 - x
            ).abs()

            total_fp16_error += (
                fp16_error.sum().item()
            )

            max_fp16_error = max(
                max_fp16_error,
                fp16_error.max().item()
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if start % (
                CHUNK_SIZE * 20
            ) == 0:

                percent = (
                    end / tokens * 100
                )

                print(
                    f"  {percent:6.2f}%"
                )

    # ========================================================
    # Results
    # ========================================================

    mean_abs = (
        sum_abs /
        total_values
    )

    mse = (
        sum_squared /
        total_values
    )

    rmse = mse ** 0.5

    mean_fp16_error = (
        total_fp16_error /
        total_values
    )

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print()
    print("Embedding value range")
    print("----------------------")

    print(
        f"Minimum value       : {min_value:.10f}"
    )

    print(
        f"Maximum value       : {max_value:.10f}"
    )

    print()
    print("BF16 numerical precision")
    print("------------------------")

    print(
        f"Mean absolute error : {mean_abs:.12g}"
    )

    print(
        f"RMSE                : {rmse:.12g}"
    )

    print(
        f"Maximum error       : {max_abs:.12g}"
    )

    print()
    print("FP16 comparison")
    print("----------------")

    print(
        f"Mean absolute error : "
        f"{mean_fp16_error:.12g}"
    )

    print(
        f"Maximum error       : "
        f"{max_fp16_error:.12g}"
    )

    print()
    print("Storage")
    print("-------")

    original_bytes = (
        tokens *
        dimensions *
        2
    )

    fp32_bytes = (
        tokens *
        dimensions *
        4
    )

    print(
        f"BF16 storage        : "
        f"{original_bytes / 1024**2:.2f} MiB"
    )

    print(
        f"FP32 storage        : "
        f"{fp32_bytes / 1024**2:.2f} MiB"
    )

    print()
    print("Important:")
    print(
        "BF16 -> FP32 is lossless."
    )
    print(
        "The BF16 numbers stored in the model are already "
        "rounded representations of the original training values."
    )


if __name__ == "__main__":
    main()
