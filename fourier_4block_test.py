import os
import math
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

# Number of tokens to test.
# Set to None to test the entire vocabulary.
NUM_TOKENS_TO_TEST = None


# ============================================================
# Load embedding
# ============================================================

def find_embedding():

    with safe_open(
        MODEL_FILE,
        framework="pt",
        device="cpu"
    ) as f:

        for key in f.keys():

            if key.endswith("embed_tokens.weight"):
                return key

    raise RuntimeError(
        "model.embed_tokens.weight not found"
    )


# ============================================================
# 4-point Fourier transform
# ============================================================

def fft4(x):
    """
    x shape:

        [..., 4]

    Returns complex Fourier coefficients.
    """

    return torch.fft.fft(
        x,
        dim=-1
    )


# ============================================================
# Inverse Fourier transform
# ============================================================

def ifft4(x):

    return torch.fft.ifft(
        x,
        dim=-1
    ).real


# ============================================================
# Calculate errors
# ============================================================

def calculate_error(
    original,
    reconstructed
):

    error = (
        reconstructed -
        original
    )

    abs_error = error.abs()

    mse = (
        error * error
    ).mean().item()

    rmse = math.sqrt(mse)

    mae = abs_error.mean().item()

    maximum = abs_error.max().item()

    return (
        mse,
        rmse,
        mae,
        maximum
    )


# ============================================================
# Test Fourier compression
# ============================================================

def test_fourier(
    weight,
    tokens,
    dimensions
):

    assert dimensions % 4 == 0

    blocks = dimensions // 4

    print()
    print("=" * 70)
    print("4-ELEMENT FOURIER EXPERIMENT")
    print("=" * 70)

    print()
    print(
        f"Embedding dimensions : {dimensions}"
    )

    print(
        f"4-value blocks/token : {blocks}"
    )

    print(
        f"Tokens tested        : {tokens:,}"
    )

    print()

    # --------------------------------------------------------
    # Statistics for different compression methods
    # --------------------------------------------------------

    methods = {

        # Keep all 4 complex coefficients.
        "FULL FFT": {
            "sum_sq": 0.0,
            "sum_abs": 0.0,
            "max": 0.0,
            "bytes": 16,
        },

        # Keep only coefficients 0 and 2.
        #
        # These are the purely real components for a real
        # 4-element input.
        "FFT 0+2": {
            "sum_sq": 0.0,
            "sum_abs": 0.0,
            "max": 0.0,
            "bytes": 8,
        },

        # Keep only DC coefficient.
        "FFT 0": {
            "sum_sq": 0.0,
            "sum_abs": 0.0,
            "max": 0.0,
            "bytes": 4,
        },

        # Keep DC + first frequency.
        "FFT 0+1": {
            "sum_sq": 0.0,
            "sum_abs": 0.0,
            "max": 0.0,
            "bytes": 8,
        },
    }

    total_values = 0

    # --------------------------------------------------------
    # Process embedding in chunks.
    # --------------------------------------------------------

    for start in range(
        0,
        tokens,
        CHUNK_SIZE
    ):

        end = min(
            start + CHUNK_SIZE,
            tokens
        )

        original = weight[
            start:end
        ]

        # Shape:
        #
        # [tokens, 768]
        #
        # -> [tokens, 192, 4]

        blocks_x = original.reshape(
            end - start,
            blocks,
            4
        )

        # ----------------------------------------------------
        # FFT
        # ----------------------------------------------------

        F = fft4(
            blocks_x
        )

        # ----------------------------------------------------
        # Method 1:
        #
        # Full FFT.
        #
        # No actual information is removed.
        # This establishes the numerical baseline.
        # ----------------------------------------------------

        reconstructed = ifft4(F)

        error = (
            reconstructed -
            blocks_x
        )

        methods["FULL FFT"]["sum_sq"] += (
            error * error
        ).sum().item()

        methods["FULL FFT"]["sum_abs"] += (
            error.abs()
        ).sum().item()

        methods["FULL FFT"]["max"] = max(
            methods["FULL FFT"]["max"],
            error.abs().max().item()
        )

        # ----------------------------------------------------
        # Method 2:
        #
        # Keep coefficients 0 and 2.
        #
        # Remove frequency 1 and 3.
        # ----------------------------------------------------

        F02 = torch.zeros_like(F)

        F02[..., 0] = F[..., 0]
        F02[..., 2] = F[..., 2]

        reconstructed = ifft4(F02)

        error = (
            reconstructed -
            blocks_x
        )

        methods["FFT 0+2"]["sum_sq"] += (
            error * error
        ).sum().item()

        methods["FFT 0+2"]["sum_abs"] += (
            error.abs()
        ).sum().item()

        methods["FFT 0+2"]["max"] = max(
            methods["FFT 0+2"]["max"],
            error.abs().max().item()
        )

        # ----------------------------------------------------
        # Method 3:
        #
        # Keep only DC.
        # ----------------------------------------------------

        F0 = torch.zeros_like(F)

        F0[..., 0] = F[..., 0]

        reconstructed = ifft4(F0)

        error = (
            reconstructed -
            blocks_x
        )

        methods["FFT 0"]["sum_sq"] += (
            error * error
        ).sum().item()

        methods["FFT 0"]["sum_abs"] += (
            error.abs()
        ).sum().item()

        methods["FFT 0"]["max"] = max(
            methods["FFT 0"]["max"],
            error.abs().max().item()
        )

        # ----------------------------------------------------
        # Method 4:
        #
        # Keep DC + frequency 1.
        # ----------------------------------------------------

        F01 = torch.zeros_like(F)

        F01[..., 0] = F[..., 0]
        F01[..., 1] = F[..., 1]
        F01[..., 3] = F[..., 3]

        reconstructed = ifft4(F01)

        error = (
            reconstructed -
            blocks_x
        )

        methods["FFT 0+1"]["sum_sq"] += (
            error * error
        ).sum().item()

        methods["FFT 0+1"]["sum_abs"] += (
            error.abs()
        ).sum().item()

        methods["FFT 0+1"]["max"] = max(
            methods["FFT 0+1"]["max"],
            error.abs().max().item()
        )

        total_values += (
            blocks_x.numel()
        )

        if start % (
            CHUNK_SIZE * 20
        ) == 0:

            print(
                f"Processed "
                f"{end:,}/{tokens:,} tokens"
            )

    # ========================================================
    # Results
    # ========================================================

    print()
    print("=" * 90)
    print("RESULTS")
    print("=" * 90)

    print()

    original_bytes_per_block = 4 * 2

    print(
        f"{'Method':<15}"
        f"{'Bytes/block':>15}"
        f"{'Bytes/token':>15}"
        f"{'Compression':>15}"
        f"{'RMSE':>15}"
        f"{'MAE':>15}"
        f"{'Max error':>15}"
    )

    print("-" * 105)

    for name, result in methods.items():

        mse = (
            result["sum_sq"] /
            total_values
        )

        rmse = math.sqrt(mse)

        mae = (
            result["sum_abs"] /
            total_values
        )

        bytes_block = result["bytes"]

        bytes_token = (
            bytes_block *
            blocks
        )

        original_token_bytes = (
            dimensions * 2
        )

        compression = (
            original_token_bytes /
            bytes_token
        )

        print(
            f"{name:<15}"
            f"{bytes_block:>15}"
            f"{bytes_token:>15}"
            f"{compression:>14.2f}x"
            f"{rmse:>15.8f}"
            f"{mae:>15.8f}"
            f"{result['max']:>15.8f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Granite 4-Element Fourier Compression Test")
    print("=" * 70)

    print()
    print("Model:")
    print(MODEL_FILE)

    key = find_embedding()

    print()
    print("Embedding:")
    print(key)

    with safe_open(
        MODEL_FILE,
        framework="pt",
        device="cpu"
    ) as f:

        shape = f.get_slice(
            key
        ).get_shape()

        tokens = shape[0]
        dimensions = shape[1]

        if NUM_TOKENS_TO_TEST is not None:
            tokens = min(
                tokens,
                NUM_TOKENS_TO_TEST
            )

        print()
        print(
            f"Shape: "
            f"{shape[0]:,} x {shape[1]}"
        )

        print(
            f"Testing {tokens:,} tokens"
        )

        # ----------------------------------------------------
        # Read only chunks.
        # ----------------------------------------------------

        class WeightReader:

            def __getitem__(
                self,
                item
            ):
                return f.get_slice(
                    key
                )[item]

        weight = WeightReader()

        # ----------------------------------------------------
        # The test function expects a tensor-like object.
        #
        # We therefore perform the actual chunk processing here.
        # ----------------------------------------------------

        # Re-implement test loop directly so we never load
        # the entire embedding matrix.

        blocks = dimensions // 4

        methods = {

            "FULL FFT": [0.0, 0.0, 0.0, 16],
            "FFT 0+2": [0.0, 0.0, 0.0, 8],
            "FFT 0":   [0.0, 0.0, 0.0, 4],
            "FFT 0+1": [0.0, 0.0, 0.0, 8],
        }

        total_values = 0

        print()
        print("Processing...")

        for start in range(
            0,
            tokens,
            CHUNK_SIZE
        ):

            end = min(
                start + CHUNK_SIZE,
                tokens
            )

            x = weight[
                start:end
            ].float()

            x = x.reshape(
                end - start,
                blocks,
                4
            )

            F = torch.fft.fft(
                x,
                dim=-1
            )

            # ------------------------------------------------
            # Full FFT
            # ------------------------------------------------

            recon = torch.fft.ifft(
                F,
                dim=-1
            ).real

            error = recon - x

            methods["FULL FFT"][0] += (
                error * error
            ).sum().item()

            methods["FULL FFT"][1] += (
                error.abs()
            ).sum().item()

            methods["FULL FFT"][2] = max(
                methods["FULL FFT"][2],
                error.abs().max().item()
            )

            # ------------------------------------------------
            # 0 + 2
            # ------------------------------------------------

            temp = torch.zeros_like(F)

            temp[..., 0] = F[..., 0]
            temp[..., 2] = F[..., 2]

            recon = torch.fft.ifft(
                temp,
                dim=-1
            ).real

            error = recon - x

            methods["FFT 0+2"][0] += (
                error * error
            ).sum().item()

            methods["FFT 0+2"][1] += (
                error.abs()
            ).sum().item()

            methods["FFT 0+2"][2] = max(
                methods["FFT 0+2"][2],
                error.abs().max().item()
            )

            # ------------------------------------------------
            # 0 only
            # ------------------------------------------------

            temp = torch.zeros_like(F)

            temp[..., 0] = F[..., 0]

            recon = torch.fft.ifft(
                temp,
                dim=-1
            ).real

            error = recon - x

            methods["FFT 0"][0] += (
                error * error
            ).sum().item()

            methods["FFT 0"][1] += (
                error.abs()
            ).sum().item()

            methods["FFT 0"][2] = max(
                methods["FFT 0"][2],
                error.abs().max().item()
            )

            # ------------------------------------------------
            # 0 + 1 + 3
            # ------------------------------------------------

            temp = torch.zeros_like(F)

            temp[..., 0] = F[..., 0]
            temp[..., 1] = F[..., 1]
            temp[..., 3] = F[..., 3]

            recon = torch.fft.ifft(
                temp,
                dim=-1
            ).real

            error = recon - x

            methods["FFT 0+1"][0] += (
                error * error
            ).sum().item()

            methods["FFT 0+1"][1] += (
                error.abs()
            ).sum().item()

            methods["FFT 0+1"][2] = max(
                methods["FFT 0+1"][2],
                error.abs().max().item()
            )

            total_values += x.numel()

            if start % (
                CHUNK_SIZE * 20
            ) == 0:

                print(
                    f"  {end:,}/{tokens:,}"
                )

    # ========================================================
    # Print final results
    # ========================================================

    print()
    print("=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)

    print()

    print(
        f"{'Method':<15}"
        f"{'B/block':>10}"
        f"{'B/token':>10}"
        f"{'Ratio':>10}"
        f"{'RMSE':>14}"
        f"{'MAE':>14}"
        f"{'Max':>14}"
    )

    print("-" * 90)

    original_per_token = dimensions * 2

    for name, values in methods.items():

        mse = (
            values[0] /
            total_values
        )

        rmse = math.sqrt(mse)

        mae = (
            values[1] /
            total_values
        )

        bytes_block = values[3]

        bytes_token = (
            blocks *
            bytes_block
        )

        ratio = (
            original_per_token /
            bytes_token
        )

        print(
            f"{name:<15}"
            f"{bytes_block:>10}"
            f"{bytes_token:>10}"
            f"{ratio:>9.2f}x"
            f"{rmse:>14.8f}"
            f"{mae:>14.8f}"
            f"{values[2]:>14.8f}"
        )

    print()
    print("Original:")
    print(
        f"  {original_per_token} bytes/token"
    )

    print(
        f"  {blocks} four-element blocks/token"
    )

    print()
    print("Note:")
    print(
        "FULL FFT should reconstruct the original "
        "values essentially perfectly."
    )


if __name__ == "__main__":
    main()
