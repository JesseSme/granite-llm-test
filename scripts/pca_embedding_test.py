#!/usr/bin/env python3

import os
import json
import math

import torch
from safetensors import safe_open


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "granite-4.0-h-350m",
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "model.safetensors",
)

EMBEDDING_KEY = "model.embed_tokens.weight"

# Process this many embedding rows at once.
# Lower this if WSL runs out of memory.
CHUNK_SIZE = 1024

# PCA ranks to test.
PCA_RANKS = [
    8,
    16,
    32,
    64,
    128,
    256,
    384,
    512,
]

# Number of tokens used to train PCA.
#
# None = all tokens.
#
# If memory/time becomes a problem, try:
# 10000
# 20000
# 50000
PCA_TRAIN_TOKENS = 20000

# Number of tokens used to evaluate reconstruction.
#
# None = all tokens.
#
# Start with 5000 if you want a fast test.
TEST_TOKENS = 5000

# Output file.
OUTPUT_FILE = "pca_embedding_results.json"


# ============================================================
# LOAD EMBEDDING SHAPE
# ============================================================

def get_embedding_shape():

    with safe_open(
        MODEL_FILE,
        framework="pt",
        device="cpu",
    ) as f:

        if EMBEDDING_KEY not in f.keys():

            print("Available tensors:")

            for key in f.keys():
                print(" ", key)

            raise RuntimeError(
                f"\nCould not find {EMBEDDING_KEY}"
            )

        shape = f.get_slice(
            EMBEDDING_KEY
        ).get_shape()

    return shape


# ============================================================
# LOAD CHUNK
# ============================================================

def load_chunk(
    tensor_slice,
    start,
    end,
):
    """
    Loads one chunk and converts BF16 -> FP32.

    BF16 -> FP32 is exact.
    """

    return tensor_slice[
        start:end
    ].float()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("GRANITE EMBEDDING PCA COMPRESSION TEST")
    print("=" * 80)

    if not os.path.exists(MODEL_FILE):

        raise FileNotFoundError(
            "\nModel file not found:\n"
            + MODEL_FILE
        )

    print()
    print("Model:")
    print(MODEL_FILE)

    # --------------------------------------------------------
    # Find shape
    # --------------------------------------------------------

    shape = get_embedding_shape()

    num_tokens = shape[0]
    dimensions = shape[1]

    train_tokens = num_tokens

    if PCA_TRAIN_TOKENS is not None:

        train_tokens = min(
            PCA_TRAIN_TOKENS,
            num_tokens,
        )

    test_tokens = num_tokens

    if TEST_TOKENS is not None:

        test_tokens = min(
            TEST_TOKENS,
            num_tokens,
        )

    print()
    print(
        f"Embedding shape : "
        f"{num_tokens:,} x {dimensions}"
    )

    print(
        f"PCA training     : "
        f"{train_tokens:,} tokens"
    )

    print(
        f"Testing          : "
        f"{test_tokens:,} tokens"
    )

    print(
        f"Chunk size       : "
        f"{CHUNK_SIZE}"
    )

    print()
    print(
        "Original storage:"
    )

    original_bytes = (
        num_tokens *
        dimensions *
        2
    )

    print(
        f"  {original_bytes:,} bytes"
    )

    print(
        f"  {original_bytes / 1024 / 1024:.2f} MiB"
    )

    # ========================================================
    # OPEN MODEL
    # ========================================================

    with safe_open(
        MODEL_FILE,
        framework="pt",
        device="cpu",
    ) as f:

        embedding = f.get_slice(
            EMBEDDING_KEY
        )

        # ====================================================
        # PHASE 1
        # Calculate mean
        # ====================================================

        print()
        print("=" * 80)
        print("PHASE 1: CALCULATING MEAN")
        print("=" * 80)

        mean = torch.zeros(
            dimensions,
            dtype=torch.float64,
        )

        count = 0

        for start in range(
            0,
            train_tokens,
            CHUNK_SIZE,
        ):

            end = min(
                start + CHUNK_SIZE,
                train_tokens,
            )

            x = embedding[
                start:end
            ].float()

            mean += x.double().sum(
                dim=0
            )

            count += x.shape[0]

            if (
                start == 0 or
                start % (
                    CHUNK_SIZE * 10
                ) == 0
            ):

                print(
                    f"Processed "
                    f"{end:,}/{train_tokens:,}"
                )

        mean /= count

        mean = mean.float()

        print()
        print("Mean calculated.")

        # ====================================================
        # PHASE 2
        # Build covariance matrix
        # ====================================================

        print()
        print("=" * 80)
        print("PHASE 2: CALCULATING COVARIANCE MATRIX")
        print("=" * 80)

        covariance = torch.zeros(
            dimensions,
            dimensions,
            dtype=torch.float64,
        )

        count = 0

        for start in range(
            0,
            train_tokens,
            CHUNK_SIZE,
        ):

            end = min(
                start + CHUNK_SIZE,
                train_tokens,
            )

            x = embedding[
                start:end
            ].float()

            x = x - mean

            covariance += (
                x.double().T @
                x.double()
            )

            count += x.shape[0]

            if (
                start == 0 or
                start % (
                    CHUNK_SIZE * 10
                ) == 0
            ):

                print(
                    f"Processed "
                    f"{end:,}/{train_tokens:,}"
                )

        covariance /= (
            count - 1
        )

        print()
        print(
            "Covariance matrix calculated."
        )

        # ====================================================
        # PHASE 3
        # Eigen decomposition
        # ====================================================

        print()
        print("=" * 80)
        print("PHASE 3: COMPUTING PCA")
        print("=" * 80)

        print()
        print(
            "Computing eigenvectors..."
        )

        eigenvalues, eigenvectors = (
            torch.linalg.eigh(
                covariance
            )
        )

        # eigh returns smallest -> largest.
        # Reverse them.

        eigenvalues = torch.flip(
            eigenvalues,
            dims=[0],
        )

        eigenvectors = torch.flip(
            eigenvectors,
            dims=[1],
        )

        eigenvalues = eigenvalues.float()
        eigenvectors = eigenvectors.float()

        print(
            "PCA calculated."
        )

        # ----------------------------------------------------
        # Explained variance
        # ----------------------------------------------------

        total_variance = (
            eigenvalues.sum().item()
        )

        print()
        print(
            "Explained variance:"
        )

        for rank in PCA_RANKS:

            if rank > dimensions:
                continue

            explained = (
                eigenvalues[:rank].sum()
                .item()
                /
                total_variance
            )

            print(
                f"  rank {rank:4d}: "
                f"{explained * 100:.4f}%"
            )

        # ====================================================
        # PHASE 4
        # Test reconstruction
        # ====================================================

        print()
        print("=" * 80)
        print("PHASE 4: RECONSTRUCTION TEST")
        print("=" * 80)

        results = {}

        for rank in PCA_RANKS:

            if rank > dimensions:
                continue

            print()
            print("-" * 80)

            print(
                f"Testing PCA rank {rank}"
            )

            # ------------------------------------------------
            # PCA basis
            # ------------------------------------------------

            basis = eigenvectors[
                :, :rank
            ]

            # ------------------------------------------------
            # Storage calculation
            # ------------------------------------------------

            # Each token stores 'rank' coefficients.
            #
            # Here we calculate both BF16 and FP32 versions.
            #
            # The basis and mean are shared globally.
            #

            token_bytes_bf16 = (
                rank * 2
            )

            token_bytes_fp16 = (
                rank * 2
            )

            token_bytes_fp32 = (
                rank * 4
            )

            basis_bytes_bf16 = (
                dimensions *
                rank *
                2
            )

            basis_bytes_fp32 = (
                dimensions *
                rank *
                4
            )

            mean_bytes = (
                dimensions * 2
            )

            compressed_bf16 = (
                num_tokens *
                token_bytes_bf16
                +
                basis_bytes_bf16
                +
                mean_bytes
            )

            compressed_fp32 = (
                num_tokens *
                token_bytes_fp32
                +
                basis_bytes_fp32
                +
                mean_bytes
            )

            ratio_bf16 = (
                original_bytes /
                compressed_bf16
            )

            ratio_fp32 = (
                original_bytes /
                compressed_fp32
            )

            # ------------------------------------------------
            # Reconstruction statistics
            # ------------------------------------------------

            total_squared_error = 0.0
            total_absolute_error = 0.0

            max_error = 0.0

            cosine_sum = 0.0

            exact_bf16 = 0

            total_test_values = (
                test_tokens *
                dimensions
            )

            # ------------------------------------------------
            # Test chunks
            # ------------------------------------------------

            for start in range(
                0,
                test_tokens,
                CHUNK_SIZE,
            ):

                end = min(
                    start + CHUNK_SIZE,
                    test_tokens,
                )

                x = embedding[
                    start:end
                ].float()

                centered = (
                    x - mean
                )

                # ------------------------------------------------
                # Encode into PCA coefficients.
                # ------------------------------------------------

                coefficients = (
                    centered @ basis
                )

                # ------------------------------------------------
                # Decode.
                # ------------------------------------------------

                reconstructed = (
                    coefficients @
                    basis.T
                    +
                    mean
                )

                error = (
                    reconstructed -
                    x
                )

                squared = (
                    error *
                    error
                )

                absolute = (
                    error.abs()
                )

                total_squared_error += (
                    squared.sum().item()
                )

                total_absolute_error += (
                    absolute.sum().item()
                )

                chunk_max = (
                    absolute.max().item()
                )

                if chunk_max > max_error:
                    max_error = chunk_max

                # ------------------------------------------------
                # Cosine similarity.
                # ------------------------------------------------

                dot = (
                    x *
                    reconstructed
                ).sum(dim=1)

                norm_x = torch.linalg.vector_norm(
                    x,
                    dim=1,
                )

                norm_reconstructed = (
                    torch.linalg.vector_norm(
                        reconstructed,
                        dim=1,
                    )
                )

                cosine = (
                    dot /
                    (
                        norm_x *
                        norm_reconstructed
                        + 1e-12
                    )
                )

                cosine_sum += (
                    cosine.sum().item()
                )

                # ------------------------------------------------
                # Exact BF16 reconstruction.
                #
                # Convert both to BF16 and compare bit-for-bit.
                # ------------------------------------------------

                original_bf16 = (
                    x.to(torch.bfloat16)
                )

                reconstructed_bf16 = (
                    reconstructed.to(
                        torch.bfloat16
                    )
                )

                exact_bf16 += (
                    (
                        original_bf16 ==
                        reconstructed_bf16
                    )
                    .sum()
                    .item()
                )

                if (
                    start == 0 or
                    start % (
                        CHUNK_SIZE * 20
                    ) == 0
                ):

                    print(
                        f"  Testing "
                        f"{end:,}/{test_tokens:,}"
                    )

            # ------------------------------------------------
            # Final statistics
            # ------------------------------------------------

            mse = (
                total_squared_error /
                total_test_values
            )

            rmse = math.sqrt(
                mse
            )

            mae = (
                total_absolute_error /
                total_test_values
            )

            cosine_average = (
                cosine_sum /
                test_tokens
            )

            exact_percentage = (
                exact_bf16 /
                total_test_values *
                100
            )

            explained_variance = (
                eigenvalues[:rank].sum()
                .item()
                /
                total_variance
            )

            results[
                str(rank)
            ] = {

                "rank": rank,

                "explained_variance":
                    explained_variance,

                "rmse":
                    rmse,

                "mae":
                    mae,

                "max_absolute_error":
                    max_error,

                "average_cosine_similarity":
                    cosine_average,

                "exact_bf16_percentage":
                    exact_percentage,

                "storage": {

                    "original_bytes":
                        original_bytes,

                    "compressed_bf16_bytes":
                        compressed_bf16,

                    "compressed_fp32_bytes":
                        compressed_fp32,

                    "compression_ratio_bf16":
                        ratio_bf16,

                    "compression_ratio_fp32":
                        ratio_fp32,

                    "token_bytes_bf16":
                        token_bytes_bf16,

                    "token_bytes_fp32":
                        token_bytes_fp32,

                    "basis_bytes_bf16":
                        basis_bytes_bf16,

                    "basis_bytes_fp32":
                        basis_bytes_fp32,
                },
            }

            print()
            print(
                f"Explained variance : "
                f"{explained_variance * 100:.4f}%"
            )

            print(
                f"RMSE               : "
                f"{rmse:.8f}"
            )

            print(
                f"MAE                : "
                f"{mae:.8f}"
            )

            print(
                f"Maximum error      : "
                f"{max_error:.8f}"
            )

            print(
                f"Cosine similarity  : "
                f"{cosine_average:.8f}"
            )

            print(
                f"Exact BF16         : "
                f"{exact_percentage:.4f}%"
            )

            print()
            print(
                f"BF16 compressed    : "
                f"{compressed_bf16 / 1024 / 1024:.2f} MiB"
            )

            print(
                f"BF16 ratio         : "
                f"{ratio_bf16:.2f}x"
            )

            print(
                f"FP32 compressed    : "
                f"{compressed_fp32 / 1024 / 1024:.2f} MiB"
            )

            print(
                f"FP32 ratio         : "
                f"{ratio_fp32:.2f}x"
            )

        # ====================================================
        # SUMMARY
        # ====================================================

        print()
        print("=" * 110)
        print("FINAL SUMMARY")
        print("=" * 110)

        print()

        print(
            f"{'Rank':>6}"
            f"{'Variance':>12}"
            f"{'RMSE':>14}"
            f"{'Cosine':>12}"
            f"{'Exact BF16':>14}"
            f"{'MiB':>12}"
            f"{'Ratio':>10}"
        )

        print("-" * 110)

        for rank in PCA_RANKS:

            if str(rank) not in results:
                continue

            r = results[
                str(rank)
            ]

            storage = r[
                "storage"
            ]

            print(
                f"{rank:>6}"
                f"{r['explained_variance'] * 100:>11.3f}%"
                f"{r['rmse']:>14.7f}"
                f"{r['average_cosine_similarity']:>12.7f}"
                f"{r['exact_bf16_percentage']:>13.4f}%"
                f"{storage['compressed_bf16_bytes'] / 1024 / 1024:>12.2f}"
                f"{storage['compression_ratio_bf16']:>10.2f}x"
            )

        # ====================================================
        # SAVE RESULTS
        # ====================================================

        output = {

            "model_file":
                MODEL_FILE,

            "embedding_key":
                EMBEDDING_KEY,

            "embedding_shape":
                [
                    num_tokens,
                    dimensions,
                ],

            "pca_training_tokens":
                train_tokens,

            "test_tokens":
                test_tokens,

            "chunk_size":
                CHUNK_SIZE,

            "original_storage_bytes":
                original_bytes,

            "results":
                results,
        }

        with open(
            OUTPUT_FILE,
            "w",
        ) as file:

            json.dump(
                output,
                file,
                indent=2,
            )

        print()
        print("=" * 80)
        print("DONE")
        print("=" * 80)

        print()
        print(
            "Results saved to:"
        )

        print(
            os.path.abspath(
                OUTPUT_FILE
            )
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
