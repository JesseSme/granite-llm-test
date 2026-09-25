#!/usr/bin/env python3

import os
import math
import json

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

# Number of embedding rows to test.
# Start small so the experiment is safe.
NUM_TOKENS = 5000

# Process rows in chunks.
CHUNK_SIZE = 1024

# Number of angle bits to test.
ANGLE_BITS = [
    4,
    6,
    8,
    10,
    12,
    14,
    16,
]

# ============================================================
# LOAD EMBEDDING
# ============================================================

def get_shape():

    with safe_open(
        MODEL_FILE,
        framework="pt",
        device="cpu",
    ) as f:

        if EMBEDDING_KEY not in f.keys():
            raise RuntimeError(
                f"Could not find {EMBEDDING_KEY}"
            )

        shape = f.get_slice(
            EMBEDDING_KEY
        ).get_shape()

    return shape


# ============================================================
# CIRCLE MAPPINGS
# ============================================================

def encode_sine(x):
    """
    x = sin(theta)

    Since sin(theta) is limited to [-1, 1], we first
    normalize the embedding values into that interval.
    """

    return torch.asin(x)


def encode_cosine(x):

    return torch.acos(x)


def encode_projection(x):
    """
    Unit circle:

        X = cos(theta)
        Y = sin(theta)

    The radial distance projected onto an axis is:

        r * cos(theta)

    with r = 1.
    """

    return torch.acos(x)


# ============================================================
# QUANTIZE ANGLE
# ============================================================

def quantize_angle(
    theta,
    bits,
):
    """
    Quantize angle to a fixed number of bits.

    Angle range:

        [-pi/2, pi/2]

    This range is sufficient for asin(x).
    """

    levels = (1 << bits) - 1

    minimum = -math.pi / 2
    maximum = math.pi / 2

    normalized = (
        theta - minimum
    ) / (
        maximum - minimum
    )

    q = torch.round(
        normalized * levels
    )

    q = torch.clamp(
        q,
        0,
        levels,
    )

    reconstructed = (
        q / levels
    ) * (
        maximum - minimum
    ) + minimum

    return reconstructed


# ============================================================
# DECODE
# ============================================================

def decode_sine(theta):

    return torch.sin(theta)


def decode_cosine(theta):

    return torch.cos(theta)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("GRANITE UNIT-CIRCLE EMBEDDING EXPERIMENT")
    print("=" * 80)

    if not os.path.exists(MODEL_FILE):

        raise FileNotFoundError(
            "\nModel file not found:\n"
            + MODEL_FILE
        )

    shape = get_shape()

    total_tokens = shape[0]
    dimensions = shape[1]

    tokens = min(
        NUM_TOKENS,
        total_tokens,
    )

    print()
    print("Model:")
    print(MODEL_FILE)

    print()
    print(
        f"Embedding shape : "
        f"{total_tokens:,} x {dimensions}"
    )

    print(
        f"Tokens tested   : "
        f"{tokens:,}"
    )

    print(
        f"Original format : BF16"
    )

    print(
        f"Original bits   : 16 bits/value"
    )

    # ========================================================
    # LOAD SAMPLE TO FIND GLOBAL RANGE
    # ========================================================

    print()
    print("=" * 80)
    print("PHASE 1: FIND EMBEDDING RANGE")
    print("=" * 80)

    global_min = float("inf")
    global_max = float("-inf")

    with safe_open(
        MODEL_FILE,
        framework="pt",
        device="cpu",
    ) as f:

        embedding = f.get_slice(
            EMBEDDING_KEY
        )

        for start in range(
            0,
            tokens,
            CHUNK_SIZE,
        ):

            end = min(
                start + CHUNK_SIZE,
                tokens,
            )

            x = embedding[
                start:end
            ].float()

            minimum = x.min().item()
            maximum = x.max().item()

            global_min = min(
                global_min,
                minimum,
            )

            global_max = max(
                global_max,
                maximum,
            )

            print(
                f"Processed "
                f"{end:,}/{tokens:,}"
            )

    print()
    print(
        f"Minimum embedding value: "
        f"{global_min:.8f}"
    )

    print(
        f"Maximum embedding value: "
        f"{global_max:.8f}"
    )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    print()
    print("=" * 80)
    print("PHASE 2: NORMALIZE ONTO UNIT CIRCLE")
    print("=" * 80)

    print()
    print(
        "A unit circle can only produce values in [-1, 1]."
    )

    print(
        "Therefore the embedding values are normalized "
        "before converting them to angles."
    )

    # ========================================================
    # RESULTS
    # ========================================================

    results = {}

    # ========================================================
    # TEST
    # ========================================================

    with safe_open(
        MODEL_FILE,
        framework="pt",
        device="cpu",
    ) as f:

        embedding = f.get_slice(
            EMBEDDING_KEY
        )

        # ----------------------------------------------------
        # Collect errors for every bit depth.
        # ----------------------------------------------------

        for bits in ANGLE_BITS:

            results[
                str(bits)
            ] = {
                "bits_per_value": bits,
                "mappings": {},
            }

        # ----------------------------------------------------
        # Accumulators
        # ----------------------------------------------------

        accumulator = {}

        for bits in ANGLE_BITS:

            accumulator[
                bits
            ] = {}

            for mapping in [
                "sine",
                "cosine",
            ]:

                accumulator[
                    bits
                ][mapping] = {
                    "squared_error": 0.0,
                    "absolute_error": 0.0,
                    "max_error": 0.0,
                    "count": 0,
                    "exact": 0,
                }

        # ====================================================
        # PROCESS
        # ====================================================

        print()
        print("=" * 80)
        print("PHASE 3: ANGLE ENCODING")
        print("=" * 80)

        for start in range(
            0,
            tokens,
            CHUNK_SIZE,
        ):

            end = min(
                start + CHUNK_SIZE,
                tokens,
            )

            original = embedding[
                start:end
            ].float()

            # ------------------------------------------------
            # Normalize embedding into [-1,1].
            #
            # IMPORTANT:
            #
            # We use the largest absolute value so that zero
            # remains zero.
            # ------------------------------------------------

            scale = max(
                abs(global_min),
                abs(global_max),
            )

            normalized = (
                original /
                scale
            )

            normalized = torch.clamp(
                normalized,
                -1.0,
                1.0,
            )

            # =================================================
            # SINE REPRESENTATION
            # =================================================

            theta_sine = encode_sine(
                normalized
            )

            # =================================================
            # COSINE REPRESENTATION
            # =================================================

            theta_cosine = encode_cosine(
                normalized
            )

            # =================================================
            # TEST ANGLE PRECISION
            # =================================================

            for bits in ANGLE_BITS:

                # ------------------------------------------------
                # SINE
                # ------------------------------------------------

                q_sine = quantize_angle(
                    theta_sine,
                    bits,
                )

                reconstructed_sine = (
                    decode_sine(
                        q_sine
                    )
                    * scale
                )

                error = (
                    reconstructed_sine -
                    original
                )

                acc = accumulator[
                    bits
                ]["sine"]

                acc[
                    "squared_error"
                ] += (
                    error *
                    error
                ).sum().item()

                acc[
                    "absolute_error"
                ] += (
                    error.abs()
                ).sum().item()

                acc[
                    "max_error"
                ] = max(
                    acc["max_error"],
                    error.abs().max().item(),
                )

                acc[
                    "count"
                ] += error.numel()

                acc[
                    "exact"
                ] += (
                    reconstructed_sine
                    .to(torch.bfloat16)
                    ==
                    original.to(torch.bfloat16)
                ).sum().item()

                # ------------------------------------------------
                # COSINE
                # ------------------------------------------------

                q_cosine = quantize_angle(
                    theta_cosine,
                    bits,
                )

                reconstructed_cosine = (
                    decode_cosine(
                        q_cosine
                    )
                    * scale
                )

                error = (
                    reconstructed_cosine -
                    original
                )

                acc = accumulator[
                    bits
                ]["cosine"]

                acc[
                    "squared_error"
                ] += (
                    error *
                    error
                ).sum().item()

                acc[
                    "absolute_error"
                ] += (
                    error.abs()
                ).sum().item()

                acc[
                    "max_error"
                ] = max(
                    acc["max_error"],
                    error.abs().max().item(),
                )

                acc[
                    "count"
                ] += error.numel()

                acc[
                    "exact"
                ] += (
                    reconstructed_cosine
                    .to(torch.bfloat16)
                    ==
                    original.to(torch.bfloat16)
                ).sum().item()

            if (
                start == 0 or
                start % (
                    CHUNK_SIZE * 10
                ) == 0
            ):

                print(
                    f"Processed "
                    f"{end:,}/{tokens:,}"
                )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 100)
    print("RESULTS")
    print("=" * 100)

    for bits in ANGLE_BITS:

        print()
        print(
            f"ANGLE PRECISION: {bits} bits"
        )

        print(
            "-" * 100
        )

        for mapping in [
            "sine",
            "cosine",
        ]:

            acc = accumulator[
                bits
            ][mapping]

            count = acc[
                "count"
            ]

            mse = (
                acc[
                    "squared_error"
                ]
                / count
            )

            rmse = math.sqrt(
                mse
            )

            mae = (
                acc[
                    "absolute_error"
                ]
                / count
            )

            exact = (
                acc[
                    "exact"
                ]
                /
                count
                *
                100
            )

            print()
            print(
                f"{mapping.upper()}"
            )

            print(
                f"  Bits/value       : "
                f"{bits}"
            )

            print(
                f"  RMSE             : "
                f"{rmse:.10f}"
            )

            print(
                f"  MAE              : "
                f"{mae:.10f}"
            )

            print(
                f"  Maximum error    : "
                f"{acc['max_error']:.10f}"
            )

            print(
                f"  Exact BF16       : "
                f"{exact:.4f}%"
            )

            original_bits = 16
            compression = (
                original_bits /
                bits
            )

            print(
                f"  Theoretical ratio: "
                f"{compression:.2f}x"
            )

            results[
                str(bits)
            ][
                "mappings"
            ][
                mapping
            ] = {

                "rmse":
                    rmse,

                "mae":
                    mae,

                "max_error":
                    acc[
                        "max_error"
                    ],

                "exact_bf16_percent":
                    exact,

                "theoretical_ratio":
                    compression,
            }

    # ========================================================
    # SAVE
    # ========================================================

    output = {

        "model_file":
            MODEL_FILE,

        "embedding_key":
            EMBEDDING_KEY,

        "embedding_shape":
            [
                total_tokens,
                dimensions,
            ],

        "tokens_tested":
            tokens,

        "global_min":
            global_min,

        "global_max":
            global_max,

        "scale":
            max(
                abs(global_min),
                abs(global_max),
            ),

        "results":
            results,
    }

    with open(
        "circle_embedding_results.json",
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
            "circle_embedding_results.json"
        )
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
