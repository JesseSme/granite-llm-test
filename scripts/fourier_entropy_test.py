#!/usr/bin/env python3

import os
import math
import json
from collections import Counter

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

# Number of tokens loaded at once.
# Keep this relatively small for WSL.
CHUNK_SIZE = 2048

# None = test the complete vocabulary.
# Example: 1000 = only test the first 1000 tokens.
NUM_TOKENS_TO_TEST = None

# Block sizes to test.
BLOCK_SIZES = [2, 4, 8, 16, 32]

# Number of Fourier scalar values used for
# the distribution/quantization experiment.
MAX_SAMPLE_VALUES = 2_000_000

# Histogram resolution.
HISTOGRAM_BINS = 4096

OUTPUT_FILE = "fourier_entropy_results.json"


# ============================================================
# FIND EMBEDDING
# ============================================================

def find_embedding_key():

    with safe_open(
        MODEL_FILE,
        framework="pt",
        device="cpu",
    ) as f:

        for key in f.keys():

            if key.endswith("embed_tokens.weight"):
                return key

    raise RuntimeError(
        "Could not find model.embed_tokens.weight"
    )


# ============================================================
# DISCRETE ENTROPY
# ============================================================

def entropy_from_counter(counter):

    total = sum(counter.values())

    if total == 0:
        return 0.0

    entropy = 0.0

    for count in counter.values():

        probability = count / total

        entropy -= (
            probability *
            math.log2(probability)
        )

    return entropy


# ============================================================
# HISTOGRAM ENTROPY
# ============================================================

def histogram_entropy(
    values,
    bins=HISTOGRAM_BINS,
):
    """
    Estimate entropy of a continuous distribution.

    This is useful for comparing the original and Fourier
    distributions.

    It is NOT an exact prediction of compressed file size.
    """

    values = (
        values
        .detach()
        .cpu()
        .float()
        .reshape(-1)
    )

    if values.numel() == 0:
        return 0.0, 0.0, 0.0

    minimum = values.min().item()
    maximum = values.max().item()

    if minimum == maximum:

        return (
            0.0,
            minimum,
            maximum,
        )

    histogram = torch.histc(
        values,
        bins=bins,
        min=minimum,
        max=maximum,
    )

    histogram = histogram[
        histogram > 0
    ]

    probabilities = (
        histogram /
        histogram.sum()
    )

    entropy = -(
        probabilities *
        torch.log2(probabilities)
    ).sum().item()

    return (
        entropy,
        minimum,
        maximum,
    )


# ============================================================
# UNIFORM QUANTIZATION
# ============================================================

def quantize(
    values,
    bits,
    minimum,
    maximum,
):
    """
    Quantize values to a fixed number of bits
    and reconstruct them.

    This is deliberately simple. The goal is to determine
    whether Fourier coefficients tolerate aggressive
    quantization.
    """

    levels = (1 << bits) - 1

    if maximum == minimum:

        reconstructed = torch.full_like(
            values,
            minimum,
        )

        return reconstructed

    normalized = (
        (values - minimum) /
        (maximum - minimum)
    )

    quantized = torch.round(
        normalized * levels
    )

    quantized = torch.clamp(
        quantized,
        0,
        levels,
    )

    reconstructed = (
        quantized / levels
    ) * (
        maximum - minimum
    ) + minimum

    return reconstructed


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("GRANITE FOURIER ENTROPY EXPERIMENT")
    print("=" * 80)

    if not os.path.exists(MODEL_FILE):

        raise FileNotFoundError(
            "\nCould not find model file:\n"
            + MODEL_FILE
        )

    print()
    print("Model:")
    print(MODEL_FILE)

    embedding_key = find_embedding_key()

    print()
    print("Embedding:")
    print(embedding_key)

    # --------------------------------------------------------
    # Open model without loading entire matrix.
    # --------------------------------------------------------

    with safe_open(
        MODEL_FILE,
        framework="pt",
        device="cpu",
    ) as f:

        shape = f.get_slice(
            embedding_key
        ).get_shape()

        total_tokens = shape[0]
        dimensions = shape[1]

        tokens = total_tokens

        if NUM_TOKENS_TO_TEST is not None:

            tokens = min(
                tokens,
                NUM_TOKENS_TO_TEST,
            )

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
            f"Block sizes     : "
            f"{BLOCK_SIZES}"
        )

        print()
        print(
            "The FFT itself is lossless."
        )

        print(
            "We are testing whether the transformed values "
            "can be encoded more efficiently."
        )

        # ====================================================
        # ORIGINAL BF16 VALUE DISTRIBUTION
        # ====================================================

        print()
        print("=" * 80)
        print("PHASE 1: ORIGINAL BF16 DISTRIBUTION")
        print("=" * 80)

        original_counter = Counter()

        total_values = 0

        for start in range(
            0,
            tokens,
            CHUNK_SIZE,
        ):

            end = min(
                start + CHUNK_SIZE,
                tokens,
            )

            tensor = f.get_slice(
                embedding_key
            )[start:end]

            # Preserve exact BF16 bit patterns.
            raw = tensor.view(
                torch.uint16
            ).reshape(-1)

            original_counter.update(
                raw.tolist()
            )

            total_values += raw.numel()

            if (
                start == 0 or
                start % (CHUNK_SIZE * 20) == 0
            ):

                print(
                    f"Processed "
                    f"{end:,}/{tokens:,}"
                )

        original_entropy = entropy_from_counter(
            original_counter
        )

        print()
        print(
            f"Unique BF16 bit patterns : "
            f"{len(original_counter):,}"
        )

        print(
            f"Entropy of BF16 values   : "
            f"{original_entropy:.6f} bits/value"
        )

        print(
            f"Nominal BF16 size        : "
            f"16 bits/value"
        )

        print(
            f"Estimated entropy saving : "
            f"{16 - original_entropy:.6f} bits/value"
        )

        # ====================================================
        # FOURIER EXPERIMENT
        # ====================================================

        results = {}

        for block_size in BLOCK_SIZES:

            if dimensions % block_size != 0:

                print()
                print(
                    f"Skipping block size "
                    f"{block_size}: "
                    f"{dimensions} is not divisible by it."
                )

                continue

            results[
                block_size
            ] = {
                "blocks_per_token":
                    dimensions // block_size,

                "original": {},

                "fourier": {},

                "quantization": {},
            }

        print()
        print("=" * 80)
        print("PHASE 2: FOURIER TRANSFORM")
        print("=" * 80)

        # ----------------------------------------------------
        # Samples for Fourier distributions.
        # ----------------------------------------------------

        fourier_samples = {}

        for block_size in results:

            fourier_samples[
                block_size
            ] = {
                "real": [],
                "imag": [],
            }

        fourier_sample_counts = {}

        for block_size in results:

            fourier_sample_counts[
                block_size
            ] = {
                "real": 0,
                "imag": 0,
            }

        # ----------------------------------------------------
        # Process model in chunks.
        # ----------------------------------------------------

        for start in range(
            0,
            tokens,
            CHUNK_SIZE,
        ):

            end = min(
                start + CHUNK_SIZE,
                tokens,
            )

            tensor = f.get_slice(
                embedding_key
            )[start:end]

            # Convert BF16 -> FP32 for FFT.
            #
            # This conversion is exact because every BF16
            # number can be represented exactly in FP32.
            x = tensor.float()

            for block_size in results:

                blocks_per_token = (
                    dimensions //
                    block_size
                )

                blocks = x.reshape(
                    end - start,
                    blocks_per_token,
                    block_size,
                )

                # ------------------------------------------------
                # FFT
                # ------------------------------------------------

                transformed = torch.fft.fft(
                    blocks,
                    dim=-1,
                )

                # ------------------------------------------------
                # Because the original values are real,
                # the FFT has conjugate symmetry.
                #
                # rfft contains only the unique half-spectrum.
                #
                # We are NOT discarding information here.
                # We are simply using the mathematically
                # redundant half only once.
                # ------------------------------------------------

                unique = torch.fft.rfft(
                    blocks,
                    dim=-1,
                )

                real = (
                    unique.real
                    .reshape(-1)
                    .cpu()
                )

                imag = (
                    unique.imag
                    .reshape(-1)
                    .cpu()
                )

                for name, values in [
                    ("real", real),
                    ("imag", imag),
                ]:

                    current_count = (
                        fourier_sample_counts[
                            block_size
                        ][name]
                    )

                    remaining = (
                        MAX_SAMPLE_VALUES -
                        current_count
                    )

                    if remaining <= 0:
                        continue

                    take = min(
                        remaining,
                        values.numel(),
                    )

                    if take > 0:

                        fourier_samples[
                            block_size
                        ][name].append(
                            values[:take]
                        )

                        fourier_sample_counts[
                            block_size
                        ][name] += take

            if (
                start == 0 or
                start % (CHUNK_SIZE * 20) == 0
            ):

                print(
                    f"Processed "
                    f"{end:,}/{tokens:,}"
                )

        # ====================================================
        # ANALYZE FOURIER DISTRIBUTIONS
        # ====================================================

        print()
        print("=" * 80)
        print("PHASE 3: FOURIER ENTROPY")
        print("=" * 80)

        for block_size in results:

            print()
            print(
                f"Block size: {block_size}"
            )

            for component in [
                "real",
                "imag",
            ]:

                if not fourier_samples[
                    block_size
                ][component]:

                    continue

                values = torch.cat(
                    fourier_samples[
                        block_size
                    ][component]
                )

                entropy, minimum, maximum = (
                    histogram_entropy(
                        values
                    )
                )

                results[
                    block_size
                ][
                    "fourier"
                ][
                    component
                ] = {
                    "sample_values":
                        int(values.numel()),

                    "histogram_entropy_bits":
                        entropy,

                    "minimum":
                        minimum,

                    "maximum":
                        maximum,
                }

                print(
                    f"  {component:5s} "
                    f"entropy = "
                    f"{entropy:.6f} bits/value"
                )

                print(
                    f"         range = "
                    f"[{minimum:.6f}, "
                    f"{maximum:.6f}]"
                )

        # ====================================================
        # QUANTIZATION EXPERIMENT
        # ====================================================

        print()
        print("=" * 80)
        print("PHASE 4: FOURIER QUANTIZATION")
        print("=" * 80)

        print()
        print(
            "This deliberately reduces the precision of "
            "the Fourier coefficients."
        )

        print(
            "The RMSE tells us how much information is lost."
        )

        for block_size in results:

            all_components = []

            for component in [
                "real",
                "imag",
            ]:

                if not fourier_samples[
                    block_size
                ][component]:

                    continue

                all_components.append(
                    torch.cat(
                        fourier_samples[
                            block_size
                        ][component]
                    )
                )

            if not all_components:
                continue

            values = torch.cat(
                all_components
            )

            minimum = values.min().item()
            maximum = values.max().item()

            print()
            print(
                f"Block size {block_size}"
            )

            for bits in [
                4,
                6,
                8,
                10,
                12,
                14,
                16,
            ]:

                reconstructed = quantize(
                    values,
                    bits,
                    minimum,
                    maximum,
                )

                error = (
                    reconstructed -
                    values
                )

                rmse = torch.sqrt(
                    torch.mean(
                        error * error
                    )
                ).item()

                mae = (
                    error.abs()
                    .mean()
                    .item()
                )

                print(
                    f"  {bits:2d} bits/scalar"
                    f"  RMSE={rmse:.8f}"
                    f"  MAE={mae:.8f}"
                )

                results[
                    block_size
                ][
                    "quantization"
                ][
                    str(bits)
                ] = {
                    "bits_per_scalar":
                        bits,

                    "rmse":
                        rmse,

                    "mae":
                        mae,
                }

        # ====================================================
        # SUMMARY
        # ====================================================

        print()
        print("=" * 100)
        print("SUMMARY")
        print("=" * 100)

        print()

        print(
            f"{'Block':>8}"
            f"{'Orig H':>14}"
            f"{'FFT Re H':>14}"
            f"{'FFT Im H':>14}"
            f"{'Original':>14}"
            f"{'FFT unique':>14}"
        )

        print("-" * 100)

        for block_size, result in results.items():

            real_entropy = result[
                "fourier"
            ].get(
                "real",
                {}
            ).get(
                "histogram_entropy_bits",
                float("nan"),
            )

            imag_entropy = result[
                "fourier"
            ].get(
                "imag",
                {}
            ).get(
                "histogram_entropy_bits",
                float("nan"),
            )

            # Original entropy per scalar.
            original_h = (
                original_entropy
            )

            # Number of real scalars required by rFFT.
            #
            # rfft has floor(N/2)+1 complex values.
            # Each complex value has real + imaginary parts.
            unique_complex = (
                block_size // 2 + 1
            )

            fft_scalars = (
                unique_complex * 2
            )

            print(
                f"{block_size:>8}"
                f"{original_h:>14.4f}"
                f"{real_entropy:>14.4f}"
                f"{imag_entropy:>14.4f}"
                f"{block_size:>14}"
                f"{fft_scalars:>14}"
            )

        # ====================================================
        # SAVE
        # ====================================================

        output = {
            "model": MODEL_FILE,
            "embedding_key": embedding_key,
            "embedding_shape": [
                total_tokens,
                dimensions,
            ],
            "tokens_tested": tokens,
            "chunk_size": CHUNK_SIZE,
            "original_bf16_entropy_bits_per_value":
                original_entropy,
            "results": results,
        }

        with open(
            OUTPUT_FILE,
            "w",
        ) as out:

            json.dump(
                output,
                out,
                indent=2,
            )

        print()
        print("=" * 80)
        print("DONE")
        print("=" * 80)

        print()
        print(
            f"Results saved to:"
        )

        print(
            os.path.abspath(
                OUTPUT_FILE
            )
        )

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "The entropy numbers are statistical estimates."
        )

        print(
            "They are not the size of a finished compressed file."
        )

        print(
            "If FFT shows a lower-entropy representation, "
            "the next test should implement an actual entropy "
            "coder and measure the resulting bytes."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
