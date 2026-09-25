#!/usr/bin/env python3

import os
import math
import json
from collections import Counter

import torch
from safetensors import safe_open


MODEL_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "granite-4.0-h-350m",
    "model.safetensors",
)

EMBEDDING_KEY = "model.embed_tokens.weight"

# Process one embedding vector at a time.
# Change to a smaller number if desired.
PROGRESS_EVERY = 1000


def entropy(counter, total):
    if total == 0:
        return 0.0

    h = 0.0

    for count in counter.values():
        p = count / total
        h -= p * math.log2(p)

    return h


def zigzag(x):
    if x >= 0:
        return x * 2
    return (-x * 2) - 1


def update_counter(counter, value):
    counter[value] += 1


def print_stats(name, counter, total, nominal_bits):
    h = entropy(counter, total)

    print(f"{name}")
    print(f"  Entropy       : {h:.6f} bits/value")
    print(f"  Unique values : {len(counter):,}")
    print(f"  Nominal       : {nominal_bits} bits/value")
    print(f"  Saving        : {nominal_bits - h:.6f} bits/value")

    return h


def main():

    print("=" * 80)
    print("GRANITE BF16 PER-VECTOR DELTA COMPRESSION TEST")
    print("=" * 80)

    print()
    print("Model:")
    print(MODEL_FILE)

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(MODEL_FILE)

    # ---------------------------------------------------------
    # Read metadata without loading the model.
    # ---------------------------------------------------------

    with safe_open(
        MODEL_FILE,
        framework="pt",
        device="cpu",
    ) as f:

        shape = f.get_slice(
            EMBEDDING_KEY
        ).get_shape()

    tokens = shape[0]
    dimensions = shape[1]

    total_values = tokens * dimensions

    print()
    print(
        f"Embedding shape : {tokens:,} x {dimensions}"
    )

    print(
        f"Values tested   : {total_values:,}"
    )

    print(
        f"Processing      : ONE VECTOR AT A TIME"
    )

    original_bytes = total_values * 2

    print()
    print(
        f"Original size   : "
        f"{original_bytes / 1024 / 1024:.2f} MiB"
    )

    # ---------------------------------------------------------
    # Counters.
    #
    # IMPORTANT:
    # We only keep frequency tables.
    # We do NOT keep all embedding values.
    # ---------------------------------------------------------

    sign_counter = Counter()

    exponent_counter = Counter()
    mantissa_counter = Counter()

    exponent_delta_counter = Counter()
    mantissa_delta_counter = Counter()

    exponent_delta2_counter = Counter()
    mantissa_delta2_counter = Counter()

    # Global counters for all BF16 bit patterns.
    full_bf16_counter = Counter()

    # Number of values processed.
    processed = 0

    print()
    print("=" * 80)
    print("PROCESSING EMBEDDINGS")
    print("=" * 80)

    # ---------------------------------------------------------
    # Open safetensors.
    #
    # Only ONE embedding vector is materialized at a time.
    # ---------------------------------------------------------

    with safe_open(
        MODEL_FILE,
        framework="pt",
        device="cpu",
    ) as f:

        tensor_slice = f.get_slice(
            EMBEDDING_KEY
        )

        for token_id in range(tokens):

            # -------------------------------------------------
            # Read exactly ONE vector.
            # Shape = [768]
            # -------------------------------------------------

            vector = tensor_slice[
                token_id:token_id + 1
            ]

            vector = vector.reshape(-1)

            # Convert BF16 bit patterns to int32.
            #
            # int32 is intentional because some PyTorch
            # bitwise operations are not implemented for UInt16.
            # -------------------------------------------------

            bits = vector.view(
                torch.uint16
            ).to(torch.int32)

            # -------------------------------------------------
            # Extract BF16 components.
            #
            # BF16:
            #
            # sign     = bit 15
            # exponent = bits 14..7
            # mantissa = bits 6..0
            # -------------------------------------------------

            signs = (
                (bits >> 15) & 1
            ).tolist()

            exponents = (
                (bits >> 7) & 0xFF
            ).tolist()

            mantissas = (
                bits & 0x7F
            ).tolist()

            full_bits = bits.tolist()

            # -------------------------------------------------
            # Update global distributions.
            # -------------------------------------------------

            for s in signs:
                sign_counter[s] += 1

            for e in exponents:
                exponent_counter[e] += 1

            for m in mantissas:
                mantissa_counter[m] += 1

            for b in full_bits:
                full_bf16_counter[b] += 1

            # -------------------------------------------------
            # DELTA CODING INSIDE THIS EMBEDDING VECTOR
            #
            # We deliberately reset at every token.
            #
            # This tests:
            #
            #   dimension 0 -> dimension 1
            #   dimension 1 -> dimension 2
            #   ...
            #
            # rather than comparing the end of one token to
            # the beginning of the next.
            # -------------------------------------------------

            if dimensions > 0:

                previous_e = exponents[0]
                previous_m = mantissas[0]

                exponent_delta_counter[
                    previous_e
                ] += 1

                mantissa_delta_counter[
                    previous_m
                ] += 1

                previous_de = previous_e
                previous_dm = previous_m

                # First-order deltas.
                for i in range(1, dimensions):

                    e = exponents[i]
                    m = mantissas[i]

                    de = e - previous_e
                    dm = m - previous_m

                    exponent_delta_counter[
                        zigzag(de)
                    ] += 1

                    mantissa_delta_counter[
                        zigzag(dm)
                    ] += 1

                    # Second-order deltas.
                    dde = de - previous_de
                    ddm = dm - previous_dm

                    exponent_delta2_counter[
                        zigzag(dde)
                    ] += 1

                    mantissa_delta2_counter[
                        zigzag(ddm)
                    ] += 1

                    previous_e = e
                    previous_m = m

                    previous_de = de
                    previous_dm = dm

            processed += dimensions

            if (
                token_id + 1
            ) % PROGRESS_EVERY == 0:

                print(
                    f"Processed "
                    f"{token_id + 1:,}/{tokens:,} "
                    f"vectors "
                    f"({processed:,} values)"
                )

    # ---------------------------------------------------------
    # ENTROPY
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("ENTROPY RESULTS")
    print("=" * 80)

    print()

    full_h = print_stats(
        "FULL BF16 BIT PATTERN",
        full_bf16_counter,
        total_values,
        16,
    )

    print()

    sign_h = print_stats(
        "SIGN",
        sign_counter,
        total_values,
        1,
    )

    print()

    exponent_h = print_stats(
        "EXPONENT",
        exponent_counter,
        total_values,
        8,
    )

    print()

    exponent_delta_h = print_stats(
        "EXPONENT DELTA",
        exponent_delta_counter,
        total_values,
        8,
    )

    print()

    exponent_delta2_h = print_stats(
        "EXPONENT DELTA²",
        exponent_delta2_counter,
        total_values,
        8,
    )

    print()

    mantissa_h = print_stats(
        "MANTISSA",
        mantissa_counter,
        total_values,
        7,
    )

    print()

    mantissa_delta_h = print_stats(
        "MANTISSA DELTA",
        mantissa_delta_counter,
        total_values,
        7,
    )

    print()

    mantissa_delta2_h = print_stats(
        "MANTISSA DELTA²",
        mantissa_delta2_counter,
        total_values,
        7,
    )

    # ---------------------------------------------------------
    # BEST COMBINATION
    # ---------------------------------------------------------

    best_exponent = min(
        exponent_h,
        exponent_delta_h,
        exponent_delta2_h,
    )

    best_mantissa = min(
        mantissa_h,
        mantissa_delta_h,
        mantissa_delta2_h,
    )

    combined_h = (
        sign_h
        + best_exponent
        + best_mantissa
    )

    estimated_bytes = (
        total_values
        * combined_h
        / 8
    )

    estimated_mib = (
        estimated_bytes
        / 1024
        / 1024
    )

    ratio = (
        16 / combined_h
    )

    # ---------------------------------------------------------
    # PRINT SUMMARY
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print()

    print(
        f"{'Component':25s}"
        f"{'Entropy':>15s}"
        f"{'Nominal':>15s}"
    )

    print("-" * 60)

    print(
        f"{'Full BF16':25s}"
        f"{full_h:15.6f}"
        f"{16:15.1f}"
    )

    print(
        f"{'Sign':25s}"
        f"{sign_h:15.6f}"
        f"{1:15.1f}"
    )

    print(
        f"{'Exponent':25s}"
        f"{exponent_h:15.6f}"
        f"{8:15.1f}"
    )

    print(
        f"{'Exponent delta':25s}"
        f"{exponent_delta_h:15.6f}"
        f"{8:15.1f}"
    )

    print(
        f"{'Exponent delta²':25s}"
        f"{exponent_delta2_h:15.6f}"
        f"{8:15.1f}"
    )

    print(
        f"{'Mantissa':25s}"
        f"{mantissa_h:15.6f}"
        f"{7:15.1f}"
    )

    print(
        f"{'Mantissa delta':25s}"
        f"{mantissa_delta_h:15.6f}"
        f"{7:15.1f}"
    )

    print(
        f"{'Mantissa delta²':25s}"
        f"{mantissa_delta2_h:15.6f}"
        f"{7:15.1f}"
    )

    # ---------------------------------------------------------
    # Best result.
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("BEST THEORETICAL COMBINATION")
    print("=" * 80)

    print()

    print(
        f"Sign entropy          : "
        f"{sign_h:.6f} bits"
    )

    print(
        f"Best exponent entropy : "
        f"{best_exponent:.6f} bits"
    )

    print(
        f"Best mantissa entropy : "
        f"{best_mantissa:.6f} bits"
    )

    print()

    print(
        f"Combined entropy      : "
        f"{combined_h:.6f} bits/value"
    )

    print(
        f"Estimated storage     : "
        f"{estimated_mib:.3f} MiB"
    )

    print(
        f"Original storage      : "
        f"{original_bytes / 1024 / 1024:.3f} MiB"
    )

    print(
        f"Theoretical ratio     : "
        f"{ratio:.3f}x"
    )

    # ---------------------------------------------------------
    # Important warning.
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("IMPORTANT")
    print("=" * 80)

    print()
    print(
        "These are entropy estimates, not an actual compressed file."
    )

    print(
        "The next step, if the deltas look promising, is to implement"
    )

    print(
        "an actual bit-packed/entropy-coded representation and verify"
    )

    print(
        "that every original BF16 bit pattern can be reconstructed."
    )

    # ---------------------------------------------------------
    # Save results.
    # ---------------------------------------------------------

    output = {
        "model_file": MODEL_FILE,
        "embedding_key": EMBEDDING_KEY,
        "shape": list(shape),
        "tokens": tokens,
        "dimensions": dimensions,
        "total_values": total_values,

        "entropy": {
            "full_bf16": full_h,
            "sign": sign_h,
            "exponent": exponent_h,
            "exponent_delta": exponent_delta_h,
            "exponent_delta2": exponent_delta2_h,
            "mantissa": mantissa_h,
            "mantissa_delta": mantissa_delta_h,
            "mantissa_delta2": mantissa_delta2_h,
        },

        "best": {
            "exponent_entropy": best_exponent,
            "mantissa_entropy": best_mantissa,
            "combined_bits_per_value": combined_h,
            "estimated_mib": estimated_mib,
            "ratio": ratio,
        },

        "unique_values": {
            "full_bf16": len(full_bf16_counter),
            "sign": len(sign_counter),
            "exponent": len(exponent_counter),
            "mantissa": len(mantissa_counter),
            "exponent_delta": len(exponent_delta_counter),
            "mantissa_delta": len(mantissa_delta_counter),
            "exponent_delta2": len(exponent_delta2_counter),
            "mantissa_delta2": len(mantissa_delta2_counter),
        },
    }

    output_file = "delta_bf16_batch_results.json"

    with open(
        output_file,
        "w",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print(
        "Results saved to:"
    )

    print(
        os.path.abspath(
            output_file
        )
    )


if __name__ == "__main__":
    main()
