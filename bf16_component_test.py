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

MODEL_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "granite-4.0-h-350m",
    "model.safetensors",
)

EMBEDDING_KEY = "model.embed_tokens.weight"

# Test the complete embedding matrix.
NUM_TOKENS = 100352

# Keep this reasonably small for WSL.
CHUNK_SIZE = 4096


# ============================================================
# HELPERS
# ============================================================

def entropy(counter, total):
    if total == 0:
        return 0.0

    h = 0.0

    for count in counter.values():
        p = count / total
        h -= p * math.log2(p)

    return h


def bits_needed(number_of_values):
    if number_of_values <= 1:
        return 0

    return math.ceil(
        math.log2(number_of_values)
    )


def mib(bits):
    return bits / 8 / 1024 / 1024


# ============================================================
# EXTRACT BF16 COMPONENTS
# ============================================================

def extract_components(x):
    """
    Convert BF16 values into their exact 16-bit bit patterns.

    BF16:

        bit 15       = sign
        bits 14..7    = exponent
        bits 6..0     = mantissa

    We convert to int32 before shifting because PyTorch
    does not implement CPU bit shifting for UInt16.
    """

    bits = x.view(torch.uint16)

    # IMPORTANT:
    # Convert to int32 before bit operations.
    bits = bits.to(torch.int32)

    sign = (
        bits >> 15
    ) & 0x1

    exponent = (
        bits >> 7
    ) & 0xFF

    mantissa = (
        bits & 0x7F
    )

    return (
        bits,
        sign,
        exponent,
        mantissa,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("GRANITE BF16 COMPONENT COMPRESSION TEST")
    print("=" * 80)

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"\nModel file not found:\n{MODEL_FILE}"
        )

    print()
    print("Model:")
    print(MODEL_FILE)

    # ========================================================
    # OPEN MODEL
    # ========================================================

    with safe_open(
        MODEL_FILE,
        framework="pt",
        device="cpu",
    ) as f:

        if EMBEDDING_KEY not in f.keys():
            raise RuntimeError(
                f"Embedding not found: {EMBEDDING_KEY}"
            )

        embedding = f.get_slice(
            EMBEDDING_KEY
        )

        shape = embedding.get_shape()

    total_tokens = shape[0]
    dimensions = shape[1]

    tokens = min(
        NUM_TOKENS,
        total_tokens,
    )

    total_values = (
        tokens * dimensions
    )

    print()
    print(
        f"Embedding shape : "
        f"{total_tokens:,} x {dimensions}"
    )

    print(
        f"Values tested   : "
        f"{total_values:,}"
    )

    # ========================================================
    # COUNTERS
    # ========================================================

    sign_counter = Counter()
    exponent_counter = Counter()
    mantissa_counter = Counter()
    full_counter = Counter()

    # ========================================================
    # PHASE 1
    # ========================================================

    print()
    print("=" * 80)
    print("PHASE 1: EXTRACT BF16 COMPONENTS")
    print("=" * 80)

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
            ]

            bits, sign, exponent, mantissa = (
                extract_components(x)
            )

            # Convert to Python integers for Counter.
            sign_counter.update(
                sign.flatten().tolist()
            )

            exponent_counter.update(
                exponent.flatten().tolist()
            )

            mantissa_counter.update(
                mantissa.flatten().tolist()
            )

            full_counter.update(
                bits.flatten().tolist()
            )

            print(
                f"Processed "
                f"{end:,}/{tokens:,}"
            )

    # ========================================================
    # PHASE 2
    # ========================================================

    print()
    print("=" * 80)
    print("PHASE 2: UNIQUE VALUES")
    print("=" * 80)

    unique_sign = len(
        sign_counter
    )

    unique_exponent = len(
        exponent_counter
    )

    unique_mantissa = len(
        mantissa_counter
    )

    unique_full = len(
        full_counter
    )

    print()
    print(
        f"Unique sign values     : "
        f"{unique_sign}"
    )

    print(
        f"Unique exponent values : "
        f"{unique_exponent}"
    )

    print(
        f"Unique mantissa values : "
        f"{unique_mantissa}"
    )

    print(
        f"Unique full BF16       : "
        f"{unique_full}"
    )

    # ========================================================
    # FIXED WIDTH LOOKUP
    # ========================================================

    sign_bits = bits_needed(
        unique_sign
    )

    exponent_bits = bits_needed(
        unique_exponent
    )

    mantissa_bits = bits_needed(
        unique_mantissa
    )

    full_bits = bits_needed(
        unique_full
    )

    print()
    print(
        "Fixed-width lookup:"
    )

    print()
    print(
        f"Sign     : "
        f"{sign_bits} bits/value"
    )

    print(
        f"Exponent : "
        f"{exponent_bits} bits/value"
    )

    print(
        f"Mantissa : "
        f"{mantissa_bits} bits/value"
    )

    print(
        f"Separate : "
        f"{sign_bits + exponent_bits + mantissa_bits} "
        f"bits/value"
    )

    print(
        f"Full BF16: "
        f"{full_bits} bits/value"
    )

    # ========================================================
    # PHASE 3
    # ========================================================

    print()
    print("=" * 80)
    print("PHASE 3: ENTROPY")
    print("=" * 80)

    sign_entropy = entropy(
        sign_counter,
        total_values,
    )

    exponent_entropy = entropy(
        exponent_counter,
        total_values,
    )

    mantissa_entropy = entropy(
        mantissa_counter,
        total_values,
    )

    full_entropy = entropy(
        full_counter,
        total_values,
    )

    separate_entropy = (
        sign_entropy
        +
        exponent_entropy
        +
        mantissa_entropy
    )

    print()
    print(
        f"Sign entropy     : "
        f"{sign_entropy:.8f} bits/value"
    )

    print(
        f"Exponent entropy : "
        f"{exponent_entropy:.8f} bits/value"
    )

    print(
        f"Mantissa entropy : "
        f"{mantissa_entropy:.8f} bits/value"
    )

    print(
        f"Separate entropy : "
        f"{separate_entropy:.8f} bits/value"
    )

    print(
        f"Full BF16 entropy: "
        f"{full_entropy:.8f} bits/value"
    )

    # ========================================================
    # PHASE 4
    # ========================================================

    print()
    print("=" * 80)
    print("PHASE 4: EXACT RECONSTRUCTION")
    print("=" * 80)

    exact_count = 0
    tested_count = 0

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
            ]

            original_bits, sign, exponent, mantissa = (
                extract_components(x)
            )

            reconstructed = (
                (sign << 15)
                |
                (exponent << 7)
                |
                mantissa
            )

            matches = (
                reconstructed
                ==
                original_bits
            )

            exact_count += (
                matches.sum().item()
            )

            tested_count += (
                matches.numel()
            )

    exact_percent = (
        exact_count
        /
        tested_count
        *
        100
    )

    print()
    print(
        f"Exact matches : "
        f"{exact_count:,}/{tested_count:,}"
    )

    print(
        f"Exact BF16 reconstruction: "
        f"{exact_percent:.8f}%"
    )

    # ========================================================
    # PHASE 5
    # ========================================================

    print()
    print("=" * 80)
    print("PHASE 5: STORAGE ANALYSIS")
    print("=" * 80)

    original_bits_total = (
        total_values * 16
    )

    # --------------------------------------------------------
    # Full-value lookup
    #
    # Every value gets an ID.
    #
    # Dictionary stores:
    #
    #     unique BF16 values × 16 bits
    # --------------------------------------------------------

    full_lookup_bits = (
        total_values * full_bits
        +
        unique_full * 16
    )

    # --------------------------------------------------------
    # Separate component lookup
    #
    # Each component gets its own ID.
    # --------------------------------------------------------

    separate_index_bits = (
        total_values
        *
        (
            sign_bits
            +
            exponent_bits
            +
            mantissa_bits
        )
    )

    separate_dictionary_bits = (
        separate_index_bits
        +
        unique_sign * 1
        +
        unique_exponent * 8
        +
        unique_mantissa * 7
    )

    print()
    print("Original BF16:")
    print(
        f"  {mib(original_bits_total):.4f} MiB"
    )

    print()
    print("Full-value lookup:")
    print(
        f"  Index size      : "
        f"{full_bits} bits/value"
    )

    print(
        f"  Total storage   : "
        f"{mib(full_lookup_bits):.4f} MiB"
    )

    print(
        f"  Compression     : "
        f"{original_bits_total / full_lookup_bits:.4f}x"
    )

    print()
    print("Separate components:")
    print(
        f"  Index size      : "
        f"{sign_bits + exponent_bits + mantissa_bits} "
        f"bits/value"
    )

    print(
        f"  Total storage   : "
        f"{mib(separate_dictionary_bits):.4f} MiB"
    )

    print(
        f"  Compression     : "
        f"{original_bits_total / separate_dictionary_bits:.4f}x"
    )

    # ========================================================
    # PHASE 6
    # ========================================================

    print()
    print("=" * 80)
    print("PHASE 6: MOST COMMON EXPONENTS")
    print("=" * 80)

    for value, count in (
        exponent_counter.most_common(30)
    ):

        percentage = (
            count
            /
            total_values
            *
            100
        )

        print(
            f"Exponent {value:3d}: "
            f"{count:12,d} "
            f"({percentage:8.4f}%)"
        )

    # ========================================================
    # PHASE 7
    # ========================================================

    print()
    print("=" * 80)
    print("PHASE 7: MOST COMMON MANTISSAS")
    print("=" * 80)

    for value, count in (
        mantissa_counter.most_common(30)
    ):

        percentage = (
            count
            /
            total_values
            *
            100
        )

        print(
            f"Mantissa {value:3d}: "
            f"{count:12,d} "
            f"({percentage:8.4f}%)"
        )

    # ========================================================
    # SAVE
    # ========================================================

    results = {

        "model_file":
            MODEL_FILE,

        "embedding_key":
            EMBEDDING_KEY,

        "embedding_shape":
            list(shape),

        "tokens_tested":
            tokens,

        "total_values":
            total_values,

        "unique_values": {

            "sign":
                unique_sign,

            "exponent":
                unique_exponent,

            "mantissa":
                unique_mantissa,

            "full_bf16":
                unique_full,
        },

        "fixed_width_bits": {

            "sign":
                sign_bits,

            "exponent":
                exponent_bits,

            "mantissa":
                mantissa_bits,

            "separate":
                (
                    sign_bits
                    +
                    exponent_bits
                    +
                    mantissa_bits
                ),

            "full_bf16":
                full_bits,
        },

        "entropy": {

            "sign":
                sign_entropy,

            "exponent":
                exponent_entropy,

            "mantissa":
                mantissa_entropy,

            "separate":
                separate_entropy,

            "full_bf16":
                full_entropy,
        },

        "exact_reconstruction_percent":
            exact_percent,

        "storage": {

            "original_mib":
                mib(original_bits_total),

            "full_lookup_mib":
                mib(full_lookup_bits),

            "separate_lookup_mib":
                mib(separate_dictionary_bits),

            "full_lookup_ratio":
                (
                    original_bits_total
                    /
                    full_lookup_bits
                ),

            "separate_lookup_ratio":
                (
                    original_bits_total
                    /
                    separate_dictionary_bits
                ),
        },
    }

    output_file = (
        "bf16_component_results.json"
    )

    with open(
        output_file,
        "w",
    ) as f:

        json.dump(
            results,
            f,
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
            output_file
        )
    )


if __name__ == "__main__":
    main()
