#!/usr/bin/env python3

import os
import json
import math
from collections import Counter

import torch
from safetensors import safe_open


MODEL_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "granite-4.0-h-350m",
    "model.safetensors",
)

EMBEDDING_KEY = "model.embed_tokens.weight"

PROGRESS_EVERY = 5000


# ============================================================
# Simple CABAC-style binary arithmetic coder
# ============================================================
#
# This is a self-contained binary arithmetic coder using
# adaptive probability estimation.
#
# It is not an H.264/H.265 standards-compatible CABAC
# implementation. It is intended to test the compression
# principle on the embedding data.
#
# Each binary decision has an adaptive probability model.
#
# ============================================================


class BinaryModel:

    def __init__(self):

        # Probability represented as an integer:
        #
        # 0 ... 4095
        #
        # probability of symbol 1.
        #
        # Start at 0.5.
        self.p1 = 2048

    def update(self, bit):

        # Adapt slowly enough to exploit statistics.
        #
        # If bit == 1:
        # probability moves upward.
        #
        # If bit == 0:
        # probability moves downward.

        if bit:

            self.p1 += (
                (4096 - self.p1) >> 5
            )

        else:

            self.p1 -= (
                self.p1 >> 5
            )

        # Avoid pathological zero/one probabilities.

        self.p1 = max(
            1,
            min(
                4095,
                self.p1,
            ),
        )


class ArithmeticEncoder:

    def __init__(self):

        self.low = 0
        self.high = (1 << 32) - 1

        self.output = bytearray()

    def encode_bit(
        self,
        bit,
        model,
    ):

        p1 = model.p1

        total = 4096

        range_value = (
            self.high
            - self.low
            + 1
        )

        split = (
            self.low
            + (
                range_value
                * (total - p1)
                // total
            )
            - 1
        )

        if bit == 0:

            self.high = split

        else:

            self.low = split + 1

        # Renormalize.

        while (
            (self.high ^ self.low)
            < (1 << 24)
        ):

            self.output.append(
                self.high >> 24
            )

            self.low = (
                (self.low << 8)
                & 0xFFFFFFFF
            )

            self.high = (
                (
                    self.high << 8
                )
                & 0xFFFFFFFF
            ) | 0xFF

        model.update(bit)

    def finish(self):

        # Emit final state.

        for shift in (
            24,
            16,
            8,
            0,
        ):

            self.output.append(
                (
                    self.low >> shift
                ) & 0xFF
            )

        return bytes(self.output)


class ArithmeticDecoder:

    def __init__(self, data):

        self.data = data

        self.position = 0

        self.low = 0
        self.high = (1 << 32) - 1

        self.code = 0

        # Initial 4 bytes.

        for _ in range(4):

            self.code = (
                (self.code << 8)
                | self.read_byte()
            )

    def read_byte(self):

        if (
            self.position
            >= len(self.data)
        ):

            return 0

        value = self.data[
            self.position
        ]

        self.position += 1

        return value

    def decode_bit(
        self,
        model,
    ):

        p1 = model.p1

        total = 4096

        range_value = (
            self.high
            - self.low
            + 1
        )

        split = (
            self.low
            + (
                range_value
                * (total - p1)
                // total
            )
            - 1
        )

        if self.code <= split:

            bit = 0

            self.high = split

        else:

            bit = 1

            self.low = split + 1

        # Renormalize.

        while (
            (self.high ^ self.low)
            < (1 << 24)
        ):

            self.low = (
                (self.low << 8)
                & 0xFFFFFFFF
            )

            self.high = (
                (
                    self.high << 8
                )
                & 0xFFFFFFFF
            ) | 0xFF

            self.code = (
                (
                    self.code << 8
                )
                & 0xFFFFFFFF
            ) | self.read_byte()

        model.update(bit)

        return bit


# ============================================================
# Bitstream helpers
# ============================================================


def encode_bitstream(values, width):

    """
    Encode a sequence of integers as individual bits.

    Each bit position gets its own adaptive binary model.

    This means:

        exponent bit 0
        exponent bit 1
        ...
        exponent bit 7

    each has its own probability model.

    Likewise for mantissa.
    """

    encoder = ArithmeticEncoder()

    models = [
        BinaryModel()
        for _ in range(width)
    ]

    for value in values:

        for bit_position in range(width):

            bit = (
                value
                >> bit_position
            ) & 1

            encoder.encode_bit(
                bit,
                models[bit_position],
            )

    return encoder.finish()


def decode_bitstream(
    data,
    count,
    width,
):

    decoder = ArithmeticDecoder(
        data
    )

    models = [
        BinaryModel()
        for _ in range(width)
    ]

    values = []

    for _ in range(count):

        value = 0

        for bit_position in range(width):

            bit = decoder.decode_bit(
                models[bit_position]
            )

            value |= (
                bit
                << bit_position
            )

        values.append(value)

    return values


# ============================================================
# Adaptive context experiment
# ============================================================


class ContextModel:

    def __init__(self):

        # Context is previous value.

        self.models = {}

    def get(self, context):

        if context not in self.models:

            self.models[
                context
            ] = BinaryModel()

        return self.models[
            context
        ]


def encode_context_values(
    values,
    width,
):

    """
    Context-adaptive binary coding.

    For each value:

        current value
             │
             ▼
       previous value
             │
             ▼
        probability
             │
             ▼
          CABAC

    This is useful for testing whether neighboring embedding
    dimensions contain predictive structure.
    """

    encoder = ArithmeticEncoder()

    contexts = [
        ContextModel()
        for _ in range(width)
    ]

    previous = 0

    for value in values:

        for bit_position in range(width):

            bit = (
                value
                >> bit_position
            ) & 1

            # Use previous value's corresponding bit
            # as the context.

            previous_bit = (
                previous
                >> bit_position
            ) & 1

            model = contexts[
                bit_position
            ].get(
                previous_bit
            )

            encoder.encode_bit(
                bit,
                model,
            )

        previous = value

    return encoder.finish()


def decode_context_values(
    data,
    count,
    width,
):

    decoder = ArithmeticDecoder(
        data
    )

    contexts = [
        ContextModel()
        for _ in range(width)
    ]

    previous = 0

    values = []

    for _ in range(count):

        value = 0

        for bit_position in range(width):

            previous_bit = (
                previous
                >> bit_position
            ) & 1

            model = contexts[
                bit_position
            ].get(
                previous_bit
            )

            bit = decoder.decode_bit(
                model
            )

            value |= (
                bit
                << bit_position
            )

        values.append(value)

        previous = value

    return values


# ============================================================
# Main
# ============================================================


def main():

    print("=" * 80)
    print("GRANITE BF16 CABAC MANTISSA + EXPONENT TEST")
    print("=" * 80)

    print()
    print("Model:")
    print(MODEL_FILE)

    if not os.path.exists(
        MODEL_FILE
    ):

        raise FileNotFoundError(
            MODEL_FILE
        )

    # --------------------------------------------------------
    # Shape
    # --------------------------------------------------------

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

    total_values = (
        tokens * dimensions
    )

    original_bytes = (
        total_values * 2
    )

    print()
    print(
        f"Embedding shape : "
        f"{tokens:,} x {dimensions}"
    )

    print(
        f"Dimensions      : "
        f"{dimensions}"
    )

    print(
        f"Total values    : "
        f"{total_values:,}"
    )

    print(
        f"Original size   : "
        f"{original_bytes / 1024 / 1024:.3f} MiB"
    )

    # --------------------------------------------------------
    # Frequency counters.
    # --------------------------------------------------------

    exponent_counter = Counter()
    mantissa_counter = Counter()

    # --------------------------------------------------------
    # PASS 1
    #
    # Determine ordinary entropy.
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("PASS 1: ANALYZING BF16 COMPONENTS")
    print("=" * 80)

    with safe_open(
        MODEL_FILE,
        framework="pt",
        device="cpu",
    ) as f:

        tensor_slice = f.get_slice(
            EMBEDDING_KEY
        )

        for token_id in range(tokens):

            vector = tensor_slice[
                token_id:token_id + 1
            ]

            vector = vector.reshape(-1)

            bits = vector.view(
                torch.uint16
            ).to(torch.int32)

            bit_list = bits.tolist()

            for value in bit_list:

                exponent = (
                    value >> 7
                ) & 0xFF

                mantissa = (
                    value & 0x7F
                )

                exponent_counter[
                    exponent
                ] += 1

                mantissa_counter[
                    mantissa
                ] += 1

            if (
                token_id + 1
            ) % PROGRESS_EVERY == 0:

                print(
                    f"Processed "
                    f"{token_id + 1:,}/"
                    f"{tokens:,}"
                )

    def entropy(counter):

        result = 0.0

        for count in counter.values():

            p = (
                count
                / total_values
            )

            result -= (
                p
                * math.log2(p)
            )

        return result

    exponent_entropy = entropy(
        exponent_counter
    )

    mantissa_entropy = entropy(
        mantissa_counter
    )

    print()
    print(
        f"Exponent entropy : "
        f"{exponent_entropy:.6f} bits"
    )

    print(
        f"Mantissa entropy : "
        f"{mantissa_entropy:.6f} bits"
    )

    print(
        f"Combined         : "
        f"{exponent_entropy + mantissa_entropy:.6f} bits"
    )

    # --------------------------------------------------------
    # PASS 2
    #
    # Build actual streams.
    #
    # IMPORTANT:
    #
    # Sign is NOT encoded here.
    #
    # It remains the first bit of every embedding record.
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("PASS 2: BUILDING EXPONENT + MANTISSA STREAMS")
    print("=" * 80)

    exponent_stream = bytearray()
    mantissa_stream = bytearray()

    # --------------------------------------------------------
    # Metadata needed to decode independent vector records.
    #
    # Each vector produces one exponent stream and one
    # mantissa stream.
    #
    # We store the lengths so that a decoder knows where each
    # vector starts.
    # --------------------------------------------------------

    exponent_lengths = []
    mantissa_lengths = []

    # We don't need to store the signs because the requested
    # representation leaves them in the record itself.
    #
    # For this experiment, sign bits are therefore counted as
    # exactly one bit/value but are NOT put into either stream.

    with safe_open(
        MODEL_FILE,
        framework="pt",
        device="cpu",
    ) as f:

        tensor_slice = f.get_slice(
            EMBEDDING_KEY
        )

        for token_id in range(tokens):

            vector = tensor_slice[
                token_id:token_id + 1
            ]

            vector = vector.reshape(-1)

            bits = vector.view(
                torch.uint16
            ).to(torch.int32)

            bit_list = bits.tolist()

            exponents = []
            mantissas = []

            for value in bit_list:

                exponents.append(
                    (value >> 7) & 0xFF
                )

                mantissas.append(
                    value & 0x7F
                )

            # ------------------------------------------------
            # Independent streams for this vector.
            # ------------------------------------------------

            e_stream = encode_bitstream(
                exponents,
                8,
            )

            m_stream = encode_bitstream(
                mantissas,
                7,
            )

            exponent_lengths.append(
                len(e_stream)
            )

            mantissa_lengths.append(
                len(m_stream)
            )

            exponent_stream.extend(
                e_stream
            )

            mantissa_stream.extend(
                m_stream
            )

            if (
                token_id + 1
            ) % PROGRESS_EVERY == 0:

                print(
                    f"Encoded "
                    f"{token_id + 1:,}/"
                    f"{tokens:,}"
                )

    # --------------------------------------------------------
    # Sign storage.
    #
    # Exactly 1 bit/value.
    #
    # In an actual structure this can be packed into a separate
    # bit array or placed as the first bit of each record.
    # --------------------------------------------------------

    sign_bytes = (
        (total_values + 7)
        // 8
    )

    exponent_bytes = len(
        exponent_stream
    )

    mantissa_bytes = len(
        mantissa_stream
    )

    payload_bytes = (
        sign_bytes
        + exponent_bytes
        + mantissa_bytes
    )

    # --------------------------------------------------------
    # Metadata overhead.
    #
    # Each vector needs lengths for the two streams.
    #
    # A real implementation could use fixed-size records or
    # another index structure. For now we explicitly measure
    # the cost.
    # --------------------------------------------------------

    length_metadata_bytes = (
        tokens * 4 * 2
    )

    total_bytes = (
        payload_bytes
        + length_metadata_bytes
    )

    # --------------------------------------------------------
    # Results.
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("ACTUAL STREAM SIZES")
    print("=" * 80)

    print()
    print(
        f"Sign bit storage       : "
        f"{sign_bytes:,} bytes"
    )

    print(
        f"Exponent stream        : "
        f"{exponent_bytes:,} bytes"
    )

    print(
        f"Mantissa stream        : "
        f"{mantissa_bytes:,} bytes"
    )

    print(
        f"Stream payload         : "
        f"{payload_bytes:,} bytes"
    )

    print(
        f"Stream payload MiB     : "
        f"{payload_bytes / 1024 / 1024:.3f}"
    )

    print()
    print(
        f"Length metadata        : "
        f"{length_metadata_bytes:,} bytes"
    )

    print(
        f"TOTAL                  : "
        f"{total_bytes:,} bytes"
    )

    print(
        f"TOTAL MiB              : "
        f"{total_bytes / 1024 / 1024:.3f}"
    )

    print()
    print(
        f"Original MiB           : "
        f"{original_bytes / 1024 / 1024:.3f}"
    )

    print(
        f"Payload ratio          : "
        f"{original_bytes / payload_bytes:.4f}x"
    )

    print(
        f"Total ratio            : "
        f"{original_bytes / total_bytes:.4f}x"
    )

    print()
    print(
        f"Payload bits/value     : "
        f"{payload_bytes * 8 / total_values:.6f}"
    )

    print(
        f"Total bits/value       : "
        f"{total_bytes * 8 / total_values:.6f}"
    )

    # --------------------------------------------------------
    # LOSSLESS VERIFICATION
    # --------------------------------------------------------
    #
    # Verify several complete embedding vectors.
    #
    # Reconstruct:
    #
    #     sign
    #       |
    #       +-- exponent
    #       |
    #       +-- mantissa
    #
    # back into the exact 16-bit BF16 pattern.
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("LOSSLESS RECONSTRUCTION TEST")
    print("=" * 80)

    test_ids = [
        0,
        1,
        2,
        100,
        1000,
        10000,
        50000,
        tokens - 1,
    ]

    test_ids = [
        x
        for x in test_ids
        if 0 <= x < tokens
    ]

    # Build offsets.

    exponent_offsets = []

    offset = 0

    for length in exponent_lengths:

        exponent_offsets.append(
            offset
        )

        offset += length

    mantissa_offsets = []

    offset = 0

    for length in mantissa_lengths:

        mantissa_offsets.append(
            offset
        )

        offset += length

    failures = 0

    with safe_open(
        MODEL_FILE,
        framework="pt",
        device="cpu",
    ) as f:

        tensor_slice = f.get_slice(
            EMBEDDING_KEY
        )

        for token_id in test_ids:

            vector = tensor_slice[
                token_id:token_id + 1
            ]

            vector = vector.reshape(-1)

            bits = vector.view(
                torch.uint16
            ).to(torch.int32)

            original = bits.tolist()

            # Extract original sign.

            signs = [
                (
                    value
                    >> 15
                ) & 1
                for value in original
            ]

            # Locate stream.

            e_start = exponent_offsets[
                token_id
            ]

            e_end = (
                e_start
                + exponent_lengths[
                    token_id
                ]
            )

            m_start = mantissa_offsets[
                token_id
            ]

            m_end = (
                m_start
                + mantissa_lengths[
                    token_id
                ]
            )

            e_data = exponent_stream[
                e_start:e_end
            ]

            m_data = mantissa_stream[
                m_start:m_end
            ]

            decoded_exponents = (
                decode_bitstream(
                    e_data,
                    dimensions,
                    8,
                )
            )

            decoded_mantissas = (
                decode_bitstream(
                    m_data,
                    dimensions,
                    7,
                )
            )

            reconstructed = []

            for i in range(dimensions):

                value = (
                    (signs[i] << 15)
                    |
                    (
                        decoded_exponents[i]
                        << 7
                    )
                    |
                    decoded_mantissas[i]
                )

                reconstructed.append(
                    value
                )

            if (
                reconstructed
                != original
            ):

                failures += 1

                print(
                    f"FAIL token "
                    f"{token_id}"
                )

            else:

                print(
                    f"PASS token "
                    f"{token_id}: "
                    f"exact BF16"
                )

    print()

    if failures == 0:

        print(
            "ALL TEST VECTORS RECONSTRUCTED "
            "EXACTLY."
        )

    else:

        print(
            f"FAILURES: {failures}"
        )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    results = {

        "model_file": MODEL_FILE,

        "embedding_key": EMBEDDING_KEY,

        "shape": list(shape),

        "total_values": total_values,

        "original_bytes": original_bytes,

        "entropy": {

            "exponent":
                exponent_entropy,

            "mantissa":
                mantissa_entropy,

            "combined":
                exponent_entropy
                + mantissa_entropy,
        },

        "streams": {

            "sign_bytes":
                sign_bytes,

            "exponent_bytes":
                exponent_bytes,

            "mantissa_bytes":
                mantissa_bytes,

            "payload_bytes":
                payload_bytes,

            "length_metadata_bytes":
                length_metadata_bytes,

            "total_bytes":
                total_bytes,

            "payload_bits_per_value":
                payload_bytes
                * 8
                / total_values,

            "total_bits_per_value":
                total_bytes
                * 8
                / total_values,

            "payload_ratio":
                original_bytes
                / payload_bytes,

            "total_ratio":
                original_bytes
                / total_bytes,
        },

        "verification": {

            "vectors_tested":
                len(test_ids),

            "failures":
                failures,

            "exact":
                failures == 0,
        },
    }

    output_file = (
        "cabac_bf16_results.json"
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
