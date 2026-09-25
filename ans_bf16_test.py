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

# rANS parameters.
# Larger precision gives a closer approximation to the ideal entropy,
# at the cost of a slightly larger frequency table.
TOTFREQ_BITS = 12
TOTFREQ = 1 << TOTFREQ_BITS

PROGRESS_EVERY = 5000


# ============================================================
# rANS
# ============================================================

def normalize_frequencies(counter, alphabet_size, total=TOTFREQ):
    """
    Convert symbol counts into integer frequencies whose sum is
    exactly TOTFREQ.

    Every symbol occurring in the data gets at least frequency 1.
    """

    raw_total = sum(counter.values())

    if raw_total == 0:
        raise ValueError("Empty frequency table.")

    symbols = list(range(alphabet_size))

    frequencies = [0] * alphabet_size

    # Initial proportional allocation.
    remainders = []

    for s in symbols:
        count = counter.get(s, 0)

        if count == 0:
            frequencies[s] = 0
            continue

        value = count * total / raw_total

        base = max(1, int(math.floor(value)))

        frequencies[s] = base

        remainders.append(
            (value - math.floor(value), s)
        )

    current = sum(frequencies)

    # Too many frequencies.
    while current > total:

        candidates = [
            s
            for s in symbols
            if frequencies[s] > 1
        ]

        if not candidates:
            raise RuntimeError(
                "Could not normalize frequencies."
            )

        # Remove from the largest frequency.
        s = max(
            candidates,
            key=lambda x: frequencies[x]
        )

        frequencies[s] -= 1
        current -= 1

    # Too few frequencies.
    remainders.sort(reverse=True)

    index = 0

    while current < total:

        if not remainders:
            raise RuntimeError(
                "Could not normalize frequencies."
            )

        s = remainders[index % len(remainders)][1]

        frequencies[s] += 1

        current += 1
        index += 1

    return frequencies


def make_cumulative(freqs):
    cumulative = [0] * len(freqs)

    running = 0

    for i, f in enumerate(freqs):

        cumulative[i] = running

        running += f

    return cumulative


def make_decode_table(freqs, cumulative):
    """
    Maps cumulative frequency slot -> symbol.
    """

    table = [0] * TOTFREQ

    for symbol, freq in enumerate(freqs):

        start = cumulative[symbol]

        end = start + freq

        for i in range(start, end):
            table[i] = symbol

    return table


# rANS uses a large enough state to avoid overflow.
RANS_L = 1 << 23


def rans_encode(symbols, freqs, cumulative):
    """
    Encode symbols in reverse order.
    """

    state = RANS_L

    output = bytearray()

    for symbol in reversed(symbols):

        freq = freqs[symbol]
        start = cumulative[symbol]

        if freq <= 0:
            raise ValueError(
                f"Symbol {symbol} has zero frequency."
            )

        # Renormalize.
        while state >= (
            ((RANS_L >> TOTFREQ_BITS) << 8)
            * freq
        ):

            output.append(
                state & 0xFF
            )

            state >>= 8

        state = (
            (state // freq)
            << TOTFREQ_BITS
        ) + (
            state % freq
        ) + start

    # Store final state followed by renormalization bytes.
    result = bytearray()

    result.extend(
        state.to_bytes(
            4,
            "little",
        )
    )

    result.extend(output[::-1])

    return bytes(result)


def rans_decode(data, count, freqs, cumulative):

    if len(data) < 4:
        raise ValueError("Invalid rANS stream.")

    state = int.from_bytes(
        data[:4],
        "little",
    )

    encoded = data[4:]

    pointer = 0

    decode_table = make_decode_table(
        freqs,
        cumulative,
    )

    result = []

    mask = TOTFREQ - 1

    for _ in range(count):

        slot = state & mask

        symbol = decode_table[slot]

        result.append(symbol)

        freq = freqs[symbol]
        start = cumulative[symbol]

        state = (
            freq
            * (state >> TOTFREQ_BITS)
            + slot
            - start
        )

        while state < RANS_L:

            if pointer >= len(encoded):
                raise ValueError(
                    "Unexpected end of rANS stream."
                )

            state = (
                state << 8
            ) | encoded[pointer]

            pointer += 1

    return result


# ============================================================
# Utility
# ============================================================

def entropy(counter, total):

    if total == 0:
        return 0.0

    result = 0.0

    for count in counter.values():

        p = count / total

        result -= p * math.log2(p)

    return result


def print_distribution(name, counter, total, nominal_bits):

    h = entropy(
        counter,
        total,
    )

    print()
    print(name)

    print(
        f"  Unique values : {len(counter):,}"
    )

    print(
        f"  Entropy       : {h:.6f} bits/value"
    )

    print(
        f"  Nominal       : {nominal_bits} bits/value"
    )

    print(
        f"  Saving        : "
        f"{nominal_bits - h:.6f} bits/value"
    )

    return h


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("GRANITE BF16 SEPARATE ANS COMPRESSION TEST")
    print("=" * 80)

    print()
    print("Model:")
    print(MODEL_FILE)

    if not os.path.exists(MODEL_FILE):

        raise FileNotFoundError(
            MODEL_FILE
        )

    # --------------------------------------------------------
    # Read shape.
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
        f"Values          : "
        f"{total_values:,}"
    )

    print(
        f"Original        : "
        f"{original_bytes / 1024 / 1024:.2f} MiB"
    )

    # --------------------------------------------------------
    # PASS 1
    #
    # Build frequency tables.
    #
    # Only counters are retained.
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("PASS 1: BUILDING FREQUENCY TABLES")
    print("=" * 80)

    sign_counter = Counter()
    exponent_counter = Counter()
    mantissa_counter = Counter()

    processed = 0

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

            bits_list = bits.tolist()

            for value in bits_list:

                sign = (
                    value >> 15
                ) & 1

                exponent = (
                    value >> 7
                ) & 0xFF

                mantissa = (
                    value & 0x7F
                )

                sign_counter[sign] += 1

                exponent_counter[
                    exponent
                ] += 1

                mantissa_counter[
                    mantissa
                ] += 1

            processed += dimensions

            if (
                token_id + 1
            ) % PROGRESS_EVERY == 0:

                print(
                    f"Processed "
                    f"{token_id + 1:,}/"
                    f"{tokens:,}"
                )

    # --------------------------------------------------------
    # Entropy.
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("ENTROPY")
    print("=" * 80)

    sign_entropy = print_distribution(
        "SIGN",
        sign_counter,
        total_values,
        1,
    )

    exponent_entropy = print_distribution(
        "EXPONENT",
        exponent_counter,
        total_values,
        8,
    )

    mantissa_entropy = print_distribution(
        "MANTISSA",
        mantissa_counter,
        total_values,
        7,
    )

    theoretical_bits = (
        sign_entropy
        + exponent_entropy
        + mantissa_entropy
    )

    theoretical_bytes = (
        total_values
        * theoretical_bits
        / 8
    )

    print()
    print(
        f"Theoretical combined entropy:"
    )

    print(
        f"  {theoretical_bits:.6f} bits/value"
    )

    print(
        f"  {theoretical_bytes / 1024 / 1024:.3f} MiB"
    )

    # --------------------------------------------------------
    # Normalize frequencies.
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("BUILDING ANS MODELS")
    print("=" * 80)

    sign_freqs = normalize_frequencies(
        sign_counter,
        2,
    )

    exponent_freqs = normalize_frequencies(
        exponent_counter,
        256,
    )

    mantissa_freqs = normalize_frequencies(
        mantissa_counter,
        128,
    )

    sign_cumulative = make_cumulative(
        sign_freqs
    )

    exponent_cumulative = make_cumulative(
        exponent_freqs
    )

    mantissa_cumulative = make_cumulative(
        mantissa_freqs
    )

    print(
        "ANS frequency precision:"
        f" {TOTFREQ_BITS} bits"
    )

    print(
        f"Sign total frequency     : "
        f"{sum(sign_freqs)}"
    )

    print(
        f"Exponent total frequency : "
        f"{sum(exponent_freqs)}"
    )

    print(
        f"Mantissa total frequency : "
        f"{sum(mantissa_freqs)}"
    )

    # --------------------------------------------------------
    # PASS 2
    #
    # Encode each component.
    #
    # We encode each component as one large stream.
    #
    # To keep memory use low, chunks are encoded separately.
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("PASS 2: ANS ENCODING")
    print("=" * 80)

    # We use one embedding vector as the maximum temporary
    # symbol buffer.
    sign_buffer = []
    exponent_buffer = []
    mantissa_buffer = []

    sign_encoded = bytearray()
    exponent_encoded = bytearray()
    mantissa_encoded = bytearray()

    processed = 0

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

            bits_list = bits.tolist()

            sign_buffer.clear()
            exponent_buffer.clear()
            mantissa_buffer.clear()

            for value in bits_list:

                sign_buffer.append(
                    (value >> 15) & 1
                )

                exponent_buffer.append(
                    (value >> 7) & 0xFF
                )

                mantissa_buffer.append(
                    value & 0x7F
                )

            sign_stream = rans_encode(
                sign_buffer,
                sign_freqs,
                sign_cumulative,
            )

            exponent_stream = rans_encode(
                exponent_buffer,
                exponent_freqs,
                exponent_cumulative,
            )

            mantissa_stream = rans_encode(
                mantissa_buffer,
                mantissa_freqs,
                mantissa_cumulative,
            )

            sign_encoded.extend(
                sign_stream
            )

            exponent_encoded.extend(
                exponent_stream
            )

            mantissa_encoded.extend(
                mantissa_stream
            )

            processed += dimensions

            if (
                token_id + 1
            ) % PROGRESS_EVERY == 0:

                print(
                    f"Encoded "
                    f"{token_id + 1:,}/"
                    f"{tokens:,} vectors"
                )

    # --------------------------------------------------------
    # Actual size.
    #
    # IMPORTANT:
    #
    # This includes rANS stream overhead caused by encoding
    # each 768-value vector independently.
    #
    # --------------------------------------------------------

    sign_bytes = len(
        sign_encoded
    )

    exponent_bytes = len(
        exponent_encoded
    )

    mantissa_bytes = len(
        mantissa_encoded
    )

    total_compressed = (
        sign_bytes
        + exponent_bytes
        + mantissa_bytes
    )

    # --------------------------------------------------------
    # Model overhead.
    #
    # The decoder also needs the frequency tables.
    # --------------------------------------------------------

    frequency_table_bytes = (
        2 * len(sign_freqs)
        + 2 * len(exponent_freqs)
        + 2 * len(mantissa_freqs)
    )

    total_with_tables = (
        total_compressed
        + frequency_table_bytes
    )

    # --------------------------------------------------------
    # Statistics.
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("ACTUAL ANS SIZE")
    print("=" * 80)

    print()

    print(
        f"{'Component':20s}"
        f"{'Bytes':>15s}"
        f"{'Bits/value':>15s}"
    )

    print("-" * 55)

    print(
        f"{'Sign':20s}"
        f"{sign_bytes:15,d}"
        f"{sign_bytes * 8 / total_values:15.6f}"
    )

    print(
        f"{'Exponent':20s}"
        f"{exponent_bytes:15,d}"
        f"{exponent_bytes * 8 / total_values:15.6f}"
    )

    print(
        f"{'Mantissa':20s}"
        f"{mantissa_bytes:15,d}"
        f"{mantissa_bytes * 8 / total_values:15.6f}"
    )

    print("-" * 55)

    print(
        f"{'TOTAL':20s}"
        f"{total_compressed:15,d}"
        f"{total_compressed * 8 / total_values:15.6f}"
    )

    print()
    print(
        f"Frequency tables : "
        f"{frequency_table_bytes:,} bytes"
    )

    print(
        f"Total with tables: "
        f"{total_with_tables:,} bytes"
    )

    print()
    print(
        f"Compressed MiB   : "
        f"{total_compressed / 1024 / 1024:.3f}"
    )

    print(
        f"With tables MiB  : "
        f"{total_with_tables / 1024 / 1024:.3f}"
    )

    print(
        f"Original MiB     : "
        f"{original_bytes / 1024 / 1024:.3f}"
    )

    print()

    print(
        f"Payload ratio    : "
        f"{original_bytes / total_compressed:.4f}x"
    )

    print(
        f"Total ratio      : "
        f"{original_bytes / total_with_tables:.4f}x"
    )

    # --------------------------------------------------------
    # Lossless verification.
    #
    # Instead of keeping the entire compressed file in RAM,
    # verify a sample of vectors by re-reading the model.
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("LOSSLESS VERIFICATION")
    print("=" * 80)

    # Since streams above were concatenated without storing
    # per-vector lengths, a complete decoder would need stream
    # framing.
    #
    # For this experiment we verify the ANS implementation on
    # representative vectors independently.

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

            signs = [
                (x >> 15) & 1
                for x in original
            ]

            exponents = [
                (x >> 7) & 0xFF
                for x in original
            ]

            mantissas = [
                x & 0x7F
                for x in original
            ]

            s_stream = rans_encode(
                signs,
                sign_freqs,
                sign_cumulative,
            )

            e_stream = rans_encode(
                exponents,
                exponent_freqs,
                exponent_cumulative,
            )

            m_stream = rans_encode(
                mantissas,
                mantissa_freqs,
                mantissa_cumulative,
            )

            s_decoded = rans_decode(
                s_stream,
                dimensions,
                sign_freqs,
                sign_cumulative,
            )

            e_decoded = rans_decode(
                e_stream,
                dimensions,
                exponent_freqs,
                exponent_cumulative,
            )

            m_decoded = rans_decode(
                m_stream,
                dimensions,
                mantissa_freqs,
                mantissa_cumulative,
            )

            reconstructed = []

            for i in range(dimensions):

                value = (
                    (s_decoded[i] << 15)
                    | (e_decoded[i] << 7)
                    | m_decoded[i]
                )

                reconstructed.append(
                    value
                )

            if reconstructed != original:

                failures += 1

                print(
                    f"FAIL token {token_id}"
                )

            else:

                print(
                    f"PASS token {token_id}: "
                    f"exact BF16 reconstruction"
                )

    print()

    if failures == 0:

        print(
            "All verification vectors reconstructed "
            "EXACTLY."
        )

    else:

        print(
            f"Verification failures: {failures}"
        )

    # --------------------------------------------------------
    # Save results.
    # --------------------------------------------------------

    results = {

        "model_file": MODEL_FILE,

        "embedding_key": EMBEDDING_KEY,

        "shape": list(shape),

        "total_values": total_values,

        "original_bytes": original_bytes,

        "entropy_bits_per_value": {
            "sign": sign_entropy,
            "exponent": exponent_entropy,
            "mantissa": mantissa_entropy,
            "combined": theoretical_bits,
        },

        "ans": {
            "frequency_bits": TOTFREQ_BITS,

            "sign_bytes": sign_bytes,

            "exponent_bytes": exponent_bytes,

            "mantissa_bytes": mantissa_bytes,

            "payload_bytes": total_compressed,

            "frequency_table_bytes":
                frequency_table_bytes,

            "total_bytes":
                total_with_tables,

            "payload_bits_per_value":
                total_compressed * 8 / total_values,

            "total_bits_per_value":
                total_with_tables * 8 / total_values,

            "payload_ratio":
                original_bytes / total_compressed,

            "total_ratio":
                original_bytes / total_with_tables,
        },

        "lossless_verification": {
            "tested_vectors": len(test_ids),
            "failures": failures,
            "exact": failures == 0,
        },
    }

    output_file = (
        "ans_bf16_results.json"
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
