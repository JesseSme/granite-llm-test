#!/usr/bin/env python3

import os
import math
import json
from collections import Counter, defaultdict

import torch
from safetensors import safe_open

MODEL = "granite-4.0-h-350m/model.safetensors"
KEY = "model.embed_tokens.weight"

PROGRESS = 5000

# ---------- Entropy ----------

def entropy(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        p = c / total
        h -= p * math.log2(p)
    return h

def conditional_entropy(context_map):
    total = 0
    weighted = 0.0
    for ctx in context_map.values():
        n = sum(ctx.values())
        total += n
        weighted += n * entropy(ctx)
    return weighted / total

# ---------- Context hash ----------

def hash_context(exp, prev_m):
    # 30 exponents -> low bits
    # previous mantissa -> high bits
    return ((prev_m << 5) ^ exp) & 255

# ---------- Scan ----------

with safe_open(MODEL, framework="pt", device="cpu") as f:
    emb = f.get_slice(KEY)
    TOKENS, DIMS = emb.get_shape()

print("="*80)
print("CABAC CONTEXT EXPERIMENT")
print("="*80)
print(f"Embedding: {TOKENS:,} x {DIMS}")

mantissa_counter = Counter()

pair_counter = Counter()
triple_counter = Counter()

prev_context = defaultdict(Counter)
exp_context = defaultdict(Counter)
hash_contexts = defaultdict(Counter)

with safe_open(MODEL, framework="pt", device="cpu") as f:
    emb = f.get_slice(KEY)

    for token in range(TOKENS):

        bits = emb[token:token+1].reshape(-1).view(torch.uint16).to(torch.int32)

        mant = (bits & 0x7F).tolist()
        exp = ((bits >> 7) & 0xFF).tolist()

        mantissa_counter.update(mant)

        for i in range(DIMS-1):
            pair_counter[(mant[i], mant[i+1])] += 1

        for i in range(DIMS-2):
            triple_counter[(mant[i], mant[i+1], mant[i+2])] += 1

        for i in range(1, DIMS):

            prev_context[mant[i-1]][mant[i]] += 1

            exp_context[exp[i]][mant[i]] += 1

            ctx = hash_context(exp[i], mant[i-1])
            hash_contexts[ctx][mant[i]] += 1

        if token % PROGRESS == 0:
            print("Processed", token)

print("\n" + "="*80)
print("GLOBAL MANTISSA ENTROPY")
print("="*80)

global_h = entropy(mantissa_counter)
print(f"H(M) = {global_h:.6f} bits")

print("\n" + "="*80)
print("CONDITIONAL ENTROPIES")
print("="*80)

h_prev = conditional_entropy(prev_context)
h_exp = conditional_entropy(exp_context)
h_hash = conditional_entropy(hash_contexts)

print(f"H(M | previous mantissa)      = {h_prev:.6f}")
print(f"H(M | exponent)               = {h_exp:.6f}")
print(f"H(M | exponent + prev hash)   = {h_hash:.6f}")

print("\nSavings versus global entropy:")
print(f"Previous mantissa : {global_h-h_prev:.6f} bits/value")
print(f"Exponent          : {global_h-h_exp:.6f} bits/value")
print(f"Hash context      : {global_h-h_hash:.6f} bits/value")

print("\n" + "="*80)
print("TOP 50 MANTISSA PAIRS")
print("="*80)

top_pairs = pair_counter.most_common(50)
for pair,count in top_pairs:
    print(pair, count)

print("\n" + "="*80)
print("TOP 50 MANTISSA TRIPLES")
print("="*80)

top_triples = triple_counter.most_common(50)
for tri,count in top_triples:
    print(tri, count)

print("\n" + "="*80)
print("MOST PREDICTABLE PREVIOUS MANTISSAS")
print("="*80)

predictability = []

for prev,counter in prev_context.items():

    total = sum(counter.values())
    best,count = counter.most_common(1)[0]

    predictability.append((
        count/total,
        prev,
        best,
        total
    ))

predictability.sort(reverse=True)

for p,prev,best,total in predictability[:30]:
    print(
        f"prev={prev:3d} -> {best:3d} "
        f"{100*p:5.2f}% ({total:,} samples)"
    )

results = {
    "global_entropy": global_h,
    "conditional_entropy": {
        "previous_mantissa": h_prev,
        "exponent": h_exp,
        "hashed_context": h_hash,
    },
    "bits_saved": {
        "previous_mantissa": global_h-h_prev,
        "exponent": global_h-h_exp,
        "hashed_context": global_h-h_hash,
    },
    "top_pairs": [
        {"pair": list(k), "count": v}
        for k,v in top_pairs
    ],
    "top_triples": [
        {"triple": list(k), "count": v}
        for k,v in top_triples
    ],
    "predictable_contexts": [
        {
            "previous": prev,
            "predict": best,
            "probability": p,
            "samples": total
        }
        for p,prev,best,total in predictability[:100]
    ]
}

with open("cabac_context_results.json","w") as fp:
    json.dump(results, fp, indent=2)

print("\nSaved cabac_context_results.json")
