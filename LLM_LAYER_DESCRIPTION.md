# LLM Layer Description — Granite 4.0-H-350M

This document describes every operation in the Granite 4.0-H-350M inference
pipeline, derived from a Python debug trace of the actual forward pass.
Tokenization is excluded — only the compute layers are described.

## Model Overview

| Property | Value |
|----------|-------|
| Architecture | `GraniteMoeHybridForCausalLM` |
| Total parameters | ~340M |
| Hidden size | 768 |
| Intermediate size (MLP) | 2048 |
| Vocab size | 100,352 |
| Dtype | bfloat16 |
| Position embeddings | None (NoPE) |
| Tied word embeddings | Yes (embed_tokens = lm_head) |

## Layer Layout

32 decoder layers. Each layer is one of two types:

| Layer Index | Type |
|-------------|------|
| 0–9, 11–12, 14–16, 18–26, 28–31 | **Mamba2** (28 layers) |
| 10, 13, 17, 27 | **Attention** (4 layers) |

Every layer has the same outer structure:

```
input → input_layernorm → [mixer] → + residual → post_attention_layernorm → shared_mlp → + residual → output
```

---

## 1. Embedding

```
input_ids (B×S int64) → hidden_states (B×S×768 bfloat16)
```

Lookup table: each token ID indexes into a (100352 × 768) weight matrix.
The result is multiplied by `embedding_multiplier = 12`.

**Weight shape:** `(100352, 768)` — 77M params (shared with lm_head).

---

## 2. Decoder Layer — Mamba2 Block (28 layers)

Each Mamba2 layer performs:

### 2a. input_layernorm (RMSNorm)
```
hidden_states (B×S×768) → normalized (B×S×768)
```
RMSNorm with eps=1e-5. Weights: (768,).

### 2b. MambaLayer — Selective State Space Model

The Mamba layer implements a selective SSM (S6) with input-dependent gating.

**Step 1 — Input projection (in_proj)**
```
normalized (B×S×768) → projected (B×S×3376)
```
Linear projection. Weight: (3376, 768) = 2.59M params.
The output is split into two parts along the last dim:
- First 1536 dims → gate branch
- Remaining 1536 dims → state branch

**Step 2 — Gated RMSNorm**
```
gate (B×S×1536 float32) + state (B×S×1536 bfloat16) → normalized (B×S×1536 float32)
```
`GraniteMoeHybridRMSNormGated` — applies RMSNorm to the gate branch
while element-wise gating the state branch. Weights: (1536,) × 2.

**Step 3 — Conv1d (causal)**
```
state (B×S×1536) → convolved (B×S×1536)
```
1D causal convolution, kernel_size=4, groups=1536 (depthwise).
Weight: (1536, 1, 4). Bias: (1536,).
Applied per-channel over the sequence dimension with left padding of 3.

**Step 4 — SiLU activation**
```
convolved → activated (B×S×1536)
```
Element-wise SiLU: `x * sigmoid(x)`.

**Step 5 — SSM selective scan**

The activated tensor is reshaped and split into:
- **B matrix** (input-dependent transition): projects 1536 → 128 (mamba_d_state)
- **C matrix** (input-dependent output mix): projects 1536 → 128
- **dt** (input-dependent step size): projects 1536 → 48 (mamba_n_heads)

The SSM recurrence:
```
for each position t:
    h = A * h + B_t * x_t      (state update, A is diagonal)
    y_t = C_t * h               (output mix)
```
Where:
- `A`: (48, 128) log-domain diagonal matrix (learned, not input-dependent)
- `B_t`: input-dependent (48, 128) per batch element
- `C_t`: input-dependent (48, 128) per batch element
- `dt`: input-dependent step size, discretized via softplus

Output: (B×S×48×32) reshaped to (B×S×1536).

**Step 6 — Gated output norm**
```
(B×S×1536) + gate → gated_output (B×S×1536)
```
Element-wise multiply with the gate branch (from Step 1's split).

**Step 7 — Output projection (out_proj)**
```
gated_output (B×S×1536) → mamba_output (B×S×768)
```
Linear projection. Weight: (768, 1536) = 1.18M params.

### 2c. Residual add
```
output = hidden_states + mamba_output
```
Scaled by `residual_multiplier = 0.246`.

---

## 3. Decoder Layer — Attention Block (4 layers)

Used at layers 10, 13, 17, 27. Standard grouped-query attention (GQA).

### 3a. input_layernorm (RMSNorm)
Same as §2a.

### 3b. Self-attention

**Parameters:**
- num_attention_heads: 12
- num_key_value_heads: 4 (GQA, each KV head shared across 3 Q heads)
- head_dim: 64 (768 / 12)

**Projections:**
```
Q: normalized (B×S×768) → q (B×S×768)     Weight: (768, 768) = 590K params
K: normalized (B×S×768) → k (B×S×256)     Weight: (256, 768) = 197K params
V: normalized (B×S×768) → v (B×S×256)     Weight: (256, 768) = 197K params
```

**Attention computation:**
```
q → reshape to (B, S, 12, 64)
k → reshape to (B, S, 4, 64), repeat_interleave(3, dim=1) → (B, S, 12, 64)
v → reshape to (B, S, 4, 64), repeat_interleave(3, dim=1) → (B, S, 12, 64)

scores = (q @ k^T) / sqrt(64)
scores = scores * attention_multiplier (= 0.015625)
attn = softmax(scores)
context = attn @ v

context → reshape to (B, S, 768)
```

**Output projection:**
```
context (B×S×768) → attn_output (B×S×768)   Weight: (768, 768) = 590K params
```

No position embeddings (NoPE). No causal mask needed for single-token generation.

### 3c. Residual add
Same scaling as §2c.

---

## 4. post_attention_layernorm (RMSNorm)

```
(B×S×768) → (B×S×768)
```
Applied after the mixer (Mamba or Attention) and before the MLP.
Weights: (768,).

---

## 5. MLP — SwiGLU (all 32 layers)

Every layer has the same MLP structure, regardless of Mamba/Attention type.

**Step 1 — Gate + Up projection (fused)**
```
input (B×S×768) → combined (B×S×4096)
```
Single linear layer: Weight (4096, 768) = 3.15M params.
Split into two halves along last dim:
- First 2048 → gate branch
- Last 2048 → up branch

**Step 2 — SiLU activation on gate**
```
gate (B×S×2048) → activated (B×S×2048)
```

**Step 3 — Element-wise multiply**
```
output = activated * up   (B×S×2048)
```

**Step 4 — Down projection**
```
(B×S×2048) → mlp_output (B×S×768)
```
Linear: Weight (768, 2048) = 1.57M params.

### 5a. Residual add
```
output = hidden_states + mlp_output
```
Scaled by `residual_multiplier = 0.246`.

---

## 6. Final Norm (RMSNorm)

```
final_hidden (B×S×768) → normalized (B×S×768)
```
Applied once after all 32 layers. Weights: (768,).

---

## 7. Output Head (lm_head)

```
normalized (B×S×768) → logits (B×S×100352) × logits_scaling (= 3)
```
Linear projection. Weight: (100352, 768) = 77M params (tied with embedding).

---

## Weight Summary

| Component | Shape | Params | Per-Layer |
|-----------|-------|--------|-----------|
| embed_tokens / lm_head | (100352, 768) | 77,070,336 | — |
| input_layernorm ×32 | (768,) | 24,576 | 768 |
| post_attention_layernorm ×32 | (768,) | 24,576 | 768 |
| **Mamba layer** ×28 | | | |
| — in_proj | (3376, 768) | 2,592,768 | 2,592,768 |
| — norm (gated) | (1536,) ×2 | 3,072 | 3,072 ×2 |
| — conv1d | (1536, 1, 4) | 6,144 | 6,144 |
| — out_proj | (768, 1536) | 1,179,648 | 1,179,648 |
| — SSM params (A, B, C, dt) | various | ~144 | ~144 |
| **Attention layer** ×4 | | | |
| — q_proj | (768, 768) | 589,824 | 589,824 |
| — k_proj | (256, 768) | 196,608 | 196,608 |
| — v_proj | (256, 768) | 196,608 | 196,608 |
| — o_proj | (768, 768) | 589,824 | 589,824 |
| **MLP (SwiGLU)** ×32 | | | |
| — input_linear | (4096, 768) | 3,145,728 | 3,145,728 |
| — output_linear | (768, 2048) | 1,572,864 | 1,572,864 |
| final_norm | (768,) | 768 | — |

---

## Tensor Shape Reference

For batch size B and sequence length S:

| Stage | Shape | Dtype |
|-------|-------|-------|
| Input tokens | (B, S) | int64 |
| After embedding | (B, S, 768) | bfloat16 |
| After any RMSNorm | (B, S, 768) | bfloat16 |
| Mamba in_proj output | (B, S, 3376) | bfloat16 |
| Mamba gate/state | (B, S, 1536) | float32 / bfloat16 |
| Mamba SSM state h | (B, 48, 128) | float32 |
| Mamba out_proj output | (B, S, 768) | bfloat16 |
| Attention Q | (B, S, 768) | bfloat16 |
| Attention K | (B, S, 256) | bfloat16 |
| Attention V | (B, S, 256) | bfloat16 |
| Attention scores | (B, 12, S, S) | float32 |
| MLP gate/up | (B, S, 2048) | bfloat16 |
| MLP combined | (B, S, 4096) | bfloat16 |
| Logits | (B, S, 100352) | bfloat16 |

---

## Operations Needed for SystemVerilog Implementation

Listed in dependency order (implement from top to bottom — each unit depends
only on units listed above it):

### Layer 1 — Leaf operations (no HW unit dependencies)

| # | Unit | Operation | Description |
|---|------|-----------|-------------|
| 1 | `fp_unit` | IEEE 754 arithmetic | Already implemented in `systemverilog_fp_unit/` |
| 2 | `sigmoid_unit` | `σ(x) = 1/(1+exp(-x))` | Element-wise, building block for SiLU |
| 3 | `SiLU_unit` | `x * σ(x)` | Element-wise, uses sigmoid_unit |
| 4 | `softmax_unit` | `exp(x-max)/sum(exp(x-max))` | Numerically stable, for attention scores |
| 5 | `RMSNorm_unit` | `x/sqrt(mean(x²)+ε)*w` | Vector normalization, 768 or 1536 wide |
| 6 | `conv1d_unit` | Causal depthwise 1D conv | 4-tap FIR, kernel=(1536,1,4), groups=1536 |
| 7 | `embedding_lookup_unit` | Table lookup | Token ID → (768,) vector, scale ×12 |
| 8 | `residual_adder_unit` | `x + residual * 0.246` | Scaled add, constant multiplier |

### Layer 2 — Composite units (depend on Layer 1)

| # | Unit | Operation | Dependencies |
|---|------|-----------|--------------|
| 9 | `matrix_unit` | `y = xW^T + b` | fp_unit |
| 10 | `SSM_unit` | S6 selective scan recurrence | matrix_unit (for B/C/dt projections) |
| 11 | `softmax_unit` | Attention score normalization | (standalone, listed in Layer 1) |

### Layer 3 — Sub-systems (depend on Layer 1+2)

| # | Unit | Operation | Dependencies |
|---|------|-----------|--------------|
| 12 | `attention_unit` | GQA: Q/K/V → scores → context → O | matrix_unit, softmax_unit |
| 13 | `SwiGLU_unit` | gate+up → SiLU → multiply → down | matrix_unit, SiLU_unit |
| 14 | `mlp_unit` | SwiGLU MLP wrapper | SwiGLU_unit |

### Layer 4 — Full mixers (depend on Layer 1+2+3)

| # | Unit | Operation | Dependencies |
|---|------|-----------|--------------|
| 15 | `mamba2_unit` | Full Mamba layer (in_proj→norm→conv→SiLU→SSM→out_proj) | matrix_unit, RMSNorm_unit, conv1d_unit, SiLU_unit, SSM_unit |
| 16 | `output_projection_unit` | lm_head: (768)→(100352), scale ×3 | matrix_unit |

### Inference execution order (per token)

```
1.  embedding_lookup_unit      token_id → (768,)
2.  FOR each of 32 layers:
2a.   RMSNorm_unit             input_layernorm
2b.   MIXER (one of):
        IF attention layer (10,13,17,27):
          attention_unit        Q/K/V → context
        ELSE (28 mamba layers):
          mamba2_unit           in_proj → norm → conv → SiLU → SSM → out_proj
2c.   residual_adder_unit      hidden + mixer_output
2d.   RMSNorm_unit             post_attention_layernorm
2e.   mlp_unit                 SwiGLU: gate+up → SiLU → mul → down
2f.   residual_adder_unit      hidden + mlp_output
3.  RMSNorm_unit               final_norm
4.  output_projection_unit     (768) → (100352) logits
```
