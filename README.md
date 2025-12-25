# GPT from Scratch (PyTorch)

This project implements a **decoder-only GPT-style language model from scratch in PyTorch**, closely following the GPT-2 architecture while remaining runnable on a **MacBook M2 (MPS)**.

The goal of this project was **deep understanding**, not just usage:

* Implement attention, transformer blocks, and training logic manually
* Understand tokenization (character-level → BPE)
* Debug real-world issues (shape bugs, MPS device quirks)

By the end, this is a **fully functional GPT** capable of training and generating text. The model still fails to generate perfectly meaningful English right now, but it's almost there and quite a signficant achievement based on the training data and compute constraint.

---

## Features

* Decoder-only Transformer (GPT-style)
* Causal multi-head self-attention
* Pre-LayerNorm architecture (GPT-2 style)
* Residual connections
* Feed-forward MLP with GELU
* Weight tying (input embeddings = output projection)
* Autoregressive text generation
* Character-level tokenizer (baseline)
* Byte Pair Encoding (BPE) tokenizer (GPT-2 compatible)
* Trains on Apple Silicon (MPS) or CPU/GPU

---

## Model Architecture

```
Tokens
  ↓
Token Embedding + Positional Embedding
  ↓
[ Transformer Block × N ]
  ↓
LayerNorm
  ↓
Linear (LM Head, weight-tied)
  ↓
Softmax → Next-token probabilities
```

### Transformer Block (GPT-2 style)

Each block contains:

```
LN → Multi-Head Causal Self-Attention → Residual
LN → MLP (4× expansion + GELU) → Residual
```

Key design choices:

* **Causal masking** prevents tokens from attending to the future
* **Pre-LayerNorm** improves training stability
* **Residual connections** enable deep stacking

---

## Tokenization

### Character-level (baseline)

* Vocabulary size: ~65 characters
* Simple and fast
* Useful for debugging and architectural validation

### Byte Pair Encoding (BPE)

* Uses GPT-2 compatible BPE via `tiktoken`
* Vocabulary size: 50,257
* Byte-level encoding (handles all Unicode safely)
* Much higher-quality text generation

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")

tokens = enc.encode(text)
decode = enc.decode
```

The **model architecture remains unchanged** when switching tokenizers.

---

## Hyperparameters (Typical)

```python
vocab_size = 50257        # GPT-2 BPE
n_embd = 256              # embedding dimension
n_head = 8                # attention heads
n_layer = 6               # transformer blocks
block_size = 256          # max context length
batch_size = 8–16
```

These settings are tuned to run comfortably on a MacBook M2.

---

## Training

Training is standard next-token prediction:

```text
Input : x₀, x₁, ..., xₜ₋₁
Target: x₁, x₂, ..., xₜ
```

Loss:

```python
F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
```

Typical training length:

* 5k steps → usable model
* 10k+ steps → good text quality

---

## Text Generation

Autoregressive sampling:

```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
out = generate(model, context, max_new_tokens=500)
print(decode(out[0].tolist()))
```

Supports:

* MPS-safe sampling
* Context cropping to `block_size`

---

## Saving & Loading

Weights are saved using `state_dict` (best practice):

```python
torch.save(model.state_dict(), "gpt.pt")
```

Or with full checkpoint metadata:

```python
{
  "model_state": ...,
  "config": ...,
  "step": ...,
  "loss": ...
}
```

Models can be safely loaded across CPU / MPS / GPU.

---

## What This Project Demonstrates

* A **true from-scratch GPT implementation**
* Deep understanding of:

  * Attention mechanics
  * Tokenization tradeoffs
  * Transformer architecture
  * Training dynamics
* Ability to debug real-world ML systems

This goes well beyond "using HuggingFace" and demonstrates **model-building literacy**.

---

## Possible Extensions

* KV caching for faster generation
* Top-k / nucleus sampling
* Mixed precision training
* Rotary positional embeddings (RoPE)
* Larger context lengths

---

## Credits & References

* "Attention Is All You Need" (Vaswani et al.)
* Andrej Karpathy – NanoGPT lectures
* "Attention in transformers, step-by-step | Deep Learning Chapter 6" - 3Blue1Brown
---

## Author

Built by **Anuj Sindgi** as a deep-dive project into transformer architectures and language modeling.

