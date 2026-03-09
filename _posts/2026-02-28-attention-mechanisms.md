---
layout: post
title: "Is Attention Really All You Need?"
date: 2026-02-28 10:00:00 -0800
categories: [deep-learning, llm]
tags: [attention, transformer, flash-attention, multi-head-attention, gqa, mqa, sparse-attention, from-scratch]
series: transformers
author: cmenguy
colab_url: "https://colab.research.google.com/github/cmenguy/cmenguy.github.io/blob/main/notebooks/2026-02-28-attention-mechanisms.ipynb"
colab_embed: false
github_notebook: "https://github.com/cmenguy/cmenguy.github.io/blob/main/notebooks/2026-02-28-attention-mechanisms.ipynb"
notebook_description: "Runnable implementations of every attention variant covered in the post, from vanilla scaled dot-product to flash attention."
---

A couple weeks ago I wrote a [walkthrough of Karpathy's microgpt.py]({% post_url 2026-02-16-gpt-in-pure-python %}), tracing every line of a GPT implementation in pure Python. While putting that post together, I kept wanting to linger on the attention section. The scaled dot-product formula showed up, multi-head attention showed up, I explained them, and moved on. But there was so much more to say. The attention mechanism in that 200-line implementation is the same one from the 2017 paper. Production models in 2026 use something very different. Flash attention, grouped-query attention, sliding windows, sparse patterns. The core Q, K, V math stays the same, but the way it gets computed, what gets masked, and how memory gets managed have all evolved significantly.

I'll be upfront: I was never a deep math guy. I didn't come up through a pure ML research track. I'm an engineer who builds systems that use these models, and for a long time I treated attention as a black box with a formula I could recite but didn't fully feel in my bones. That was fine when every model used the same vanilla multi-head attention. It's not fine anymore. When you're picking between GQA and MQA for a serving setup, or trying to understand why your 128k context model OOMs while your colleague's doesn't, you need to know what changed and why. So I went back and traced the evolution properly, starting from the original 2017 formulation and working forward to what's actually running in production today.

This post walks through every major attention variant you'll encounter in modern transformer models. For each one: the math, a visual to build intuition, working code, and when you'd actually use it. Think of it as the attention chapter that the GPT post didn't have room for.

## Scaled Dot-Product Attention: The Foundation

Everything starts here. This is Section 3.2.1 of [Attention Is All You Need](https://arxiv.org/abs/1706.03762), and it's what we saw in the microgpt.py walkthrough.

You have three matrices: queries (Q), keys (K), and values (V). The query asks "what am I looking for?", the key says "what do I contain?", and the value says "here's my actual information." Attention computes a weighted sum of values, where the weights come from how well each query matches each key.

The formula:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Let's break this into steps. Say we have a sequence of 4 tokens, each with embedding dimension 8. Q, K, and V are all shape (4, 8).

```
Step 1: QK^T       -> (4, 8) x (8, 4) = (4, 4) matrix of raw scores
Step 2: / sqrt(dk) -> scale each score by 1/sqrt(8) ≈ 0.354
Step 3: softmax    -> each row becomes a probability distribution
Step 4: x V        -> (4, 4) x (4, 8) = (4, 8) weighted sum of values
```

The output has the same shape as the input. Each position gets a new representation that's a weighted mix of all the value vectors, with weights determined by query-key similarity.

Here's the implementation in PyTorch:

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V):
    """
    Q, K, V: shape (seq_len, d_k)
    Returns: shape (seq_len, d_k)
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    return weights @ V
```

And a pure Python version that mirrors what microgpt.py does, without any libraries:

```python
def attention_pure(Q, K, V):
    """Pure Python attention. Q, K, V are lists of lists."""
    d_k = len(Q[0])
    scale = d_k ** 0.5
    seq_len = len(Q)

    # Step 1-2: scaled dot products
    scores = []
    for i in range(seq_len):
        row = []
        for j in range(seq_len):
            dot = sum(Q[i][k] * K[j][k] for k in range(d_k))
            row.append(dot / scale)
        scores.append(row)

    # Step 3: softmax per row
    weights = []
    for row in scores:
        max_val = max(row)
        exps = [math.exp(x - max_val) for x in row]
        total = sum(exps)
        weights.append([e / total for e in exps])

    # Step 4: weighted sum of values
    out = []
    for i in range(seq_len):
        vec = [0.0] * d_k
        for j in range(seq_len):
            for k in range(d_k):
                vec[k] += weights[i][j] * V[j][k]
        out.append(vec)
    return out
```

Let's verify they produce the same result:

```python
torch.manual_seed(42)
seq_len, d_k = 4, 8
Q = torch.randn(seq_len, d_k)
K = torch.randn(seq_len, d_k)
V = torch.randn(seq_len, d_k)

out_torch = scaled_dot_product_attention(Q, K, V)
out_pure = attention_pure(
    Q.tolist(), K.tolist(), V.tolist()
)
out_pure_t = torch.tensor(out_pure)

print(f"Max difference: {(out_torch - out_pure_t).abs().max().item():.2e}")
# Max difference: 5.96e-08
```

Same result, give or take floating point precision. The PyTorch version runs ~1000x faster on GPU because it's one batched matrix multiply instead of nested Python loops.

**Why the $\sqrt{d_k}$ scaling?** Without it, dot products grow proportionally to $d_k$. If your keys and queries have entries with zero mean and unit variance, the expected value of each dot product is 0, but the variance is $d_k$. For $d_k = 64$ (common in practice), that means scores can easily be in the range [-20, 20], which pushes softmax into near-one-hot territory where gradients vanish. Dividing by $\sqrt{d_k}$ brings the variance back to 1.

**When is this used?** This exact formulation appears inside every transformer, but never alone. It's always wrapped in multi-head attention (next section). The standalone version is mostly useful for teaching and debugging.

## Multi-Head Attention: Parallel Perspectives

Single-head attention has a limitation: the model gets one "perspective" on how tokens relate. Multi-head attention runs several attention operations in parallel, each with different learned projections, then concatenates the results.

From Section 3.2.2 of the paper:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

Visually, here's what happens for 4 heads with $d_{model} = 16$ and $d_k = 4$ per head:

```
Input x (seq_len, 16)
  │
  ├─ W_Q1 ──► Q1 (seq_len, 4) ─┐
  ├─ W_K1 ──► K1 (seq_len, 4)  ├─► Attention ──► head1 (seq_len, 4)
  ├─ W_V1 ──► V1 (seq_len, 4) ─┘
  │
  ├─ W_Q2 ──► Q2 (seq_len, 4) ─┐
  ├─ W_K2 ──► K2 (seq_len, 4)  ├─► Attention ──► head2 (seq_len, 4)
  ├─ W_V2 ──► V2 (seq_len, 4) ─┘
  │
  ├─ ... (heads 3, 4)
  │
  └─► Concat(head1..4) = (seq_len, 16) ──► W_O ──► output (seq_len, 16)
```

Each head operates in a smaller subspace ($d_k = d_{model} / h$), so the total computation cost is the same as a single full-dimension attention.

In microgpt.py, this was the loop over heads with slice indexing:

```python
# From microgpt.py - the head loop
for h in range(n_head):
    hs = h * head_dim
    q_h = q[hs:hs+head_dim]
    k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
    v_h = [vi[hs:hs+head_dim] for vi in values[li]]
```

In practice, the projections are done as a single matrix multiply and then reshaped. Here's a clean PyTorch implementation:

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, C = x.shape  # batch, seq_len, d_model

        # Project and reshape: (B, T, C) -> (B, n_heads, T, d_k)
        Q = self.W_Q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention per head
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        attn_out = weights @ V  # (B, n_heads, T, d_k)

        # Concat heads and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_O(attn_out)
```

The key insight: the `view` and `transpose` operations are free (just pointer math, no data movement). A single `nn.Linear(d_model, d_model)` produces all head projections at once, and the reshape splits them into heads. This is how every production transformer does it.

Let's verify it works:

```python
torch.manual_seed(42)
mha = MultiHeadAttention(d_model=16, n_heads=4)
x = torch.randn(2, 8, 16)  # batch=2, seq_len=8, d_model=16
out = mha(x)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {out.shape}")
# Input shape:  torch.Size([2, 8, 16])
# Output shape: torch.Size([2, 8, 16])
```

**What do different heads learn?** Research shows that heads tend to specialize. In trained models, some heads attend to the previous token, some to syntactic structures (subject-verb pairs), others to positional patterns. Removing certain heads barely affects performance; removing others is catastrophic. This emergent specialization is why multi-head works better than single-head despite using the same total compute.

**When is this used?** Every transformer model since 2017. This is the standard. The variants that follow are all modifications of this core structure.

## Causal (Masked) Attention: No Peeking

In language models, a token at position $i$ should only attend to tokens at positions $\leq i$. You can't let the model see future tokens during training, or it'll just copy the answer instead of learning to predict.

The fix is a causal mask: set the upper-right triangle of the attention score matrix to $-\infty$ before softmax. Since $e^{-\infty} = 0$, those positions get zero weight.

```
Attention scores before masking (4 tokens):

         key0   key1   key2   key3
query0 [ 0.5    -∞     -∞     -∞  ]     <- can only see itself
query1 [ 0.3    0.8    -∞     -∞  ]     <- sees positions 0-1
query2 [ 0.1    0.4    0.6    -∞  ]     <- sees positions 0-2
query3 [ 0.2    0.3    0.1    0.9 ]     <- sees all positions

After softmax, each row sums to 1.0 over visible positions only.
```

The math is the same as before, with a mask added:

$$\text{CausalAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

where $M_{ij} = 0$ if $i \geq j$ (allowed), and $M_{ij} = -\infty$ if $i < j$ (blocked).

The implementation is just adding the mask to our existing code:

```python
def causal_attention(Q, K, V):
    d_k = Q.size(-1)
    T = Q.size(-2)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    # Create causal mask: lower triangle = True, upper = False
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float('-inf'))

    weights = F.softmax(scores, dim=-1)
    return weights @ V
```

Let's see the mask in action:

```python
torch.manual_seed(42)
T, d_k = 4, 8
Q = torch.randn(T, d_k)
K = torch.randn(T, d_k)
V = torch.randn(T, d_k)

d_k_val = Q.size(-1)
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k_val)
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
scores_masked = scores.masked_fill(~mask, float('-inf'))
weights = F.softmax(scores_masked, dim=-1)

print("Attention weights (causal):")
print(weights.numpy().round(3))
```

```
Attention weights (causal):
[[1.    0.    0.    0.   ]
 [0.059 0.941 0.    0.   ]
 [0.211 0.418 0.371 0.   ]
 [0.193 0.18  0.195 0.432]]
```

Row 0 puts all weight on position 0 (only option). Row 1 heavily favors position 1 (0.941) over position 0 (0.059). By row 3, the weights spread across all four positions, with position 3 getting the largest share. The upper triangle is all zeros.

In microgpt.py, causal masking was implicit. Because it processed one token at a time and only stored past keys/values in the KV cache, the model could never attend to future positions. The explicit mask approach is what you use during training, when the full sequence is processed in parallel.

**When is this used?** Every autoregressive language model (GPT, Llama, Claude, Mistral, Gemini). If it generates text left-to-right, it uses causal masking.

## Cross-Attention: Connecting Two Sequences

The original transformer was built for translation. You have an input sentence in French and want to produce output in English. The encoder processes the French input; the decoder generates the English output. Cross-attention is how the decoder "looks at" the encoder's representation.

The math is identical to self-attention, except Q comes from the decoder and K, V come from the encoder:

$$\text{CrossAttention} = \text{softmax}\left(\frac{Q_{dec} K_{enc}^T}{\sqrt{d_k}}\right) V_{enc}$$

```
Encoder output (French): (src_len, d_model)  <- "Bonjour le monde"
Decoder state (English): (tgt_len, d_model)  <- "Hello" (so far)

Q = decoder_state @ W_Q    -> (tgt_len, d_k)     <- decoder asks questions
K = encoder_output @ W_K   -> (src_len, d_k)      <- encoder provides keys
V = encoder_output @ W_V   -> (src_len, d_k)      <- encoder provides values

scores: (tgt_len, src_len)  <- each decoder position scores every encoder position
output: (tgt_len, d_k)      <- decoder gets encoder-informed representations
```

Here's the implementation. It's the same `MultiHeadAttention` class, but called with different inputs for Q vs. K/V:

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, decoder_state, encoder_output):
        B, T_dec, C = decoder_state.shape
        T_enc = encoder_output.size(1)

        Q = self.W_Q(decoder_state).view(B, T_dec, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(encoder_output).view(B, T_enc, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(encoder_output).view(B, T_enc, self.n_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        weights = F.softmax(scores, dim=-1)
        attn_out = weights @ V

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_dec, C)
        return self.W_O(attn_out)
```

```python
torch.manual_seed(42)
cross_attn = CrossAttention(d_model=16, n_heads=4)
enc_out = torch.randn(2, 10, 16)   # 10 source tokens
dec_state = torch.randn(2, 6, 16)  # 6 target tokens so far
out = cross_attn(dec_state, enc_out)
print(f"Decoder queries: {dec_state.shape}")
print(f"Encoder context: {enc_out.shape}")
print(f"Output shape:    {out.shape}")
# Decoder queries: torch.Size([2, 6, 16])
# Encoder context: torch.Size([2, 10, 16])
# Output shape:    torch.Size([2, 6, 16])
```

Notice there's no causal mask here. Each decoder position should be able to attend to any encoder position. The word "monde" might be most relevant for generating "world" regardless of position.

**When is this used?** Encoder-decoder models: the original Transformer, T5, BART, Whisper (speech-to-text), and vision-language models where the "encoder" is an image encoder and the "decoder" generates text. GPT-style decoder-only models don't use cross-attention at all. They threw out the encoder and replaced cross-attention with just longer context windows.

## Multi-Query Attention: One KV Head for All

Here's where we move from the 2017 paper to modern efficiency optimizations. Multi-Query Attention (MQA), introduced by [Shazeer (2019)](https://arxiv.org/abs/1911.02150), makes a simple but effective change: instead of giving each head its own K and V projections, share a single K and single V across all heads. Each head still gets its own Q.

```
Standard Multi-Head (n_heads=8, d_model=64, d_k=8):
  Q: 8 heads, each d_k=8  -> KV cache per token: 8 x 8 = 64 floats
  K: 8 heads, each d_k=8  -> KV cache per token: 8 x 8 = 64 floats
  V: 8 heads, each d_k=8  -> KV cache per token: 8 x 8 = 64 floats

Multi-Query (n_heads=8, d_model=64, d_k=8):
  Q: 8 heads, each d_k=8  -> same as MHA
  K: 1 head,  d_k=8       -> KV cache per token: 8 floats  <- shared
  V: 1 head,  d_k=8       -> KV cache per token: 8 floats  <- shared
```

Why does this matter? The KV cache. During inference, you store one K and one V vector per token per layer. With 32 layers and 32 heads at $d_k = 128$, that's $32 \times 32 \times 128 \times 2 = 262{,}144$ floats per token. For a 100k context, that's ~100GB in float32. MQA cuts the KV cache by the number of heads (32x in this case), bringing it down to ~3GB.

```python
class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Each head gets its own Q projection
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        # Single shared K and V projection
        self.W_K = nn.Linear(d_model, self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, self.d_k, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        Q = self.W_Q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # K, V: (B, 1, T, d_k) - broadcast across heads
        K = self.W_K(x).view(B, T, 1, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, 1, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        attn_out = weights @ V

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_O(attn_out)
```

The K and V tensors have shape `(B, 1, T, d_k)` and PyTorch broadcasts them across all `n_heads` during the matrix multiplications. No data duplication, just pointer tricks.

```python
torch.manual_seed(42)
mqa = MultiQueryAttention(d_model=64, n_heads=8)
x = torch.randn(2, 16, 64)
out = mqa(x)
print(f"Output shape: {out.shape}")

q_params = sum(p.numel() for p in [mqa.W_Q.weight])
kv_params = sum(p.numel() for p in [mqa.W_K.weight, mqa.W_V.weight])
print(f"Q params: {q_params}, KV params: {kv_params}")
print(f"KV reduction vs MHA: {8}x")
# Output shape: torch.Size([2, 16, 64])
# Q params: 4096, KV params: 1024
# KV reduction vs MHA: 8x
```

The trade-off: some quality loss, because all heads now share the same key-value representation. In practice the degradation is small (0.3-0.5% on benchmarks), but it's measurable.

**When is this used?** PaLM (2022), Falcon, StarCoder, and several inference-optimized models. It's most valuable when you're serving many concurrent requests and KV cache memory is the bottleneck. For training, the compute savings are minimal since there's no KV cache.

## Grouped-Query Attention: The Middle Ground

GQA, introduced by [Ainslie et al. (2023)](https://arxiv.org/abs/2305.13245), splits the difference between standard multi-head attention and MQA. Instead of one shared KV head or one per query head, you use a small number of KV head groups.

```
Multi-Head Attention (MHA):   8 query heads, 8 KV heads  (1:1)
Grouped-Query Attention (GQA): 8 query heads, 2 KV heads  (4:1)
Multi-Query Attention (MQA):  8 query heads, 1 KV head   (8:1)

GQA with n_kv_heads=2, n_heads=8:

  Queries:  [head0, head1, head2, head3] [head4, head5, head6, head7]
               │       │      │      │      │       │      │      │
  KV groups:  [────── kv0 ──────────]    [────── kv1 ──────────────]
```

Four query heads share one KV head. This gives you most of MQA's memory savings while keeping more representational capacity in the keys and values.

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_heads
        self.heads_per_group = n_heads // n_kv_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        Q = self.W_Q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        K = self.W_K(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)

        # Repeat KV heads to match query heads
        # (B, n_kv_heads, T, d_k) -> (B, n_heads, T, d_k)
        K = K.repeat_interleave(self.heads_per_group, dim=1)
        V = V.repeat_interleave(self.heads_per_group, dim=1)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        attn_out = weights @ V

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_O(attn_out)
```

```python
torch.manual_seed(42)
gqa = GroupedQueryAttention(d_model=64, n_heads=8, n_kv_heads=2)
x = torch.randn(2, 16, 64)
out = gqa(x)
print(f"Output shape: {out.shape}")

kv_params = sum(p.numel() for p in [gqa.W_K.weight, gqa.W_V.weight])
mha_kv_params = 2 * 64 * 64  # what MHA would use
print(f"KV params: {kv_params} (vs {mha_kv_params} for MHA)")
print(f"KV cache reduction: {8 // 2}x vs MHA")
# Output shape: torch.Size([2, 16, 64])
# KV params: 2048 (vs 8192 for MHA)
# KV cache reduction: 4x vs MHA
```

The `repeat_interleave` call copies each KV head to match the query heads in its group. Alternatively, you can use `expand()` to avoid the copy and just broadcast, which is what optimized implementations do.

Here's the comparison table:

| Variant | KV Heads (for 32 Q heads) | KV Cache Size | Quality | Used In |
|---------|---------------------------|---------------|---------|---------|
| MHA     | 32                        | 1x (baseline) | Best    | GPT-2, GPT-3, BERT |
| GQA     | 4-8                       | 4-8x smaller  | ~MHA    | Llama 2/3, Mistral, Gemma |
| MQA     | 1                         | 32x smaller   | Slightly worse | PaLM, Falcon, StarCoder |

**When is this used?** Llama 2 70B, Llama 3, Mistral 7B, Gemma, and most new models released after mid-2023. GQA has become the default choice because it gets ~95% of MQA's memory savings with nearly zero quality loss. The Llama 2 paper showed you can even take an MHA-trained model and "uptrain" it to GQA with a small fraction of the original training budget.

## Sparse Attention: Breaking the Quadratic Wall

Standard attention computes scores between every pair of tokens. For a sequence of length $n$, that's $O(n^2)$ operations and $O(n^2)$ memory. Double the context length, quadruple the cost. At 100k tokens, the attention matrix alone is $100{,}000 \times 100{,}000 = 10$ billion entries per head per layer.

Sparse attention patterns solve this by only computing a subset of the attention scores. The idea, introduced by [Child et al. (2019)](https://arxiv.org/abs/1904.10509) in "Generating Long Sequences with Sparse Transformers", is that most attention weights are near zero anyway. If you can identify which ones matter, you can skip the rest.

The most common sparse patterns:

```
Full attention (standard):     Strided sparse:          Fixed sparse:
  ████████                       █ █ █ █                  ████
  ████████                       █ █ █ █                  ████
  ████████                       █ █ █ █                  ████
  ████████                       █ █ █ █                  ████
  ████████                       ████████                      ████
  ████████                       ████████                      ████
  ████████                       ████████                      ████
  ████████                       ████████                      ████

  O(n^2)                         O(n√n)                   O(n√n)
```

In strided sparse, each token attends to every $s$-th token (stride) plus a local window. In fixed sparse, the sequence is split into blocks and tokens attend within their block plus to a set of "summary" positions.

Here's a practical implementation of block-sparse attention:

```python
def block_sparse_attention(Q, K, V, block_size=4):
    """
    Each block of tokens attends to itself and adjacent blocks.
    Q, K, V: (seq_len, d_k)
    """
    T, d_k = Q.shape
    n_blocks = (T + block_size - 1) // block_size
    output = torch.zeros_like(Q)

    for i in range(n_blocks):
        q_start = i * block_size
        q_end = min(q_start + block_size, T)
        q_block = Q[q_start:q_end]

        # Attend to current block and one block on each side
        k_start = max(0, (i - 1) * block_size)
        k_end = min((i + 2) * block_size, T)
        k_block = K[k_start:k_end]
        v_block = V[k_start:k_end]

        scores = q_block @ k_block.T / math.sqrt(d_k)
        weights = F.softmax(scores, dim=-1)
        output[q_start:q_end] = weights @ v_block

    return output
```

```python
torch.manual_seed(42)
T, d_k = 16, 8
Q = torch.randn(T, d_k)
K = torch.randn(T, d_k)
V = torch.randn(T, d_k)

out_full = scaled_dot_product_attention(Q, K, V)
out_sparse = block_sparse_attention(Q, K, V, block_size=4)

# Compare: sparse only sees local context
print(f"Full attention output[0]:   {out_full[0, :4].tolist()}")
print(f"Sparse attention output[0]: {out_sparse[0, :4].tolist()}")
print("(Different because sparse can only see block 0-1)")
```

The complexity drops from $O(n^2)$ to $O(n \cdot b)$ where $b$ is the effective neighborhood size (block size times number of neighbor blocks). For long sequences, this is the difference between "fits in memory" and "doesn't."

**When is this used?** BigBird (Google), Longformer (AI2), and various long-context models. In practice, pure sparse attention is less common in 2025-era LLMs. Most modern models use sliding window attention (next section) or combine sparse patterns with full attention in alternating layers.

## Sliding Window Attention: Local Context, Global Reach

Sliding window attention is a specific, clean form of sparse attention. Each token attends to only a fixed window of $w$ tokens around it. Introduced in Longformer and popularized by [Mistral 7B](https://arxiv.org/abs/2310.06825).

```
Window size w=3 (attend to 3 tokens back):

Token:    t0   t1   t2   t3   t4   t5   t6   t7
t0 sees:  [t0]
t1 sees:  [t0, t1]
t2 sees:  [t0, t1, t2]
t3 sees:  [t1, t2, t3]       <- window slides
t4 sees:  [t2, t3, t4]
t5 sees:  [t3, t4, t5]
t6 sees:  [t4, t5, t6]
t7 sees:  [t5, t6, t7]

Attention matrix (1 = attends, . = masked):
     t0 t1 t2 t3 t4 t5 t6 t7
t0 [ 1  .  .  .  .  .  .  . ]
t1 [ 1  1  .  .  .  .  .  . ]
t2 [ 1  1  1  .  .  .  .  . ]
t3 [ .  1  1  1  .  .  .  . ]
t4 [ .  .  1  1  1  .  .  . ]
t5 [ .  .  .  1  1  1  .  . ]
t6 [ .  .  .  .  1  1  1  . ]
t7 [ .  .  .  .  .  1  1  1 ]
```

The math is the same, just a different mask. For a window of size $w$:

$$M_{ij} = \begin{cases} 0 & \text{if } i - w < j \leq i \\ -\infty & \text{otherwise} \end{cases}$$

```python
def sliding_window_attention(Q, K, V, window_size=3):
    """
    Q, K, V: (seq_len, d_k)
    window_size: number of past tokens to attend to
    """
    T, d_k = Q.shape

    # Build sliding window mask
    mask = torch.zeros(T, T, dtype=torch.bool)
    for i in range(T):
        start = max(0, i - window_size + 1)
        mask[i, start:i+1] = True

    scores = Q @ K.T / math.sqrt(d_k)
    scores = scores.masked_fill(~mask, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    return weights @ V
```

```python
torch.manual_seed(42)
T, d_k = 8, 4
Q = torch.randn(T, d_k)
K = torch.randn(T, d_k)
V = torch.randn(T, d_k)

out = sliding_window_attention(Q, K, V, window_size=3)
print(f"Output shape: {out.shape}")

# Show the mask
mask = torch.zeros(T, T, dtype=torch.bool)
for i in range(T):
    start = max(0, i - 3 + 1)
    mask[i, start:i+1] = True
print("\nWindow mask:")
for row in mask:
    print("".join(["1 " if x else ". " for x in row]))
```

```
Output shape: torch.Size([8, 4])

Window mask:
1 . . . . . . .
1 1 . . . . . .
1 1 1 . . . . .
. 1 1 1 . . . .
. . 1 1 1 . . .
. . . 1 1 1 . .
. . . . 1 1 1 .
. . . . . 1 1 1
```

"But wait, if token 7 can only see tokens 5-7 directly, how does it get information from token 0?" Through stacking layers. In a model with $L$ layers, information can propagate $L \times w$ positions. Mistral 7B uses $w = 4{,}096$ with 32 layers, giving an effective receptive field of $32 \times 4{,}096 = 131{,}072$ tokens. That's how you get long context without quadratic scaling.

The memory savings are real. Instead of a KV cache that grows linearly with sequence length, you only need to store the last $w$ key-value pairs per layer. Once a token slides out of the window, its KV entry gets recycled. This is done with a rolling buffer:

```python
class SlidingWindowKVCache:
    def __init__(self, window_size, n_layers, d_k):
        self.w = window_size
        self.keys = [torch.zeros(window_size, d_k) for _ in range(n_layers)]
        self.values = [torch.zeros(window_size, d_k) for _ in range(n_layers)]
        self.pos = 0

    def update(self, layer, k, v):
        idx = self.pos % self.w  # circular buffer
        self.keys[layer][idx] = k
        self.values[layer][idx] = v

    def get(self, layer):
        return self.keys[layer], self.values[layer]

    def advance(self):
        self.pos += 1
```

Fixed memory, regardless of how long the conversation gets.

**When is this used?** Mistral 7B was the model that popularized this approach. Mistral uses sliding window in some layers and full attention in others (alternating pattern). Mixtral follows the same strategy. Many newer models use a similar hybrid: full attention for some layers to maintain global context, sliding window for others to save memory.

## Flash Attention: Same Math, Different Memory

Flash Attention, introduced by [Dao et al. (2022)](https://arxiv.org/abs/2205.14135), is the most impactful attention optimization of the past few years. The key insight: standard attention is bottlenecked not by compute, but by memory bandwidth.

Here's the problem. Standard attention materializes the full $n \times n$ attention matrix in GPU high-bandwidth memory (HBM):

```
Standard attention memory flow:

1. Compute S = QK^T          -> write n×n matrix to HBM     (SLOW)
2. Read S, compute softmax   -> read n×n from HBM           (SLOW)
3. Write softmax result P    -> write n×n to HBM             (SLOW)
4. Read P, compute PV        -> read n×n from HBM            (SLOW)

Total HBM reads/writes: O(n^2) each step
```

GPU SRAM (on-chip cache) is ~100x faster than HBM, but much smaller (20MB vs 40-80GB on an A100). Flash Attention restructures the computation to work in tiles that fit in SRAM, never materializing the full attention matrix:

```
Flash Attention memory flow:

For each tile of Q (fits in SRAM):
    For each tile of K, V (fits in SRAM):
        1. Compute tile of QK^T     -> stays in SRAM        (FAST)
        2. Update running softmax   -> stays in SRAM         (FAST)
        3. Accumulate tile of PV    -> stays in SRAM         (FAST)
    Write final output tile to HBM                           (one write)

Total HBM reads/writes: O(n^2 / SRAM_size)
```

The math is identical to standard attention. Same inputs, same outputs, same gradients. The trick is an online softmax algorithm that processes the attention matrix in tiles without needing the full matrix at once.

The online softmax works like this. For standard softmax, you need the max over all elements (for numerical stability) before computing anything. The tiled version keeps a running max and a running sum, correcting for new tiles as they arrive:

```python
def online_softmax_demo(scores_list):
    """
    Process softmax in chunks, maintaining running statistics.
    scores_list: list of score chunks (simulating tiles)
    """
    running_max = float('-inf')
    running_sum = 0.0
    weighted_values = []

    for chunk_scores in scores_list:
        # New max across this chunk
        chunk_max = max(chunk_scores)
        new_max = max(running_max, chunk_max)

        # Correction factor for previous sum
        correction = math.exp(running_max - new_max) if running_max != float('-inf') else 0
        running_sum = running_sum * correction

        # Add this chunk's contribution
        chunk_exps = [math.exp(s - new_max) for s in chunk_scores]
        running_sum += sum(chunk_exps)
        running_max = new_max

        weighted_values.append((chunk_exps, new_max))

    # Final softmax values
    all_weights = []
    for chunk_exps, chunk_max in weighted_values:
        correction = math.exp(chunk_max - running_max)
        all_weights.extend([e * correction / running_sum for e in chunk_exps])
    return all_weights
```

```python
# Verify: online softmax matches standard softmax
scores = [1.2, 0.5, -0.3, 2.1, 0.8, -1.0, 0.3, 1.5]

# Standard softmax
max_s = max(scores)
exps = [math.exp(s - max_s) for s in scores]
total = sum(exps)
standard = [e / total for e in exps]

# Online softmax (processed in chunks of 3)
chunks = [scores[0:3], scores[3:6], scores[6:8]]
online = online_softmax_demo(chunks)

print("Standard:", [f"{x:.4f}" for x in standard])
print("Online:  ", [f"{x:.4f}" for x in online])
print(f"Max diff: {max(abs(a-b) for a,b in zip(standard, online)):.2e}")
```

```
Standard: ['0.1489', '0.0739', '0.0332', '0.3662', '0.0998', '0.0165', '0.0605', '0.2010']
Online:   ['0.1489', '0.0739', '0.0332', '0.3662', '0.0998', '0.0165', '0.0605', '0.2010']
Max diff: 6.94e-18
```

Exact same result, computed without ever storing all scores simultaneously.

In practice, you don't implement Flash Attention yourself. It's a CUDA kernel that PyTorch exposes via `F.scaled_dot_product_attention`:

```python
# PyTorch >= 2.0 uses Flash Attention automatically
# when inputs are on CUDA and in float16/bfloat16

# This dispatches to FlashAttention-2 on compatible hardware:
out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

# You can check which backend was used:
# torch.backends.cuda.flash_sdp_enabled()
```

The speedup numbers:

| Sequence Length | Standard Attention | Flash Attention | Speedup |
|----------------|-------------------|-----------------|---------|
| 1,024          | 1.0x              | 1.3x            | 1.3x    |
| 4,096          | 1.0x              | 2.4x            | 2.4x    |
| 16,384         | OOM               | 3.1x            | works   |
| 65,536         | OOM               | 3.8x            | works   |

The longer the sequence, the bigger the win, because HBM bandwidth becomes more of a bottleneck relative to compute. Flash Attention 2 improved on the original by better partitioning work across GPU warps and reducing non-matmul operations.

**When is this used?** Basically everywhere in 2025-2026. PyTorch's `scaled_dot_product_attention` uses Flash Attention by default when available. Every major framework (vLLM, TGI, llama.cpp) uses it. If you're running inference or training on a modern GPU with sequences longer than a few hundred tokens, Flash Attention is what's running under the hood.

## Putting It All Together: What Modern Models Actually Use

Let's map these attention variants to the models you're actually working with:

| Model | Attention Type | KV Heads | Context | Window | Flash |
|-------|---------------|----------|---------|--------|-------|
| GPT-2 (2019) | MHA | 12 | 1,024 | Full | No |
| GPT-3 (2020) | MHA | 96 | 2,048 | Full | No |
| PaLM (2022) | MQA | 1 | 2,048 | Full | Yes |
| Llama 1 (2023) | MHA | 32 | 2,048 | Full | Yes |
| Llama 2 70B (2023) | GQA | 8 | 4,096 | Full | Yes |
| Mistral 7B (2023) | GQA | 8 | 32,768 | 4,096 | Yes |
| Llama 3 (2024) | GQA | 8 | 128,000 | Full | Yes |
| Gemma 2 (2024) | GQA + Sliding | 4-8 | 8,192 | Alternating | Yes |
| DeepSeek V2 (2024) | MLA | 1 (compressed) | 128,000 | Full | Yes |

A few patterns to notice:

**GQA won.** After Llama 2 showed that GQA matches MHA quality with a fraction of the KV cache, everyone adopted it. The 8 KV heads number appears repeatedly because it's a sweet spot: enough capacity for quality, small enough for efficient serving.

**Flash Attention is everywhere.** It went from research paper to industry standard in about a year. Any model released after 2023 assumes Flash Attention is available.

**Window attention is a practical choice.** Mistral proved that you don't need full quadratic attention for every layer. Alternating full and windowed attention across layers gives you global reach (through stacking) with local efficiency. Gemma 2 followed the same idea.

**Context lengths exploded.** From GPT-2's 1,024 tokens to Llama 3's 128,000. This happened because of three compounding factors: Flash Attention making long sequences computationally feasible, GQA/MQA reducing KV cache memory, and techniques like RoPE (Rotary Position Embeddings) that generalize to unseen positions.

## The Decision Tree

If you're building or fine-tuning a model and need to pick an attention variant, here's how to think about it:

**Choosing KV head strategy:**
- Training a new model with < 3B params? MHA is fine, the KV cache overhead is small.
- Training a new model with > 7B params? GQA with 4-8 KV heads. This is the industry default now.
- Extremely memory-constrained inference (edge devices, massive batch sizes)? Consider MQA, but benchmark quality first.

**Choosing attention pattern:**
- Context < 8k tokens? Full attention. The quadratic cost is negligible at this scale.
- Context 8k-32k? Full attention with Flash Attention. Still manageable.
- Context > 32k? Consider alternating full and sliding window layers (the Mistral pattern).
- Context > 128k? You'll likely need a combination of techniques: sliding window, GQA, Flash Attention, and possibly some form of sparse attention or attention sinks.

**The implementation choice:**
- If using PyTorch: `F.scaled_dot_product_attention` handles the kernel dispatch for you (Flash Attention, memory-efficient attention, or math fallback based on hardware and dtype).
- If serving with vLLM or TGI: they handle PagedAttention for KV cache management automatically.
- If implementing from scratch for learning: start with the naive version, understand it, then use the optimized library calls.

## Beyond Attention: What's Coming

The attention mechanism has been the core of the transformer since 2017, and every model released since then uses some variant of it. But the design space keeps evolving.

Multi-head Latent Attention (MLA), used in DeepSeek V2, compresses the KV cache into a learned low-rank latent space, getting even smaller than GQA. Linear attention variants like RWKV and Mamba replace the softmax attention entirely with recurrent-style computations that scale linearly with sequence length. These aren't mainstream yet for LLMs, but they're competitive on several benchmarks and have O(1) memory per token during inference.

The 2017 paper asked if attention is all you need. Nine years later, the answer is nuanced: attention is definitely what you need, but which attention and how you compute it matters more than anyone expected. The Q, K, V math from the original paper is still there in every model. Everything built on top of it (multi-head, causal masking, GQA, Flash Attention, sliding windows) is the engineering that makes it work at scale. If you understand each layer of that stack, you can make informed decisions about which pieces to use and when. That's the whole point.
