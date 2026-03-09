---
layout: post
title: "LoRA From the Ground Up: The Math, the Matrices, and the Merge"
date: 2025-12-28 10:00:00 -0800
categories: [llm, fine-tuning]
tags: [lora, peft, fine-tuning, low-rank, linear-algebra, merging]
series: fine-tuning
author: cmenguy
colab_url: "https://colab.research.google.com/github/cmenguy/cmenguy.github.io/blob/main/notebooks/2025-12-28-lora-deep-dive.ipynb"
colab_embed: false
github_notebook: "https://github.com/cmenguy/cmenguy.github.io/blob/main/notebooks/2025-12-28-lora-deep-dive.ipynb"
notebook_description: "Hands-on LoRA implementation from scratch: building, training, merging, and inspecting low-rank adapters with PyTorch."
---

In my [last post](/llm/fine-tuning/2025/12/22/fine-tuning-llms-sft-dpo-rlhf/), I walked through SFT, DPO, and RLHF for fine-tuning LLMs. Throughout that entire post, LoRA kept showing up in every code example, every training config, every `LoraConfig(r=16, lora_alpha=32)` call. I used it the way most of us do: copy the config from a tutorial, set `r=16` because that's what everyone uses, set `lora_alpha` to double the rank because... reasons, and move on. The model trains, the loss goes down, the outputs improve. Ship it.

But a few days ago I got into a discussion with a colleague about fine-tuning efficiency: how much memory we were actually saving with LoRA, whether we could push the rank lower without hurting quality, whether it even mattered which layers we targeted. I had opinions on all of this, but when I tried to back them up with anything beyond "it worked last time," I realized I was hand-waving. I knew *what* LoRA did at a high level (low-rank matrices, fewer parameters, memory efficient), but I couldn't actually explain *why* those specific numbers mattered. What does rank even mean in this context? Why does `lora_alpha` scale the way it does? What's actually happening to the weight matrices during training? I'd been treating LoRA like a black box with good defaults, and that bothered me.

So I blocked out a weekend, pulled up the [original paper](https://arxiv.org/abs/2106.09685), and went through the math line by line. What follows is what I wish someone had explained to me before I started using LoRA in production.

## The Problem LoRA Solves

Let's start with why LoRA exists. A model like Llama 3.1 8B has roughly 8 billion parameters. Full fine-tuning means updating all of them: every weight in every layer gets a gradient, an optimizer state, and a momentum term. For Adam, that's 3x the model size in memory just for the optimizer. On a Llama 8B in float32, that's:

$$\text{Memory}_{\text{full}} = 8\text{B} \times 4\text{ bytes} \times 3 = 96\text{ GB (optimizer alone)}$$

Add the model weights, gradients, and activations, and you're looking at needing multiple A100 80GB GPUs just for fine-tuning. For most teams, that's impractical.

LoRA's insight: when you fine-tune a large model on a specific task, the weight updates don't use the full dimensionality of the weight matrices. The *change* in weights during fine-tuning is low-rank. It lies in a much smaller subspace than the original weights. So instead of updating a giant matrix, you can decompose the update into two small matrices and only train those.

## The Core Idea: Low-Rank Decomposition

Here's the key equation. For a pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA constrains the update $\Delta W$ to be a low-rank decomposition:

$$W = W_0 + \Delta W = W_0 + BA$$

Where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, with rank $r \ll \min(d, k)$.

That's it. That's the whole trick. Instead of learning a $d \times k$ matrix of updates (potentially millions of parameters), you learn two smaller matrices whose product has the same shape but far fewer total parameters.

Let me make this concrete with a picture. Say you have a weight matrix in a transformer attention layer with $d = 4096$ and $k = 4096$:

```
Full fine-tuning:                  LoRA (r=8):

Update ΔW (4096 × 4096)           B (4096 × 8)    A (8 × 4096)

┌─────────────────────┐            ┌──┐             ┌─────────────────────┐
│                     │            │  │             │                     │
│                     │            │  │             └─────────────────────┘
│                     │            │  │              8 × 4096 = 32,768
│    16,777,216       │            │  │              parameters
│    parameters       │            │  │
│                     │            │  │
│                     │            │  │
│                     │            │  │
└─────────────────────┘            └──┘
                                    4096 × 8 = 32,768
                                    parameters

Total full: 16,777,216             Total LoRA: 65,536 (0.4%)
```

With $r = 8$, you're training 65,536 parameters instead of 16.7 million — a **256x reduction** for this single layer. Across the entire model, LoRA typically trains 0.1-1% of the total parameters.

## The Forward Pass: How It Actually Computes

During a forward pass, the original weight and the LoRA update combine like this. For an input $x$:

$$h = W_0 x + \Delta W x = W_0 x + BAx$$

Here's what that looks like step by step:

```
Input x ─────────────────┬───────────────────── Output h
(batch × k)              │                      (batch × d)
                         │
              ┌──────────┴──────────┐
              │                     │
         Pretrained path       LoRA path
              │                     │
           W₀ · x              B · (A · x)
         (frozen)           (trainable)
              │                     │
              └──────────┬──────────┘
                         │
                        (+)  ← element-wise add
                         │
                      Output h
```

The pretrained weights $W_0$ stay **completely frozen**: no gradients, no optimizer states, no memory overhead. Only $B$ and $A$ receive gradients. This is why LoRA is so memory-efficient: you only store optimizer states for the tiny adapter matrices, not the full model.

Let's implement this from scratch in PyTorch so you can see exactly what's happening:

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, original_layer, r=8, lora_alpha=16):
        super().__init__()
        self.original = original_layer
        self.original.weight.requires_grad = False  # freeze

        d, k = original_layer.weight.shape
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
```

A few things to notice here. The original layer is frozen (`requires_grad = False`). And there's a `scaling` factor that we'll come back to shortly. Now the adapter matrices:

```python
        # A is initialized with Kaiming uniform (like the paper)
        self.lora_A = nn.Parameter(torch.empty(r, k))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

        # B is initialized to zero so ΔW = BA = 0 at start
        self.lora_B = nn.Parameter(torch.zeros(d, r))
```

This initialization is critical. $B$ starts at zero, which means $\Delta W = BA = 0$ at the beginning of training. The model starts producing exactly the same outputs as the pretrained model. Training then gradually learns the update. $A$ uses Kaiming uniform initialization to break symmetry.

The forward pass puts it all together:

```python
    def forward(self, x):
        # Original frozen path
        base_output = self.original(x)
        # LoRA path: x → A → B → scale
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_output + lora_output
```

Two separate matrix multiplications through the bottleneck: $x \cdot A^T$ compresses to rank $r$, then $\cdot B^T$ projects back up, plus the scaling factor. Let's see the parameter savings in action:

```python
d, k, r = 4096, 4096, 8
full_params = d * k
lora_params = (d * r) + (r * k)
print(f"Full fine-tuning: {full_params:,} parameters")
print(f"LoRA (r={r}):     {lora_params:,} parameters")
print(f"Reduction:        {full_params / lora_params:.0f}x")
```

```
Full fine-tuning: 16,777,216 parameters
LoRA (r=8):       65,536 parameters
Reduction:        256x
```

## What Rank Actually Means

The rank $r$ is LoRA's most important hyperparameter, and it's worth building intuition about what it controls.

In linear algebra, the rank of a matrix is the number of linearly independent rows (or equivalently, columns). A rank-$r$ matrix can be expressed as the sum of $r$ rank-1 outer products. Think of it as the number of "independent directions" the matrix can push information through.

When we constrain $\Delta W = BA$ with $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, the product $BA$ has rank at most $r$. This means the weight update can only modify the model's behavior along $r$ independent directions in the weight space.

The original LoRA paper found something surprising: even $r = 1$ or $r = 2$ works reasonably well for many tasks. The weight updates during fine-tuning really are low-rank. Here's an intuition for why: when you fine-tune on a specific task (like marketing copy), you're not rewiring the model's entire understanding of language. You're making a targeted adjustment: "write in this style" or "prefer these patterns." That adjustment occupies a small subspace of what the model's weights can represent.

Here's a practical way to see this. Let's create a weight update, compute its singular values, and see how the energy concentrates:

```python
import torch

torch.manual_seed(42)
d, k = 4096, 4096

# Simulate a fine-tuning weight update
delta_W = torch.randn(d, k) * 0.01  # small perturbation
U, S, Vt = torch.linalg.svd(delta_W, full_matrices=False)

# How much "energy" is captured by the top-r singular values?
total_energy = (S ** 2).sum()
for r in [1, 2, 4, 8, 16, 32, 64]:
    captured = (S[:r] ** 2).sum() / total_energy * 100
    print(f"r={r:3d}: {captured:.2f}% of energy captured")
```

```
r=  1: 0.10% of energy captured
r=  2: 0.19% of energy captured
r=  4: 0.39% of energy captured
r=  8: 0.77% of energy captured
r= 16: 1.52% of energy captured
r= 32: 3.00% of energy captured
r= 64: 5.85% of energy captured
```

A random matrix spreads its energy uniformly across all singular values, which is why even $r = 64$ only captures ~2%. But real fine-tuning updates aren't random. They concentrate on a few directions that matter for the task. In practice, $r = 8$ or $r = 16$ captures the meaningful signal while ignoring noise.

### Choosing Rank in Practice

| Rank | Trainable params (per 4096×4096 layer) | Use case |
|------|---------------------------------------|----------|
| 4    | 32,768   | Simple style transfer, narrow tasks |
| 8    | 65,536   | Most fine-tuning tasks (good default) |
| 16   | 131,072  | Complex tasks, multi-skill learning |
| 32   | 262,144  | Approaching diminishing returns |
| 64   | 524,288  | Rarely needed; consider full fine-tuning |

The sweet spot for most tasks is $r \in [8, 16]$. Going higher adds parameters without proportional improvement. Going lower risks underfitting complex tasks.

## The Scaling Factor: Why lora_alpha Exists

If you've ever stared at `lora_alpha=32` in a config and wondered what it does, here's the answer. The LoRA forward pass applies a scaling factor:

$$h = W_0 x + \frac{\alpha}{r} \cdot BAx$$

Where $\alpha$ is `lora_alpha` and $r$ is the rank. This $\frac{\alpha}{r}$ scaling serves a critical purpose: it **decouples the learning rate from the rank**.

Without this scaling, changing the rank would change the magnitude of the LoRA update. If you double $r$, you'd roughly double the norm of $BA$ (more parameters contributing to the output), and you'd need to halve the learning rate to compensate. The $\frac{\alpha}{r}$ factor normalizes this away.

Here's the practical implication. When `lora_alpha = 2 * r` (the common convention), the scaling factor is $\frac{2r}{r} = 2$. The LoRA update gets amplified by 2x. This means:

```python
# These two configs behave similarly despite different ranks:
# Config A: r=8,  alpha=16 → scaling = 16/8  = 2.0
# Config B: r=16, alpha=32 → scaling = 32/16 = 2.0
# Config C: r=8,  alpha=8  → scaling = 8/8   = 1.0  (no amplification)
```

You can think of `lora_alpha` as a "volume knob" for the LoRA update. Higher alpha amplifies the adapter's effect. The convention of `alpha = 2 * r` works well in practice, but you can tune it, especially if you notice training instability (lower alpha) or the model not learning fast enough (higher alpha).

Let's see this in action:

```python
import torch

d, k, r = 128, 128, 8
x = torch.randn(1, k)
B = torch.randn(d, r) * 0.01
A = torch.randn(r, k) * 0.01

for alpha in [4, 8, 16, 32]:
    scaling = alpha / r
    lora_out = (x @ A.T @ B.T) * scaling
    print(f"alpha={alpha:2d}, scaling={scaling:.1f}, "
          f"output norm={lora_out.norm():.4f}")
```

```
alpha= 4, scaling=0.5, output norm=0.0250
alpha= 8, scaling=1.0, output norm=0.0501
alpha=16, scaling=2.0, output norm=0.1002
alpha=32, scaling=4.0, output norm=0.2003
```

Linear relationship: double the alpha, double the output magnitude. The learning rate and scaling factor interact, which is why the convention of fixing `alpha = 2r` and tuning only the learning rate is the pragmatic approach.

## Which Layers Get LoRA?

In a transformer, LoRA is typically applied to the attention projection matrices. Looking at a standard multi-head attention block:

```
Input
  │
  ├──→ Q = W_q · x     ← LoRA here (q_proj)
  ├──→ K = W_k · x     ← LoRA here (k_proj)
  ├──→ V = W_v · x     ← LoRA here (v_proj)
  │
  │    Attention(Q, K, V)
  │
  └──→ O = W_o · attn   ← LoRA here (o_proj)
```

The original paper applied LoRA only to $W_q$ and $W_v$, but modern practice targets all four attention projections. Some people also include the MLP layers (`gate_proj`, `up_proj`, `down_proj`), though the marginal benefit varies.

Here's the config you'll see in most production setups:

```python
from peft import LoraConfig

# Standard config — attention projections only
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
```

And if you want to be more aggressive:

```python
# Extended config — attention + MLP
config_extended = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    task_type="CAUSAL_LM",
)
```

Let's count the parameter difference across a full model:

```python
# Llama 3.1 8B has 32 transformer layers
num_layers = 32
d_model = 4096
d_ffn = 14336  # MLP intermediate size
r = 16

# Attention-only LoRA
attn_params = num_layers * 4 * (2 * d_model * r)
# Attention + MLP LoRA
mlp_params = num_layers * 3 * ((d_model * r) + (d_ffn * r))
total_attn = attn_params
total_all = attn_params + mlp_params

print(f"Attention-only LoRA: {total_attn:>12,} params")
print(f"Attention + MLP LoRA: {total_all:>11,} params")
print(f"Full model:           8,000,000,000 params")
print(f"Attn LoRA %:          {total_attn/8e9*100:.2f}%")
print(f"All LoRA %:           {total_all/8e9*100:.2f}%")
```

```
Attention-only LoRA:    16,777,216 params
Attention + MLP LoRA:   45,088,768 params
Full model:           8,000,000,000 params
Attn LoRA %:          0.21%
All LoRA %:           0.56%
```

Even the aggressive "all projections" approach trains less than 1% of the model. That's LoRA's superpower.

## A Complete Training Example

Let's put all the pieces together with a real training example. We'll fine-tune a small model so you can actually run this, and inspect the LoRA matrices at each stage.

First, let's create a minimal dataset and load a model with LoRA:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch

model_name = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float32
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

```
trainable params: 460,800 || all params: 134,975,808 || trainable%: 0.3414
```

Only 0.34% of parameters are trainable. Let's inspect what the LoRA matrices look like before training:

```python
# Grab the LoRA matrices from the first layer's q_proj
q_lora = model.model.model.layers[0].self_attn.q_proj
lora_A = q_lora.lora_A.default.weight.data
lora_B = q_lora.lora_B.default.weight.data

print(f"lora_A shape: {lora_A.shape}, norm: {lora_A.norm():.4f}")
print(f"lora_B shape: {lora_B.shape}, norm: {lora_B.norm():.4f}")
print(f"ΔW = B·A norm: {(lora_B @ lora_A).norm():.4f}")
```

```
lora_A shape: torch.Size([8, 576]), norm: 1.6235
lora_B shape: torch.Size([576, 8]), norm: 0.0000
ΔW = B·A norm: 0.0000
```

Exactly as expected. $A$ is initialized with random values, $B$ is all zeros, so $\Delta W = BA = 0$. The model starts as if no adapter exists.

Now let's train it on a few examples and see how the matrices change:

```python
from trl import SFTConfig, SFTTrainer
from datasets import Dataset

# Simple dataset for demonstration
data = Dataset.from_list([
    {"text": "The capital of France is Paris."},
    {"text": "Machine learning models learn from data."},
    {"text": "PyTorch is a deep learning framework."},
    {"text": "Transformers use self-attention mechanisms."},
] * 50)  # repeat for a few epochs worth of data

tokenizer.pad_token = tokenizer.eos_token
training_args = SFTConfig(
    output_dir="./lora-demo",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=1e-3,
    logging_steps=25,
    max_length=64,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=data,
    processing_class=tokenizer,
)

trainer.train()
```

After training, let's check the matrices again:

```python
lora_A_after = q_lora.lora_A.default.weight.data
lora_B_after = q_lora.lora_B.default.weight.data
delta_W = lora_B_after @ lora_A_after

print(f"lora_A norm: {lora_A_after.norm():.4f}")
print(f"lora_B norm: {lora_B_after.norm():.4f}")
print(f"ΔW = B·A norm: {delta_W.norm():.4f}")
print(f"ΔW shape: {delta_W.shape}")
print(f"ΔW rank: ≤ {lora_config.r} (by construction)")
```

$B$ is no longer zero. Training has learned a low-rank update. The model's behavior has shifted, but only along 8 independent directions in the weight space.

## Merging: Collapsing the Adapter Into the Model

This is where things get practically interesting. You've trained your LoRA adapter. Now what? You have two options: keep the adapter separate, or merge it into the base model. The choice has real implications for serving.

### What Merging Means

Merging is just matrix addition. You take the pretrained weight $W_0$ and permanently add the LoRA update:

$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} \cdot BA$$

After merging, the model is a regular model again: no adapter, no separate matrices, no extra computation at inference time.

```
Before merge (inference):           After merge (inference):

  x ──┬──→ W₀·x ──┐                  x ──→ W_merged·x ──→ output
      │             (+) → output
      └──→ B·A·x ──┘                  One matrix multiply.
                                       No adapter overhead.
  Two paths computed.
  Adapter stored separately.
```

Here's how you do it in code:

```python
from peft import PeftModel

# Option 1: merge and get a regular model
merged_model = model.merge_and_unload()

# The merged model is a standard transformers model now
# No LoRA layers, no adapters: just modified weights
merged_model.save_pretrained("./merged-model")
```

Let's verify the merge is mathematically correct:

```python
import torch

# Before: base weight + LoRA
W0 = q_lora.weight.data.clone()       # original frozen weight
scaling = lora_config.lora_alpha / lora_config.r
delta_W = (lora_B_after @ lora_A_after) * scaling

# Manually compute the merged weight
W_manual = W0 + delta_W

# Compare with what PEFT computes
W_peft = merged_model.model.layers[0].self_attn.q_proj.weight.data

print(f"Manual merge matches PEFT: "
      f"{torch.allclose(W_manual, W_peft, atol=1e-5)}")
```

```
Manual merge matches PEFT: True
```

Just matrix addition with scaling. Nothing mysterious.

### What Happens If You Don't Merge

If you skip the merge, the LoRA adapter stays separate from the base model. This isn't just an academic distinction; it affects both performance and flexibility.

**Inference overhead.** Without merging, every forward pass computes two paths: the base model path and the LoRA path. For a single request, the overhead is small. But at scale, those extra matrix multiplications add up:

```python
import torch
import time

d, k, r = 4096, 4096, 16
x = torch.randn(32, k)  # batch of 32

W0 = torch.randn(d, k)
B = torch.randn(d, r)
A = torch.randn(r, k)
W_merged = W0 + B @ A

# Benchmark: merged (single matmul) vs unmerged (two paths)
def bench_merged(x, W, n=1000):
    for _ in range(n):
        _ = x @ W.T

def bench_unmerged(x, W0, A, B, n=1000):
    for _ in range(n):
        _ = x @ W0.T + x @ A.T @ B.T

t0 = time.perf_counter()
bench_merged(x, W_merged)
t_merged = time.perf_counter() - t0

t0 = time.perf_counter()
bench_unmerged(x, W0, A, B)
t_unmerged = time.perf_counter() - t0

print(f"Merged:   {t_merged:.3f}s")
print(f"Unmerged: {t_unmerged:.3f}s")
print(f"Overhead: {(t_unmerged/t_merged - 1)*100:.1f}%")
```

The exact overhead depends on hardware, but expect 5-15% extra latency on the forward pass. Not catastrophic, but not free.

**Multi-adapter serving.** Here's the flip side: not merging is actually a *feature* when you need to serve multiple adapters. If you have one base model and 50 brand-specific LoRA adapters (like the marketing scenario from the previous post), you can:

```python
from peft import PeftModel

# Load base model once (most of the memory)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load different adapters on the fly
model = PeftModel.from_pretrained(base_model, "./adapter-brand-A")
output_A = model.generate(...)

# Switch to a different adapter (near instant)
model.load_adapter("./adapter-brand-B", adapter_name="brand_b")
model.set_adapter("brand_b")
output_B = model.generate(...)
```

Each adapter is a few megabytes. The base model is tens of gigabytes. Without merging, you store one base model + N tiny adapters instead of N full model copies. That's the difference between needing 1 GPU and needing 50.

```
Merged approach:                     Adapter approach:

┌────────────────────┐               ┌────────────────────┐
│ Merged Model A     │  ~16 GB       │ Base Model         │  ~16 GB
│ (base + adapter A) │               │ (shared)           │
└────────────────────┘               └────────────────────┘
┌────────────────────┐                   │
│ Merged Model B     │  ~16 GB          ┌┴─────┐
│ (base + adapter B) │               ┌──┤      ├──┐
└────────────────────┘               │  └──────┘  │
┌────────────────────┐            ┌──┴──┐      ┌──┴──┐
│ Merged Model C     │  ~16 GB   │ A   │      │ B   │  ~10 MB each
│ (base + adapter C) │           │     │      │     │
└────────────────────┘           └─────┘      └─────┘
                                              ┌─────┐
Total: ~48 GB                                 │ C   │
                                              └─────┘
                                 Total: ~16.03 GB
```

### When to Merge vs. Keep Separate

| Scenario | Merge? | Why |
|----------|--------|-----|
| Single model in production | Yes | No overhead, simpler serving |
| Multiple adapters, one base | No | Memory efficient, hot-swappable |
| Distributing fine-tuned model | Yes | Easier for users, no PEFT dependency |
| Continued training / experimentation | No | Keep adapter separate for iteration |
| Stacking adapters (LoRA + LoRA) | No | Can combine multiple adapters |

### The Merge in Detail

Let's trace exactly what `merge_and_unload` does under the hood. It's simple but worth understanding:

```python
import torch

# Simulating the merge process
d, k, r = 512, 512, 8
alpha, rank = 16, 8
scaling = alpha / rank

# Pretrained weight (frozen during training)
W0 = torch.randn(d, k)

# Learned LoRA matrices
B = torch.randn(d, r) * 0.01
A = torch.randn(r, k) * 0.01

# Step 1: compute the low-rank update
delta_W = B @ A  # (d × r) @ (r × k) = (d × k)
print(f"ΔW shape: {delta_W.shape}")

# Step 2: apply the scaling factor
delta_W_scaled = delta_W * scaling
print(f"Scaling factor (α/r): {scaling}")

# Step 3: add to original weights
W_merged = W0 + delta_W_scaled
print(f"W0 norm:      {W0.norm():.2f}")
print(f"ΔW norm:      {delta_W_scaled.norm():.2f}")
print(f"W_merged norm: {W_merged.norm():.2f}")
```

The merged weight is a regular matrix. No special structure, no adapter overhead. But you lose the ability to "un-merge"; the adapter's contribution is baked into the weights permanently.

## Practical Tips From Production

A few things I've learned the hard way that the paper doesn't tell you:

**Start with r=8 and alpha=16.** This is a good default for 7B-13B parameter models on most tasks. Only increase rank if you see clear signs of underfitting (training loss not decreasing fast enough despite reasonable learning rate).

**Learning rate matters more than rank.** The learning rate for LoRA should typically be 5-10x higher than what you'd use for full fine-tuning. This is because you're only updating a small subset of parameters, so they need to move more per step to have the same overall effect. Start with `2e-4` and adjust from there.

**Dropout is your friend for small datasets.** `lora_dropout=0.05` is the default, but if you're training on fewer than 1000 examples, bump it to `0.1`. The low-rank bottleneck is already a form of regularization, but it's not always enough.

**Save adapters, not merged models**, at least during development. A LoRA adapter for a 7B model is ~10-50 MB. A merged model is ~14 GB. When you're running dozens of experiments, that storage difference matters.

**Double-check your target modules.** Different model families have different linear layer names. Llama uses `q_proj`, `k_proj`, `v_proj`, `o_proj`. Other models might use `query`, `key`, `value`, or `qkv_proj`. Check with:

```python
from peft import LoraConfig

# Let PEFT figure out the right layer names
config = LoraConfig(r=8, target_modules="all-linear")
# Or inspect the model manually:
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(f"{name}: {module.weight.shape}")
```

## Wrapping Up

LoRA's elegance is in how simple it actually is once you see the math. Freeze the pretrained weights, learn a low-rank update decomposed into two small matrices, and add it to the forward pass with a scaling factor. That's the whole algorithm. The rest is engineering: choosing which layers to target, setting the rank and scaling, deciding whether to merge for serving or keep adapters separate for flexibility.

The next time you write `LoraConfig(r=16, lora_alpha=32)`, you'll know exactly what those numbers mean and why they matter. And when someone on your team asks "can we make r bigger?" you'll be able to explain *what* it actually changes in the weight space, not just *whether* to do it.
