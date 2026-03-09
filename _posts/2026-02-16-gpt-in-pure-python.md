---
layout: post
title: "GPT in 200 Lines of Pure Python: Every Line Explained"
date: 2026-02-16 10:00:00 -0800
categories: [deep-learning, llm]
tags: [gpt, transformer, attention, backpropagation, autograd, karpathy, from-scratch]
series: transformers
author: cmenguy
colab_url: "https://colab.research.google.com/github/cmenguy/cmenguy.github.io/blob/main/notebooks/2026-02-16-gpt-in-pure-python.ipynb"
colab_embed: false
github_notebook: "https://github.com/cmenguy/cmenguy.github.io/blob/main/notebooks/2026-02-16-gpt-in-pure-python.ipynb"
notebook_description: "The complete microgpt.py code with inline explanations, runnable end-to-end in pure Python with zero dependencies."
---

I was scrolling through X late on a Friday night (as one does) when I spotted a post from Andrej Karpathy linking to a new [gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). The title: *"The most atomic way to train and run inference for a GPT in pure, dependency-free Python."* I clicked through, started reading, and an hour later I was still sitting there. Not because the code was long. The opposite. It was about 200 lines. No PyTorch, no NumPy, no dependencies at all. Just `math`, `random`, and `os`. And it trains a working GPT model.

That hit me in a specific way. I've been working with transformers for years at this point, and most of that time is spent at a level of abstraction where you're calling `model = AutoModelForCausalLM.from_pretrained(...)` and trusting that the 47 layers of library code underneath are doing the right thing. You know the architecture conceptually. You can sketch the attention diagram on a whiteboard. But could you write it from scratch, with no libraries, and get it to actually learn something? I wasn't sure I could, and that bothered me.

So I spent the weekend going through Karpathy's code line by line, making sure I understood every piece, every derivative, every design choice. What follows is that exercise: a complete walkthrough of a GPT implementation in pure Python, mapped back to the concepts from [Attention Is All You Need](https://arxiv.org/abs/1706.03762) and the broader transformer literature. If you're someone who uses these models daily but hasn't looked under the hood in a while, this is your excuse to refresh the fundamentals.

## The Setup: Data and Tokenization

The code starts by downloading a dataset of names (from Karpathy's [makemore](https://github.com/karpathy/makemore) project) and building a character-level tokenizer.

```python
import os
import math
import random

random.seed(42)

if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
```

Nothing fancy here. We download a list of ~32,000 names, strip whitespace, and shuffle. Each name is a "document" in this mini language model. Here's what the raw data looks like:

```
docs[0]: "yuheng"
docs[1]: "diondre"
docs[2]: "xavien"
docs[3]: "jori"
docs[4]: "juanluis"
docs[5]: "erandi"
```

The model's job: learn the statistical patterns of how characters follow each other in English names, then generate new ones.

The tokenizer is as simple as it gets:

```python
uchars = sorted(set(''.join(docs)))  # unique characters -> token ids 0..n-1
BOS = len(uchars)                     # special Beginning of Sequence token
vocab_size = len(uchars) + 1          # +1 for BOS
```

This gives us 27 tokens: 26 lowercase letters plus a BOS (beginning-of-sequence) token. The mapping is just alphabetical order:

```
a=0, b=1, c=2, d=3, e=4, f=5, g=6, h=7, i=8, j=9, k=10, l=11, m=12,
n=13, o=14, p=15, q=16, r=17, s=18, t=19, u=20, v=21, w=22, x=23, y=24, z=25
BOS=26
```

In production LLMs, you'd use something like BPE with a vocab of 32k-128k tokens. Here, character-level keeps things dead simple. The BOS token is used both to signal the start and end of a name, which is a common trick in small language models.

## Autograd: Building the Computation Graph

Here's where things get interesting. Real frameworks like PyTorch give you automatic differentiation for free. Karpathy builds his own from scratch in about 50 lines. This is the `Value` class, and it's the engine that makes backpropagation work.

### The Value Node

```python
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads
```

Each `Value` holds four things:
- `data`: the actual scalar number (forward pass result)
- `grad`: the gradient of the loss with respect to this node (filled in during backward pass)
- `_children`: the inputs that produced this node
- `_local_grads`: the partial derivatives of this node with respect to each child

The `__slots__` line is a Python optimization that prevents dynamic attribute creation, saving memory. When you're about to create thousands of `Value` objects per training step, that matters.

### Arithmetic Operations

Each math operation creates a new `Value` and records the local gradient. This is the chain rule made explicit:

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data + other.data, (self, other), (1, 1))

def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data * other.data, (self, other), (other.data, self.data))
```

For addition $f(a, b) = a + b$: the partial derivatives are $\frac{\partial f}{\partial a} = 1$ and $\frac{\partial f}{\partial b} = 1$. That's the `(1, 1)` tuple.

For multiplication $f(a, b) = a \cdot b$: the partial derivatives are $\frac{\partial f}{\partial a} = b$ and $\frac{\partial f}{\partial b} = a$. That's the `(other.data, self.data)` tuple. Classic product rule.

### Power, Log, Exp, and ReLU

```python
def __pow__(self, other):
    return Value(self.data**other, (self,), (other * self.data**(other-1),))

def log(self):
    return Value(math.log(self.data), (self,), (1/self.data,))

def exp(self):
    return Value(math.exp(self.data), (self,), (math.exp(self.data),))

def relu(self):
    return Value(max(0, self.data), (self,), (float(self.data > 0),))
```

Same pattern. The power rule: $\frac{d}{dx} x^n = n \cdot x^{n-1}$. The log derivative: $\frac{d}{dx} \ln(x) = \frac{1}{x}$. The exp derivative: $\frac{d}{dx} e^x = e^x$ (the function is its own derivative, which is why exponentials are so useful in math). ReLU: gradient is 1 if positive, 0 otherwise.

### The Backward Pass

This is the heart of backpropagation:

```python
def backward(self):
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)
            topo.append(v)

    build_topo(self)
    self.grad = 1
    for v in reversed(topo):
        for child, local_grad in zip(v._children, v._local_grads):
            child.grad += local_grad * v.grad
```

Two things happening here. First, a topological sort of the computation graph (DFS post-order). This gives us an ordering where every node appears after all of its children. Then we walk backwards through that order, applying the chain rule at each node.

The chain rule says: if $\frac{\partial L}{\partial v}$ is the gradient at node $v$, and $v$ was computed from child $c$ with local gradient $\frac{\partial v}{\partial c}$, then:

$$\frac{\partial L}{\partial c} \mathrel{+}= \frac{\partial v}{\partial c} \cdot \frac{\partial L}{\partial v}$$

That `+=` is key. A child node might be used by multiple parents (think of a variable used in two different expressions). Each parent contributes to the child's gradient, and they accumulate additively. This is the multivariate chain rule in action.

Setting `self.grad = 1` at the top makes sense because $\frac{\partial L}{\partial L} = 1$: the loss's gradient with respect to itself is 1.

### Helper Operations

The remaining methods define subtraction, division, and reverse operations:

```python
def __neg__(self):       return self * -1
def __radd__(self, other): return self + other
def __sub__(self, other):  return self + (-other)
def __rsub__(self, other): return other + (-self)
def __rmul__(self, other): return self * other
def __truediv__(self, other):  return self * other**-1
def __rtruediv__(self, other): return other * self**-1
```

Division is implemented as multiplication by the inverse ($a / b = a \cdot b^{-1}$), so the power rule and multiplication rule handle the gradient automatically. The `__r*__` methods handle cases where a plain number is on the left side: `3 + value` calls `value.__radd__(3)`.

## Model Parameters: What the Network Knows

Now we set up the model's parameters. These are the learnable weights that store everything the model learns during training.

```python
n_layer = 1       # number of transformer layers
n_embd = 16       # embedding dimension
block_size = 16   # maximum context length
n_head = 4        # number of attention heads
head_dim = n_embd // n_head  # = 4, dimension per head
```

This is a tiny GPT: 1 layer, 16-dimensional embeddings, 4 attention heads. GPT-2 Small used 12 layers, 768-dimensional embeddings, and 12 heads. GPT-3 used 96 layers and 12,288 dimensions. But the architecture is identical. Everything scales.

```python
matrix = lambda nout, nin, std=0.08: [
    [Value(random.gauss(0, std)) for _ in range(nin)]
    for _ in range(nout)
]
```

This helper creates a 2D list of `Value` objects initialized from a Gaussian distribution with standard deviation 0.08. In PyTorch you'd use `torch.randn(nout, nin) * std`. The small std keeps initial values near zero, which helps training stability.

### The State Dict

```python
state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd),
}

for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
```

This mirrors exactly what you'd see in a PyTorch GPT model's `state_dict()`. Here's what each matrix does:

| Matrix | Shape | Purpose |
|--------|-------|---------|
| `wte` | (27, 16) | Token embedding: maps each token ID to a 16-d vector |
| `wpe` | (16, 16) | Position embedding: maps each position to a 16-d vector |
| `lm_head` | (27, 16) | Output projection: maps hidden states back to vocab logits |
| `attn_wq` | (16, 16) | Query projection for attention |
| `attn_wk` | (16, 16) | Key projection for attention |
| `attn_wv` | (16, 16) | Value projection for attention |
| `attn_wo` | (16, 16) | Output projection after attention heads are concatenated |
| `mlp_fc1` | (64, 16) | First MLP layer (expand 16 -> 64) |
| `mlp_fc2` | (16, 64) | Second MLP layer (contract 64 -> 16) |

The MLP expansion factor of 4x is standard in transformers. The original "Attention Is All You Need" paper used $d_{ff} = 2048$ with $d_{model} = 512$, which is the same 4:1 ratio.

Finally, all parameters get flattened into a single list for the optimizer:

```python
params = [p for mat in state_dict.values() for row in mat for p in row]
```

This gives us 4,192 parameters. GPT-2 had 117 million. Same idea, different scale.

## The Model Architecture: A GPT Forward Pass

Now for the actual model. The `gpt()` function takes a single token and its position, runs it through the transformer, and returns logits (unnormalized scores) over the vocabulary.

### Building Blocks: Linear, Softmax, RMSNorm

Before the main model, three utility functions:

```python
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

A matrix-vector multiply. For each row `wo` in weight matrix `w`, compute the dot product with input `x`. If `w` is shape (m, n) and `x` has length n, the output has length m. This is the fundamental operation in neural networks: $y = Wx$.

```python
def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]
```

Softmax converts raw scores into a probability distribution. The formula from [Attention Is All You Need](https://arxiv.org/abs/1706.03762):

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

The `max_val` subtraction is a numerical stability trick. Since $\text{softmax}(z) = \text{softmax}(z - c)$ for any constant $c$, subtracting the max prevents overflow in the exponential. This doesn't change the result but keeps numbers in a range where `math.exp` won't explode.

```python
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

RMSNorm (Root Mean Square Normalization) is a simpler alternative to LayerNorm. Instead of subtracting the mean and dividing by std (like LayerNorm), it just divides by the RMS:

$$\text{RMSNorm}(x_i) = \frac{x_i}{\sqrt{\frac{1}{n}\sum_{j} x_j^2 + \epsilon}}$$

The original transformer paper used LayerNorm, but most modern architectures (Llama, Gemma, etc.) switched to RMSNorm because it's cheaper and works just as well. No bias subtraction, no learned affine parameters here. The $\epsilon = 10^{-5}$ prevents division by zero.

### The GPT Function

Now the main event. I'll walk through the `gpt()` function section by section. It processes one token at a time (as opposed to a full sequence), keeping a running KV cache for previous positions.

```python
def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)
```

**Step 1: Embedding.** Look up the token embedding and position embedding, add them element-wise. This is Section 3.4 of the original paper: "We use learned embeddings to convert the input tokens and output tokens to vectors of dimension $d_{model}$. [...] we also use the usual learned positional embeddings."

The sum of token + position gives the model both "what token is this?" and "where in the sequence is it?" in a single vector. Let's trace what happens when we feed in BOS (token 26) at position 0, the first step when processing "yuheng":

```
tok_emb (wte[26]):  [+0.0821, -0.0360, -0.1205, -0.0332, ...]  (16 values)
pos_emb (wpe[0]):   [-0.0222, -0.0696, -0.1805, +0.0569, ...]  (16 values)
x = tok + pos:      [+0.0599, -0.1056, -0.3010, +0.0237, ...]  (element-wise add)
x after rmsnorm:    [+0.4951, -0.8726, -2.4877, +0.1957, ...]  (normalized)
```

Small random numbers get added together, then RMSNorm rescales them. At initialization this is all noise, but after training these embeddings will encode meaningful character and position information. RMSNorm stabilizes the values before they enter the transformer layers.

```python
    for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
```

**Step 2: Compute Q, K, V.** This is the setup for attention. The input `x` is projected three times through different weight matrices to produce queries (Q), keys (K), and values (V). From Section 3.2.1 of the paper: "The queries, keys and values are [...] linear projections."

The key and value vectors get appended to a running cache (`keys[li]`, `values[li]`), which stores the K and V vectors from all previous positions. This is the **KV cache** that makes autoregressive generation efficient: you don't need to recompute K and V for positions you've already processed.

Continuing our trace of BOS at position 0:

```
q (query):  [+0.0347, +0.3948, -0.3227, +0.2163, ...]  (16 values)
k (key):    [-0.0677, -0.0466, +0.2234, -0.4468, ...]  (16 values)
v (value):  [-0.2006, -0.1163, +0.6520, +0.1435, ...]  (16 values)
```

Three different 16-d vectors from the same input. Q asks "what am I looking for?", K says "what do I contain?", V says "what information should I pass forward if attended to?"

Note the `x_residual = x` before the norm. This sets up the residual connection, which we'll use after attention.

```python
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
```

**Step 3: Split into heads.** The full Q, K, V vectors (dimension 16) get split into 4 heads of dimension 4 each. This is multi-head attention: instead of one big attention operation, we run 4 smaller ones in parallel, each attending to different aspects of the input. Section 3.2.2: "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions."

```python
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)
```

**Step 4: Scaled dot-product attention scores.** For each past position `t`, compute the dot product between the current query and that position's key, then scale by $\sqrt{d_k}$. This is the core attention formula from the paper:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Why the $\sqrt{d_k}$ scaling? The paper explains it directly (Section 3.2.1): for large $d_k$, the dot products grow large in magnitude, pushing the softmax into regions with extremely small gradients. Dividing by $\sqrt{d_k}$ keeps the variance of the dot product at 1 regardless of dimension.

In our trace, this is position 0 so there's only one key in the cache (itself). The attention weight is trivially 1.0:

```
head 0 attn_logits:  [-0.0947]  (dot product of q·k, scaled)
head 0 attn_weights: [1.0000]   (softmax of a single value is always 1.0)
```

At position 3 processing "h", there would be 4 keys in the cache (BOS, y, u, h) and the attention weights would be a distribution over all four, like `[0.18, 0.31, 0.22, 0.29]`. That's when attention gets interesting: the model learns which previous characters are most relevant for predicting what comes next.

Note there's no causal mask here. Since we process one token at a time and only store keys/values up to the current position, causality is enforced implicitly. The model can only attend to positions it has already seen.

```python
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)
```

**Step 5: Weighted sum of values.** The attention weights (a probability distribution over all past positions) are used to compute a weighted average of the value vectors. Positions with higher attention scores contribute more to the output. Each head's output is concatenated into `x_attn`.

This is the "V" part of the attention formula: after softmax gives us the weights, we multiply them against the values to get the head's output.

```python
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
```

**Step 6: Output projection and residual connection.** The concatenated multi-head output passes through one more linear layer (`attn_wo`), then gets added back to the residual. Section 3.2.2: "Multi-head attention [...] $\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$".

The residual connection ($x = x + \text{attention}(x)$) is what makes deep transformers trainable. Without it, gradients would vanish or explode through many layers. With it, gradients have a "highway" to flow directly from the loss back to early layers.

```python
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
```

**Step 7: Feed-forward (MLP) block.** Section 3.3 of the paper: "Each of the layers in our encoder and decoder contains a fully connected feed-forward network [...] $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$."

The pattern is: expand (16 -> 64), nonlinearity (ReLU), contract (64 -> 16), residual add. The expansion gives the model more "workspace" to compute nonlinear features. The original paper used ReLU; GPT-2 switched to GELU. This code uses ReLU for simplicity (since it's trivially differentiable).

In the trace, the expand-relu-contract pattern looks like this:

```
after fc1 (64d):   [+0.4231, -0.0127, +0.5659, ...]  (expanded to 64 dims)
after relu:        32/64 neurons active                 (half are zeroed out)
after fc2 + resid: [+0.4800, -1.0092, -2.5156, ...]   (back to 16 dims)
```

ReLU kills roughly half the neurons, which introduces sparsity. The surviving neurons carry the signal forward. The residual connection wraps the entire MLP block. No biases in this implementation, which follows modern practice (Llama, PaLM, and others all dropped biases).

```python
    logits = linear(x, state_dict['lm_head'])
    return logits
```

**Step 8: Project to vocabulary.** The final hidden state gets projected to vocab-sized logits through the `lm_head` matrix. Each element of the output represents how likely that token is to come next. These are raw scores (logits), not probabilities. Here's the end of our trace:

```
logits (27 values): [+0.005, +0.239, +0.543, -0.477, +0.351, ...]

After softmax -> probabilities:
  P("o") = 0.0617   (top prediction, wrong)
  P("s") = 0.0545
  P("c") = 0.0541
  P("z") = 0.0506
  P("k") = 0.0497
  ...
  P("y") = 0.0266   (actual target)

loss = -log(0.0266) = 3.63
```

At initialization, the model's predictions are basically random. It assigns probability 0.027 to "y" (the correct next token after BOS for "yuheng"), which isn't far from uniform random chance (1/27 = 0.037). Training will push that probability up over 1000 steps. The training loop applies softmax to the logits and computes the loss.

## The Training Loop

With the model defined, here's how it learns.

### Optimizer Setup

```python
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)  # first moment (mean of gradients)
v = [0.0] * len(params)  # second moment (mean of squared gradients)
```

Adam optimizer, initialized from scratch. Two buffers: `m` tracks the exponential moving average of gradients (momentum), `v` tracks the exponential moving average of squared gradients (adaptive learning rate per parameter). This is why Adam uses 2x the memory of vanilla SGD.

### The Loop

```python
num_steps = 1000

for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
```

Each step picks one document (cycling through the shuffled list), tokenizes it to a list of integers, and wraps it with BOS tokens on both sides. For the name "yuheng", here's what tokenization produces:

```
characters: ['y', 'u', 'h', 'e', 'n', 'g']
token ids:  [24,  20,  7,   4,   13,  6 ]
with BOS:   [26,  24,  20,  7,   4,   13,  6,  26]
```

The BOS at the start tells the model "a name is starting." The BOS at the end acts as an end-of-sequence signal: the model learns that after the last character of a name, BOS comes next, which during generation tells us to stop.

```python
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []

    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)

    loss = (1 / n) * sum(losses)
```

The forward pass processes each token position sequentially, building up the KV cache. At each position, the model predicts the next token. For "yuheng", the training pairs look like this:

```
pos 0: "BOS" -> "y"    (given start-of-name, predict first letter)
pos 1: "y"   -> "u"    (given y, predict next letter)
pos 2: "u"   -> "h"
pos 3: "h"   -> "e"
pos 4: "e"   -> "n"
pos 5: "n"   -> "g"
pos 6: "g"   -> "BOS"  (given last letter, predict end-of-name)
```

The loss is the negative log-likelihood of the correct next token:

$$\mathcal{L} = -\frac{1}{n}\sum_{t=1}^{n} \log P(x_t \mid x_{<t})$$

This is cross-entropy loss. If the model assigns probability 1.0 to the correct next token, $-\log(1.0) = 0$, and the loss is zero. If it assigns 0.01, $-\log(0.01) = 4.6$. The model is incentivized to put as much probability mass as possible on the correct next token at every position.

### Backward and Adam Update

```python
    loss.backward()

    lr_t = learning_rate * (1 - step / num_steps)

    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0
```

`loss.backward()` triggers the backward pass through the entire computation graph, populating `.grad` on every `Value` node. Then the Adam update rule, which was introduced in [Kingma & Ba, 2015](https://arxiv.org/abs/1412.6980):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_t = \theta_{t-1} - \alpha_t \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

The bias correction ($\hat{m}$ and $\hat{v}$) compensates for the fact that `m` and `v` are initialized to zero, which biases them toward zero in early steps. The learning rate decays linearly from 0.01 to 0 over training. After updating each parameter, its gradient is reset to 0 for the next step.

## Inference: Generating New Names

After training, the model generates text autoregressively:

```python
temperature = 0.5

for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []

    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(
            range(vocab_size), weights=[p.data for p in probs]
        )[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])

    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
```

The generation loop: start with BOS, feed it through the model, get logits, apply temperature scaling, sample the next token, repeat until BOS appears again (end of name) or we hit the max length.

Temperature controls the "sharpness" of the distribution. Before softmax, each logit is divided by the temperature:

$$P(x_i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

At $T = 1.0$, you get the raw model distribution. At $T < 1.0$ (like 0.5 here), the distribution gets peakier, making the model more likely to pick high-probability tokens. At $T \to 0$, it becomes greedy (always picks the most likely token). At $T > 1.0$, the distribution flattens, producing more "creative" but less coherent output.

`random.choices` does weighted sampling from the distribution. This is the stochastic part that gives you different outputs each time, unlike greedy decoding.

## Mapping to "Attention Is All You Need"

Here's how Karpathy's ~200 lines map to the original transformer paper:

| Paper Section | Paper Concept | microgpt.py Implementation |
|---------------|---------------|---------------------------|
| 3.1 | Encoder-Decoder Architecture | Decoder-only (GPT-style), no encoder |
| 3.2.1 | Scaled Dot-Product Attention | `attn_logits` computation with `/ head_dim**0.5` |
| 3.2.2 | Multi-Head Attention | Loop over `n_head` with split Q, K, V |
| 3.2.3 | Causal Masking | Implicit via sequential processing + KV cache |
| 3.3 | Position-wise FFN | `mlp_fc1` (expand) -> ReLU -> `mlp_fc2` (contract) |
| 3.4 | Embeddings | `wte` (token) + `wpe` (position) |
| 5.3 | Optimizer | Adam with linear LR decay |

Key differences from the original paper:
- **Decoder-only**: the paper described an encoder-decoder model for translation. GPT dropped the encoder and cross-attention, keeping only the decoder with causal self-attention.
- **RMSNorm instead of LayerNorm**: simpler and cheaper. Modern LLMs almost universally use RMSNorm.
- **ReLU instead of GELU**: GPT-2 used GELU, but ReLU is easier to differentiate by hand and works fine here.
- **No biases**: the original paper used biases everywhere. Modern practice (Llama, PaLM, Gemma) dropped them with no loss in quality.
- **No dropout**: with ~4k parameters and 32k training examples, regularization isn't the bottleneck.

## What "Everything Else Is Just Efficiency" Means

Karpathy's gist opens with: *"This file is the complete algorithm. Everything else is just efficiency."* That line is doing a lot of work. Here's what "everything else" means:

**Tensor operations.** This code does scalar math: each multiply is one `Value * Value`. PyTorch batches these into matrix operations on GPU, getting 1000x+ speedups. The algorithm is identical. The hardware parallelism is the efficiency.

**Batching.** This code processes one document per step. Real training packs hundreds of sequences into a batch, running them through the model in parallel. Same gradients in expectation, but much better GPU utilization.

**Mixed precision.** Training in float16 or bfloat16 instead of float32 halves memory and doubles throughput on modern GPUs. The math is the same, just with fewer bits.

**Parallelism.** Data parallel, tensor parallel, pipeline parallel, FSDP. All of them are strategies for splitting the same computation across multiple GPUs. The fundamental operations stay the same.

**Flash Attention.** A memory-efficient reformulation of the exact same attention computation, using tiling to avoid materializing the full attention matrix. Same result, 2-4x faster.

Every one of these is a performance optimization layered on top of the same core algorithm that Karpathy expressed in 200 lines of Python. If you understand this code, you understand what GPT-4, Claude, and every other transformer-based model is doing at its core. The rest is engineering.

## Running It Yourself

The code trains in about 30 minutes on a laptop CPU (no GPU needed). After 1000 steps, it generates plausible-looking names: things that could be English names but aren't. The loss drops from ~3.3 (random guessing across 27 tokens: $-\log(1/27) \approx 3.3$) to around 2.0, meaning the model is assigning roughly 7x higher probability to correct tokens than chance.

You can find the full code in the [companion notebook](https://colab.research.google.com/github/cmenguy/cmenguy.github.io/blob/main/notebooks/2026-02-16-gpt-in-pure-python.ipynb) or grab it directly from [Karpathy's gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).

What I took away from this exercise: the gap between "I understand transformers" and "I can implement one from scratch" is worth closing. You don't need to do it for production work. But the next time you're debugging a training run and the loss is doing something weird, or you're trying to understand why a specific hyperparameter matters, having that ground-level intuition makes a real difference. The algorithm fits in 200 lines. The understanding takes a weekend. Worth the trade.
