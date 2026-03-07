# GPT in 200 Lines of Pure Python: Every Line Explained

---

**The complete microgpt.py code with inline explanations, runnable end-to-end in pure Python with zero dependencies.**

[Run in Google Colab](https://colab.research.google.com/github/cmenguy/cmenguy.github.io/blob/main/notebooks/2026-02-16-gpt-in-pure-python.ipynb) | [View on GitHub](https://github.com/cmenguy/cmenguy.github.io/blob/main/notebooks/2026-02-16-gpt-in-pure-python.ipynb)

---

I was scrolling through X late on a Friday night (as one does) when I spotted a post from Andrej Karpathy linking to a new [gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). The title: *"The most atomic way to train and run inference for a GPT in pure, dependency-free Python."* I clicked through, started reading, and an hour later I was still sitting there. Not because the code was long. The opposite. It was about 200 lines. No PyTorch, no NumPy, no dependencies at all. Just `math`, `random`, and `os`. And it trains a working GPT model.

That hit me in a specific way. I've been working with transformers for years at this point, and most of that time is spent at a level of abstraction where you're calling `model = AutoModelForCausalLM.from_pretrained(...)` and trusting that the 47 layers of library code underneath are doing the right thing. You know the architecture conceptually. You can sketch the attention diagram on a whiteboard. But could you write it from scratch, with no libraries, and get it to actually learn something? I wasn't sure I could, and that bothered me.

So I spent the weekend going through Karpathy's code line by line, making sure I understood every piece, every derivative, every design choice. What follows is that exercise: a complete walkthrough of a GPT implementation in pure Python, mapped back to the concepts from [Attention Is All You Need](https://arxiv.org/abs/1706.03762) and the broader transformer literature. If you're someone who uses these models daily but hasn't looked under the hood in a while, this is your excuse to refresh the fundamentals.

## 1. The Setup: Data and Tokenization

The code starts by downloading a dataset of names (from Karpathy's [makemore](https://github.com/karpathy/makemore) project) and building a character-level tokenizer.


https://gist.github.com/cmenguy/53234b377cba5b494b8c49095486c008


Nothing fancy here. We download a list of ~32,000 names, strip whitespace, and shuffle. Each name is a "document" in this mini language model. Here's what the raw data looks like:


https://gist.github.com/cmenguy/de625c48d4d6af0a88c45bd9b2720317


The model's job: learn the statistical patterns of how characters follow each other in English names, then generate new ones.

The tokenizer is as simple as it gets:


https://gist.github.com/cmenguy/fe69c3a011c4d09b0a1f3b2dcdb72119


This gives us 27 tokens: 26 lowercase letters plus a BOS (beginning-of-sequence) token. The mapping is just alphabetical order:


https://gist.github.com/cmenguy/a4ee84c97a029fdf322b4dfee36d6673


In production LLMs, you'd use something like BPE with a vocab of 32k-128k tokens. Here, character-level keeps things dead simple. The BOS token is used both to signal the start and end of a name, which is a common trick in small language models.

## 2. Autograd: Building the Computation Graph

Here's where things get interesting. Real frameworks like PyTorch give you automatic differentiation for free. Karpathy builds his own from scratch in about 50 lines. This is the `Value` class, and it's the engine that makes backpropagation work.

#### 2.1 The Value Node


https://gist.github.com/cmenguy/801e725d1cb7e318f62efa093d36911f


Each `Value` holds four things:
- `data`: the actual scalar number (forward pass result)
- `grad`: the gradient of the loss with respect to this node (filled in during backward pass)
- `_children`: the inputs that produced this node
- `_local_grads`: the partial derivatives of this node with respect to each child

The `__slots__` line is a Python optimization that prevents dynamic attribute creation, saving memory. When you're about to create thousands of `Value` objects per training step, that matters.

#### 2.2 Arithmetic Operations

Each math operation creates a new `Value` and records the local gradient. This is the chain rule made explicit:


https://gist.github.com/cmenguy/fe15929463b4ad4527e2d5bdf5b873c9


For addition $f(a, b) = a + b$: the partial derivatives are ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20a%7D%20%3D%201) and ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20b%7D%20%3D%201). That's the `(1, 1)` tuple.

For multiplication ![equation](https://latex.codecogs.com/png.latex?\inline%20f%28a%2C%20b%29%20%3D%20a%20%5Ccdot%20b): the partial derivatives are ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20a%7D%20%3D%20b) and ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20b%7D%20%3D%20a). That's the `(other.data, self.data)` tuple. Classic product rule.

#### 2.3 Power, Log, Exp, and ReLU


https://gist.github.com/cmenguy/5b39d0536b30d2c16e4734227b5c0a20


Same pattern. The power rule: ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Cfrac%7Bd%7D%7Bdx%7D%20x%5En%20%3D%20n%20%5Ccdot%20x%5E%7Bn-1%7D). The log derivative: ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Cfrac%7Bd%7D%7Bdx%7D%20%5Cln%28x%29%20%3D%20%5Cfrac%7B1%7D%7Bx%7D). The exp derivative: ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Cfrac%7Bd%7D%7Bdx%7D%20e%5Ex%20%3D%20e%5Ex) (the function is its own derivative, which is why exponentials are so useful in math). ReLU: gradient is 1 if positive, 0 otherwise.

#### 2.4 The Backward Pass

This is the heart of backpropagation:


https://gist.github.com/cmenguy/685a7073468a6725124200b79de4d292


Two things happening here. First, a topological sort of the computation graph (DFS post-order). This gives us an ordering where every node appears after all of its children. Then we walk backwards through that order, applying the chain rule at each node.

The chain rule says: if ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20v%7D) is the gradient at node $v$, and $v$ was computed from child $c$ with local gradient ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Cfrac%7B%5Cpartial%20v%7D%7B%5Cpartial%20c%7D), then:

![equation](https://latex.codecogs.com/png.latex?\dpi{150}\bg_white%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20c%7D%20%5Cmathrel%7B%2B%7D%3D%20%5Cfrac%7B%5Cpartial%20v%7D%7B%5Cpartial%20c%7D%20%5Ccdot%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20v%7D)

That `+=` is key. A child node might be used by multiple parents (think of a variable used in two different expressions). Each parent contributes to the child's gradient, and they accumulate additively. This is the multivariate chain rule in action.

Setting `self.grad = 1` at the top makes sense because ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20L%7D%20%3D%201): the loss's gradient with respect to itself is 1.

#### 2.5 Helper Operations

The remaining methods define subtraction, division, and reverse operations:


https://gist.github.com/cmenguy/6e2130434274596f659e7675d9e2d0d3


Division is implemented as multiplication by the inverse (![equation](https://latex.codecogs.com/png.latex?\inline%20a%20%2F%20b%20%3D%20a%20%5Ccdot%20b%5E%7B-1%7D)), so the power rule and multiplication rule handle the gradient automatically. The `__r*__` methods handle cases where a plain number is on the left side: `3 + value` calls `value.__radd__(3)`.

## 3. Model Parameters: What the Network Knows

Now we set up the model's parameters. These are the learnable weights that store everything the model learns during training.


https://gist.github.com/cmenguy/30cbf19fe87d9fb62e77ecd8f4f403bf


This is a tiny GPT: 1 layer, 16-dimensional embeddings, 4 attention heads. GPT-2 Small used 12 layers, 768-dimensional embeddings, and 12 heads. GPT-3 used 96 layers and 12,288 dimensions. But the architecture is identical. Everything scales.


https://gist.github.com/cmenguy/ccc7a9d0f17c5dde67de00d78302c634


This helper creates a 2D list of `Value` objects initialized from a Gaussian distribution with standard deviation 0.08. In PyTorch you'd use `torch.randn(nout, nin) * std`. The small std keeps initial values near zero, which helps training stability.

#### 3.1 The State Dict


https://gist.github.com/cmenguy/44984dc3e4d1fcabc0c8440e78f6b334


This mirrors exactly what you'd see in a PyTorch GPT model's `state_dict()`. Here's what each matrix does:


https://gist.github.com/cmenguy/2bf14c4523477fe3f79c194c9f654d0d


The MLP expansion factor of 4x is standard in transformers. The original "Attention Is All You Need" paper used ![equation](https://latex.codecogs.com/png.latex?\inline%20d_%7Bff%7D%20%3D%202048) with ![equation](https://latex.codecogs.com/png.latex?\inline%20d_%7Bmodel%7D%20%3D%20512), which is the same 4:1 ratio.

Finally, all parameters get flattened into a single list for the optimizer:


https://gist.github.com/cmenguy/29a64eadd01836e7ede6b4304a3b781f


This gives us 4,192 parameters. GPT-2 had 117 million. Same idea, different scale.

## 4. The Model Architecture: A GPT Forward Pass

Now for the actual model. The `gpt()` function takes a single token and its position, runs it through the transformer, and returns logits (unnormalized scores) over the vocabulary.

#### 4.1 Building Blocks: Linear, Softmax, RMSNorm

Before the main model, three utility functions:


https://gist.github.com/cmenguy/87428ad7508f3d9f11f29d4b384b0d02


A matrix-vector multiply. For each row `wo` in weight matrix `w`, compute the dot product with input `x`. If `w` is shape (m, n) and `x` has length n, the output has length m. This is the fundamental operation in neural networks: $y = Wx$.


https://gist.github.com/cmenguy/1a359e5cf54d019bdb3e4edf8113e100


Softmax converts raw scores into a probability distribution. The formula from [Attention Is All You Need](https://arxiv.org/abs/1706.03762):

![equation](https://latex.codecogs.com/png.latex?\dpi{150}\bg_white%20%5Ctext%7Bsoftmax%7D%28z_i%29%20%3D%20%5Cfrac%7Be%5E%7Bz_i%7D%7D%7B%5Csum_j%20e%5E%7Bz_j%7D%7D)

The `max_val` subtraction is a numerical stability trick. Since ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Ctext%7Bsoftmax%7D%28z%29%20%3D%20%5Ctext%7Bsoftmax%7D%28z%20-%20c%29) for any constant $c$, subtracting the max prevents overflow in the exponential. This doesn't change the result but keeps numbers in a range where `math.exp` won't explode.


https://gist.github.com/cmenguy/484d4ec9cf840aad3b65fee306412953


RMSNorm (Root Mean Square Normalization) is a simpler alternative to LayerNorm. Instead of subtracting the mean and dividing by std (like LayerNorm), it just divides by the RMS:

![equation](https://latex.codecogs.com/png.latex?\dpi{150}\bg_white%20%5Ctext%7BRMSNorm%7D%28x_i%29%20%3D%20%5Cfrac%7Bx_i%7D%7B%5Csqrt%7B%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bj%7D%20x_j%5E2%20%2B%20%5Cepsilon%7D%7D)

The original transformer paper used LayerNorm, but most modern architectures (Llama, Gemma, etc.) switched to RMSNorm because it's cheaper and works just as well. No bias subtraction, no learned affine parameters here. The ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Cepsilon%20%3D%2010%5E%7B-5%7D) prevents division by zero.

#### 4.2 The GPT Function

Now the main event. I'll walk through the `gpt()` function section by section. It processes one token at a time (as opposed to a full sequence), keeping a running KV cache for previous positions.


https://gist.github.com/cmenguy/f0176d898f73dd8e94e66d46c483fa46


**Step 1: Embedding.** Look up the token embedding and position embedding, add them element-wise. This is Section 3.4 of the original paper: "We use learned embeddings to convert the input tokens and output tokens to vectors of dimension ![equation](https://latex.codecogs.com/png.latex?\inline%20d_%7Bmodel%7D). [...] we also use the usual learned positional embeddings."

The sum of token + position gives the model both "what token is this?" and "where in the sequence is it?" in a single vector. Let's trace what happens when we feed in BOS (token 26) at position 0, the first step when processing "yuheng":


https://gist.github.com/cmenguy/52e511b30ebaf01997f22db665fd9b43


Small random numbers get added together, then RMSNorm rescales them. At initialization this is all noise, but after training these embeddings will encode meaningful character and position information. RMSNorm stabilizes the values before they enter the transformer layers.


https://gist.github.com/cmenguy/5b7b53670de0bdd13c67b1a7ef731227


**Step 2: Compute Q, K, V.** This is the setup for attention. The input `x` is projected three times through different weight matrices to produce queries (Q), keys (K), and values (V). From Section 3.2.1 of the paper: "The queries, keys and values are [...] linear projections."

The key and value vectors get appended to a running cache (`keys[li]`, `values[li]`), which stores the K and V vectors from all previous positions. This is the **KV cache** that makes autoregressive generation efficient: you don't need to recompute K and V for positions you've already processed.

Continuing our trace of BOS at position 0:


https://gist.github.com/cmenguy/ca1a76f92b47746ddbcf128fbd39c8fc


Three different 16-d vectors from the same input. Q asks "what am I looking for?", K says "what do I contain?", V says "what information should I pass forward if attended to?"

Note the `x_residual = x` before the norm. This sets up the residual connection, which we'll use after attention.


https://gist.github.com/cmenguy/406aca856ff30749300c8f36d31c709d


**Step 3: Split into heads.** The full Q, K, V vectors (dimension 16) get split into 4 heads of dimension 4 each. This is multi-head attention: instead of one big attention operation, we run 4 smaller ones in parallel, each attending to different aspects of the input. Section 3.2.2: "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions."


https://gist.github.com/cmenguy/6cd1838c1e0d36753fa447f30014bb6a


**Step 4: Scaled dot-product attention scores.** For each past position `t`, compute the dot product between the current query and that position's key, then scale by ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Csqrt%7Bd_k%7D). This is the core attention formula from the paper:

![equation](https://latex.codecogs.com/png.latex?\dpi{150}\bg_white%20%5Ctext%7BAttention%7D%28Q%2C%20K%2C%20V%29%20%3D%20%5Ctext%7Bsoftmax%7D%5Cleft%28%5Cfrac%7BQK%5ET%7D%7B%5Csqrt%7Bd_k%7D%7D%5Cright%29V)

Why the ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Csqrt%7Bd_k%7D) scaling? The paper explains it directly (Section 3.2.1): for large ![equation](https://latex.codecogs.com/png.latex?\inline%20d_k), the dot products grow large in magnitude, pushing the softmax into regions with extremely small gradients. Dividing by ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Csqrt%7Bd_k%7D) keeps the variance of the dot product at 1 regardless of dimension.

In our trace, this is position 0 so there's only one key in the cache (itself). The attention weight is trivially 1.0:


https://gist.github.com/cmenguy/7c19cd22cd9bec0aa6179b8a7db3cdff


At position 3 processing "h", there would be 4 keys in the cache (BOS, y, u, h) and the attention weights would be a distribution over all four, like `[0.18, 0.31, 0.22, 0.29]`. That's when attention gets interesting: the model learns which previous characters are most relevant for predicting what comes next.

Note there's no causal mask here. Since we process one token at a time and only store keys/values up to the current position, causality is enforced implicitly. The model can only attend to positions it has already seen.


https://gist.github.com/cmenguy/41665ba2c247f75e8645860a3ab181da


**Step 5: Weighted sum of values.** The attention weights (a probability distribution over all past positions) are used to compute a weighted average of the value vectors. Positions with higher attention scores contribute more to the output. Each head's output is concatenated into `x_attn`.

This is the "V" part of the attention formula: after softmax gives us the weights, we multiply them against the values to get the head's output.


https://gist.github.com/cmenguy/66ab70889da3a87956dcc8034a96b00c


**Step 6: Output projection and residual connection.** The concatenated multi-head output passes through one more linear layer (`attn_wo`), then gets added back to the residual. Section 3.2.2: "Multi-head attention [...] ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Ctext%7BMultiHead%7D%28Q%2C%20K%2C%20V%29%20%3D%20%5Ctext%7BConcat%7D%28%5Ctext%7Bhead%7D_1%2C%20...%2C%20%5Ctext%7Bhead%7D_h%29W%5EO)".

The residual connection (![equation](https://latex.codecogs.com/png.latex?\inline%20x%20%3D%20x%20%2B%20%5Ctext%7Battention%7D%28x%29)) is what makes deep transformers trainable. Without it, gradients would vanish or explode through many layers. With it, gradients have a "highway" to flow directly from the loss back to early layers.


https://gist.github.com/cmenguy/84e3e5b62de393b3045713c1122b4bb5


**Step 7: Feed-forward (MLP) block.** Section 3.3 of the paper: "Each of the layers in our encoder and decoder contains a fully connected feed-forward network [...] ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Ctext%7BFFN%7D%28x%29%20%3D%20%5Cmax%280%2C%20xW_1%20%2B%20b_1%29W_2%20%2B%20b_2)."

The pattern is: expand (16 -> 64), nonlinearity (ReLU), contract (64 -> 16), residual add. The expansion gives the model more "workspace" to compute nonlinear features. The original paper used ReLU; GPT-2 switched to GELU. This code uses ReLU for simplicity (since it's trivially differentiable).

In the trace, the expand-relu-contract pattern looks like this:


https://gist.github.com/cmenguy/b52a2d9b4b7b05ae575c3974c0abe92d


ReLU kills roughly half the neurons, which introduces sparsity. The surviving neurons carry the signal forward. The residual connection wraps the entire MLP block. No biases in this implementation, which follows modern practice (Llama, PaLM, and others all dropped biases).


https://gist.github.com/cmenguy/4cf16d4faa323fdd6a92c3c966a3b954


**Step 8: Project to vocabulary.** The final hidden state gets projected to vocab-sized logits through the `lm_head` matrix. Each element of the output represents how likely that token is to come next. These are raw scores (logits), not probabilities. Here's the end of our trace:


https://gist.github.com/cmenguy/4e81676dc6f4b2b2fa45bbaad58d2fb6


At initialization, the model's predictions are basically random. It assigns probability 0.027 to "y" (the correct next token after BOS for "yuheng"), which isn't far from uniform random chance (1/27 = 0.037). Training will push that probability up over 1000 steps. The training loop applies softmax to the logits and computes the loss.

## 5. The Training Loop

With the model defined, here's how it learns.

#### 5.1 Optimizer Setup


https://gist.github.com/cmenguy/508bc4e3f92c6fd705e639782a17ca11


Adam optimizer, initialized from scratch. Two buffers: `m` tracks the exponential moving average of gradients (momentum), `v` tracks the exponential moving average of squared gradients (adaptive learning rate per parameter). This is why Adam uses 2x the memory of vanilla SGD.

#### 5.2 The Loop


https://gist.github.com/cmenguy/05dad841c3da2977aaf29f7791107fa3


Each step picks one document (cycling through the shuffled list), tokenizes it to a list of integers, and wraps it with BOS tokens on both sides. For the name "yuheng", here's what tokenization produces:


https://gist.github.com/cmenguy/df10a8109f839f67cd6672c01958cdcb


The BOS at the start tells the model "a name is starting." The BOS at the end acts as an end-of-sequence signal: the model learns that after the last character of a name, BOS comes next, which during generation tells us to stop.


https://gist.github.com/cmenguy/c271c4b3e7ed167fece61b39ca0e25e1


The forward pass processes each token position sequentially, building up the KV cache. At each position, the model predicts the next token. For "yuheng", the training pairs look like this:


https://gist.github.com/cmenguy/60b3ef8e7668d26bbdaa2281eaf1c0f1


The loss is the negative log-likelihood of the correct next token:

![equation](https://latex.codecogs.com/png.latex?\dpi{150}\bg_white%20%5Cmathcal%7BL%7D%20%3D%20-%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bt%3D1%7D%5E%7Bn%7D%20%5Clog%20P%28x_t%20%5Cmid%20x_%7B%3Ct%7D%29)

This is cross-entropy loss. If the model assigns probability 1.0 to the correct next token, ![equation](https://latex.codecogs.com/png.latex?\inline%20-%5Clog%281.0%29%20%3D%200), and the loss is zero. If it assigns 0.01, ![equation](https://latex.codecogs.com/png.latex?\inline%20-%5Clog%280.01%29%20%3D%204.6). The model is incentivized to put as much probability mass as possible on the correct next token at every position.

#### 5.3 Backward and Adam Update


https://gist.github.com/cmenguy/f715218d194c8b5388383f866f824593


`loss.backward()` triggers the backward pass through the entire computation graph, populating `.grad` on every `Value` node. Then the Adam update rule, which was introduced in [Kingma & Ba, 2015](https://arxiv.org/abs/1412.6980):

![equation](https://latex.codecogs.com/png.latex?\dpi{150}\bg_white%20m_t%20%3D%20%5Cbeta_1%20m_%7Bt-1%7D%20%2B%20%281%20-%20%5Cbeta_1%29%20g_t)

![equation](https://latex.codecogs.com/png.latex?\dpi{150}\bg_white%20v_t%20%3D%20%5Cbeta_2%20v_%7Bt-1%7D%20%2B%20%281%20-%20%5Cbeta_2%29%20g_t%5E2)

![equation](https://latex.codecogs.com/png.latex?\dpi{150}\bg_white%20%5Chat%7Bm%7D_t%20%3D%20%5Cfrac%7Bm_t%7D%7B1%20-%20%5Cbeta_1%5Et%7D%2C%20%5Cquad%20%5Chat%7Bv%7D_t%20%3D%20%5Cfrac%7Bv_t%7D%7B1%20-%20%5Cbeta_2%5Et%7D)

![equation](https://latex.codecogs.com/png.latex?\dpi{150}\bg_white%20%5Ctheta_t%20%3D%20%5Ctheta_%7Bt-1%7D%20-%20%5Calpha_t%20%5Ccdot%20%5Cfrac%7B%5Chat%7Bm%7D_t%7D%7B%5Csqrt%7B%5Chat%7Bv%7D_t%7D%20%2B%20%5Cepsilon%7D)

The bias correction (![equation](https://latex.codecogs.com/png.latex?\inline%20%5Chat%7Bm%7D) and ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Chat%7Bv%7D)) compensates for the fact that `m` and `v` are initialized to zero, which biases them toward zero in early steps. The learning rate decays linearly from 0.01 to 0 over training. After updating each parameter, its gradient is reset to 0 for the next step.

## 6. Inference: Generating New Names

After training, the model generates text autoregressively:


https://gist.github.com/cmenguy/66a4c2baa4f189ca6537e92f86e2ad12


The generation loop: start with BOS, feed it through the model, get logits, apply temperature scaling, sample the next token, repeat until BOS appears again (end of name) or we hit the max length.

Temperature controls the "sharpness" of the distribution. Before softmax, each logit is divided by the temperature:

![equation](https://latex.codecogs.com/png.latex?\dpi{150}\bg_white%20P%28x_i%29%20%3D%20%5Cfrac%7Be%5E%7Bz_i%20%2F%20T%7D%7D%7B%5Csum_j%20e%5E%7Bz_j%20%2F%20T%7D%7D)

At $T = 1.0$, you get the raw model distribution. At $T < 1.0$ (like 0.5 here), the distribution gets peakier, making the model more likely to pick high-probability tokens. At ![equation](https://latex.codecogs.com/png.latex?\inline%20T%20%5Cto%200), it becomes greedy (always picks the most likely token). At $T > 1.0$, the distribution flattens, producing more "creative" but less coherent output.

`random.choices` does weighted sampling from the distribution. This is the stochastic part that gives you different outputs each time, unlike greedy decoding.

## 7. Mapping to "Attention Is All You Need"

Here's how Karpathy's ~200 lines map to the original transformer paper:


https://gist.github.com/cmenguy/7419372023b6152aa7113eaaff1f3d13


Key differences from the original paper:
- **Decoder-only**: the paper described an encoder-decoder model for translation. GPT dropped the encoder and cross-attention, keeping only the decoder with causal self-attention.
- **RMSNorm instead of LayerNorm**: simpler and cheaper. Modern LLMs almost universally use RMSNorm.
- **ReLU instead of GELU**: GPT-2 used GELU, but ReLU is easier to differentiate by hand and works fine here.
- **No biases**: the original paper used biases everywhere. Modern practice (Llama, PaLM, Gemma) dropped them with no loss in quality.
- **No dropout**: with ~4k parameters and 32k training examples, regularization isn't the bottleneck.

## 8. What "Everything Else Is Just Efficiency" Means

Karpathy's gist opens with: *"This file is the complete algorithm. Everything else is just efficiency."* That line is doing a lot of work. Here's what "everything else" means:

**Tensor operations.** This code does scalar math: each multiply is one `Value * Value`. PyTorch batches these into matrix operations on GPU, getting 1000x+ speedups. The algorithm is identical. The hardware parallelism is the efficiency.

**Batching.** This code processes one document per step. Real training packs hundreds of sequences into a batch, running them through the model in parallel. Same gradients in expectation, but much better GPU utilization.

**Mixed precision.** Training in float16 or bfloat16 instead of float32 halves memory and doubles throughput on modern GPUs. The math is the same, just with fewer bits.

**Parallelism.** Data parallel, tensor parallel, pipeline parallel, FSDP. All of them are strategies for splitting the same computation across multiple GPUs. The fundamental operations stay the same.

**Flash Attention.** A memory-efficient reformulation of the exact same attention computation, using tiling to avoid materializing the full attention matrix. Same result, 2-4x faster.

Every one of these is a performance optimization layered on top of the same core algorithm that Karpathy expressed in 200 lines of Python. If you understand this code, you understand what GPT-4, Claude, and every other transformer-based model is doing at its core. The rest is engineering.

## 9. Running It Yourself

The code trains in about 30 minutes on a laptop CPU (no GPU needed). After 1000 steps, it generates plausible-looking names: things that could be English names but aren't. The loss drops from ~3.3 (random guessing across 27 tokens: ![equation](https://latex.codecogs.com/png.latex?\inline%20-%5Clog%281%2F27%29%20%5Capprox%203.3)) to around 2.0, meaning the model is assigning roughly 7x higher probability to correct tokens than chance.

You can find the full code in the [companion notebook](https://colab.research.google.com/github/cmenguy/cmenguy.github.io/blob/main/notebooks/2026-02-16-gpt-in-pure-python.ipynb) or grab it directly from [Karpathy's gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).

What I took away from this exercise: the gap between "I understand transformers" and "I can implement one from scratch" is worth closing. You don't need to do it for production work. But the next time you're debugging a training run and the loss is doing something weird, or you're trying to understand why a specific hyperparameter matters, having that ground-level intuition makes a real difference. The algorithm fits in 200 lines. The understanding takes a weekend. Worth the trade.


---

*Originally published on [AI Terminal](https://ai-terminal.net/deep-learning/llm/2026/02/16/gpt-in-pure-python/).*

Tags: gpt, autograd, karpathy, attention, transformer
