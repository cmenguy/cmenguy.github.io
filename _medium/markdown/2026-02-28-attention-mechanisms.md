# Is Attention Really All You Need?

---

**Runnable implementations of every attention variant covered in the post, from vanilla scaled dot-product to flash attention.**

[Run in Google Colab](https://colab.research.google.com/github/cmenguy/cmenguy.github.io/blob/main/notebooks/2026-02-28-attention-mechanisms.ipynb) | [View on GitHub](https://github.com/cmenguy/cmenguy.github.io/blob/main/notebooks/2026-02-28-attention-mechanisms.ipynb)

---

A couple weeks ago I wrote a [walkthrough of Karpathy's microgpt.py](https://ai-terminal.net/deep-learning/llm/2026/02/16/gpt-in-pure-python/), tracing every line of a GPT implementation in pure Python. While putting that post together, I kept wanting to linger on the attention section. The scaled dot-product formula showed up, multi-head attention showed up, I explained them, and moved on. But there was so much more to say. The attention mechanism in that 200-line implementation is the same one from the 2017 paper. Production models in 2026 use something very different. Flash attention, grouped-query attention, sliding windows, sparse patterns. The core Q, K, V math stays the same, but the way it gets computed, what gets masked, and how memory gets managed have all evolved significantly.

I'll be upfront: I was never a deep math guy. I didn't come up through a pure ML research track. I'm an engineer who builds systems that use these models, and for a long time I treated attention as a black box with a formula I could recite but didn't fully feel in my bones. That was fine when every model used the same vanilla multi-head attention. It's not fine anymore. When you're picking between GQA and MQA for a serving setup, or trying to understand why your 128k context model OOMs while your colleague's doesn't, you need to know what changed and why. So I went back and traced the evolution properly, starting from the original 2017 formulation and working forward to what's actually running in production today.

This post walks through every major attention variant you'll encounter in modern transformer models. For each one: the math, a visual to build intuition, working code, and when you'd actually use it. Think of it as the attention chapter that the GPT post didn't have room for.

## 1. Scaled Dot-Product Attention: The Foundation

Everything starts here. This is Section 3.2.1 of [Attention Is All You Need](https://arxiv.org/abs/1706.03762), and it's what we saw in the microgpt.py walkthrough.

You have three matrices: queries (Q), keys (K), and values (V). The query asks "what am I looking for?", the key says "what do I contain?", and the value says "here's my actual information." Attention computes a weighted sum of values, where the weights come from how well each query matches each key.

The formula:

![equation](https://latex.codecogs.com/png.latex?\dpi{150}\bg_white%20%5Ctext%7BAttention%7D%28Q%2C%20K%2C%20V%29%20%3D%20%5Ctext%7Bsoftmax%7D%5Cleft%28%5Cfrac%7BQK%5ET%7D%7B%5Csqrt%7Bd_k%7D%7D%5Cright%29V)

Let's break this into steps. Say we have a sequence of 4 tokens, each with embedding dimension 8. Q, K, and V are all shape (4, 8).


https://gist.github.com/cmenguy/4cbf19f6dc4128a2f20dace78d257d5b


The output has the same shape as the input. Each position gets a new representation that's a weighted mix of all the value vectors, with weights determined by query-key similarity.

Here's the implementation in PyTorch:


https://gist.github.com/cmenguy/0424e08d165d6c7ff451e1cad03d2857


And a pure Python version that mirrors what microgpt.py does, without any libraries:


https://gist.github.com/cmenguy/567f7910ae1eab87ad6c4e3b0b34af87


Let's verify they produce the same result:


https://gist.github.com/cmenguy/5ce1a5683b9bf61d5e533ff5035f265f


Same result, give or take floating point precision. The PyTorch version runs ~1000x faster on GPU because it's one batched matrix multiply instead of nested Python loops.

**Why the ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Csqrt%7Bd_k%7D) scaling?** Without it, dot products grow proportionally to ![equation](https://latex.codecogs.com/png.latex?\inline%20d_k). If your keys and queries have entries with zero mean and unit variance, the expected value of each dot product is 0, but the variance is ![equation](https://latex.codecogs.com/png.latex?\inline%20d_k). For ![equation](https://latex.codecogs.com/png.latex?\inline%20d_k%20%3D%2064) (common in practice), that means scores can easily be in the range [-20, 20], which pushes softmax into near-one-hot territory where gradients vanish. Dividing by ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Csqrt%7Bd_k%7D) brings the variance back to 1.

**When is this used?** This exact formulation appears inside every transformer, but never alone. It's always wrapped in multi-head attention (next section). The standalone version is mostly useful for teaching and debugging.

## 2. Multi-Head Attention: Parallel Perspectives

Single-head attention has a limitation: the model gets one "perspective" on how tokens relate. Multi-head attention runs several attention operations in parallel, each with different learned projections, then concatenates the results.

From Section 3.2.2 of the paper:

![equation](https://latex.codecogs.com/png.latex?\dpi{150}\bg_white%20%5Ctext%7BMultiHead%7D%28Q%2C%20K%2C%20V%29%20%3D%20%5Ctext%7BConcat%7D%28%5Ctext%7Bhead%7D_1%2C%20%5Cdots%2C%20%5Ctext%7Bhead%7D_h%29W%5EO)

where ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Ctext%7Bhead%7D_i%20%3D%20%5Ctext%7BAttention%7D%28QW_i%5EQ%2C%20KW_i%5EK%2C%20VW_i%5EV%29)

Visually, here's what happens for 4 heads with ![equation](https://latex.codecogs.com/png.latex?\inline%20d_%7Bmodel%7D%20%3D%2016) and ![equation](https://latex.codecogs.com/png.latex?\inline%20d_k%20%3D%204) per head:


https://gist.github.com/cmenguy/ffbed2bc3c50930c274eec23fbd55842


Each head operates in a smaller subspace (![equation](https://latex.codecogs.com/png.latex?\inline%20d_k%20%3D%20d_%7Bmodel%7D%20%2F%20h)), so the total computation cost is the same as a single full-dimension attention.

In microgpt.py, this was the loop over heads with slice indexing:


https://gist.github.com/cmenguy/91bd816f8a8a6e213850daf0338487f3


In practice, the projections are done as a single matrix multiply and then reshaped. Here's a clean PyTorch implementation:


https://gist.github.com/cmenguy/62db27533375980b0bb2002ac7a6e8fa


The key insight: the `view` and `transpose` operations are free (just pointer math, no data movement). A single `nn.Linear(d_model, d_model)` produces all head projections at once, and the reshape splits them into heads. This is how every production transformer does it.

Let's verify it works:


https://gist.github.com/cmenguy/0dd29f5aaf32fb6b15139defd9af6643


**What do different heads learn?** Research shows that heads tend to specialize. In trained models, some heads attend to the previous token, some to syntactic structures (subject-verb pairs), others to positional patterns. Removing certain heads barely affects performance; removing others is catastrophic. This emergent specialization is why multi-head works better than single-head despite using the same total compute.

**When is this used?** Every transformer model since 2017. This is the standard. The variants that follow are all modifications of this core structure.

## 3. Causal (Masked) Attention: No Peeking

In language models, a token at position $i$ should only attend to tokens at positions ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Cleq%20i). You can't let the model see future tokens during training, or it'll just copy the answer instead of learning to predict.

The fix is a causal mask: set the upper-right triangle of the attention score matrix to ![equation](https://latex.codecogs.com/png.latex?\inline%20-%5Cinfty) before softmax. Since ![equation](https://latex.codecogs.com/png.latex?\inline%20e%5E%7B-%5Cinfty%7D%20%3D%200), those positions get zero weight.


https://gist.github.com/cmenguy/c6a39bc2fd61c5470d813fc5f0c53576


The math is the same as before, with a mask added:

![equation](https://latex.codecogs.com/png.latex?\dpi{150}\bg_white%20%5Ctext%7BCausalAttention%7D%28Q%2C%20K%2C%20V%29%20%3D%20%5Ctext%7Bsoftmax%7D%5Cleft%28%5Cfrac%7BQK%5ET%7D%7B%5Csqrt%7Bd_k%7D%7D%20%2B%20M%5Cright%29V)

where ![equation](https://latex.codecogs.com/png.latex?\inline%20M_%7Bij%7D%20%3D%200) if ![equation](https://latex.codecogs.com/png.latex?\inline%20i%20%5Cgeq%20j) (allowed), and ![equation](https://latex.codecogs.com/png.latex?\inline%20M_%7Bij%7D%20%3D%20-%5Cinfty) if $i < j$ (blocked).

The implementation is just adding the mask to our existing code:


https://gist.github.com/cmenguy/bd8f56d12dc64e3facee620097f3957a


Let's see the mask in action:


https://gist.github.com/cmenguy/02a293f875962eb9fda695e5192ed206



https://gist.github.com/cmenguy/6dd90691e6673707683c330b8a23c2ec


Row 0 puts all weight on position 0 (only option). Row 1 heavily favors position 1 (0.941) over position 0 (0.059). By row 3, the weights spread across all four positions, with position 3 getting the largest share. The upper triangle is all zeros.

In microgpt.py, causal masking was implicit. Because it processed one token at a time and only stored past keys/values in the KV cache, the model could never attend to future positions. The explicit mask approach is what you use during training, when the full sequence is processed in parallel.

**When is this used?** Every autoregressive language model (GPT, Llama, Claude, Mistral, Gemini). If it generates text left-to-right, it uses causal masking.

## 4. Cross-Attention: Connecting Two Sequences

The original transformer was built for translation. You have an input sentence in French and want to produce output in English. The encoder processes the French input; the decoder generates the English output. Cross-attention is how the decoder "looks at" the encoder's representation.

The math is identical to self-attention, except Q comes from the decoder and K, V come from the encoder:

![equation](https://latex.codecogs.com/png.latex?\dpi{150}\bg_white%20%5Ctext%7BCrossAttention%7D%20%3D%20%5Ctext%7Bsoftmax%7D%5Cleft%28%5Cfrac%7BQ_%7Bdec%7D%20K_%7Benc%7D%5ET%7D%7B%5Csqrt%7Bd_k%7D%7D%5Cright%29%20V_%7Benc%7D)


https://gist.github.com/cmenguy/1880589204b930edc761de5099055497


Here's the implementation. It's the same `MultiHeadAttention` class, but called with different inputs for Q vs. K/V:


https://gist.github.com/cmenguy/fd19943fd8d1f51d696c76a02f8df4c4



https://gist.github.com/cmenguy/82012dcc6b01acf64542cb9574707610


Notice there's no causal mask here. Each decoder position should be able to attend to any encoder position. The word "monde" might be most relevant for generating "world" regardless of position.

**When is this used?** Encoder-decoder models: the original Transformer, T5, BART, Whisper (speech-to-text), and vision-language models where the "encoder" is an image encoder and the "decoder" generates text. GPT-style decoder-only models don't use cross-attention at all. They threw out the encoder and replaced cross-attention with just longer context windows.

## 5. Multi-Query Attention: One KV Head for All

Here's where we move from the 2017 paper to modern efficiency optimizations. Multi-Query Attention (MQA), introduced by [Shazeer (2019)](https://arxiv.org/abs/1911.02150), makes a simple but effective change: instead of giving each head its own K and V projections, share a single K and single V across all heads. Each head still gets its own Q.


https://gist.github.com/cmenguy/92c610842f984f3726a6f3566be8fb95


Why does this matter? The KV cache. During inference, you store one K and one V vector per token per layer. With 32 layers and 32 heads at ![equation](https://latex.codecogs.com/png.latex?\inline%20d_k%20%3D%20128), that's $32 \times 32 \times 128 \times 2 = 262{,}144$ floats per token. For a 100k context, that's ~100GB in float32. MQA cuts the KV cache by the number of heads (32x in this case), bringing it down to ~3GB.


https://gist.github.com/cmenguy/7820331b78d5f4323a88d7a538c20e4c


The K and V tensors have shape `(B, 1, T, d_k)` and PyTorch broadcasts them across all `n_heads` during the matrix multiplications. No data duplication, just pointer tricks.


https://gist.github.com/cmenguy/2757231b0efd269798da3cda4965c604


The trade-off: some quality loss, because all heads now share the same key-value representation. In practice the degradation is small (0.3-0.5% on benchmarks), but it's measurable.

**When is this used?** PaLM (2022), Falcon, StarCoder, and several inference-optimized models. It's most valuable when you're serving many concurrent requests and KV cache memory is the bottleneck. For training, the compute savings are minimal since there's no KV cache.

## 6. Grouped-Query Attention: The Middle Ground

GQA, introduced by [Ainslie et al. (2023)](https://arxiv.org/abs/2305.13245), splits the difference between standard multi-head attention and MQA. Instead of one shared KV head or one per query head, you use a small number of KV head groups.


https://gist.github.com/cmenguy/a72e35342a326f3b78662b23a62d021b


Four query heads share one KV head. This gives you most of MQA's memory savings while keeping more representational capacity in the keys and values.


https://gist.github.com/cmenguy/8db8c01489ba03d4eaddd49a5efc2cec



https://gist.github.com/cmenguy/3ed6971cf9e6b6cb991fceec50948c8a


The `repeat_interleave` call copies each KV head to match the query heads in its group. Alternatively, you can use `expand()` to avoid the copy and just broadcast, which is what optimized implementations do.

Here's the comparison table:


https://gist.github.com/cmenguy/26743a90d58b557b61c2de6b03717867


**When is this used?** Llama 2 70B, Llama 3, Mistral 7B, Gemma, and most new models released after mid-2023. GQA has become the default choice because it gets ~95% of MQA's memory savings with nearly zero quality loss. The Llama 2 paper showed you can even take an MHA-trained model and "uptrain" it to GQA with a small fraction of the original training budget.

## 7. Sparse Attention: Breaking the Quadratic Wall

Standard attention computes scores between every pair of tokens. For a sequence of length $n$, that's ![equation](https://latex.codecogs.com/png.latex?\inline%20O%28n%5E2%29) operations and ![equation](https://latex.codecogs.com/png.latex?\inline%20O%28n%5E2%29) memory. Double the context length, quadruple the cost. At 100k tokens, the attention matrix alone is $100{,}000 \times 100{,}000 = 10$ billion entries per head per layer.

Sparse attention patterns solve this by only computing a subset of the attention scores. The idea, introduced by [Child et al. (2019)](https://arxiv.org/abs/1904.10509) in "Generating Long Sequences with Sparse Transformers", is that most attention weights are near zero anyway. If you can identify which ones matter, you can skip the rest.

The most common sparse patterns:


https://gist.github.com/cmenguy/df4872f46acaa9b0065ff9373828e858


In strided sparse, each token attends to every $s$-th token (stride) plus a local window. In fixed sparse, the sequence is split into blocks and tokens attend within their block plus to a set of "summary" positions.

Here's a practical implementation of block-sparse attention:


https://gist.github.com/cmenguy/b7d62fb64a861da791dd1179fa54336f



https://gist.github.com/cmenguy/55909197e3899176b3ff87c2601a1122


The complexity drops from ![equation](https://latex.codecogs.com/png.latex?\inline%20O%28n%5E2%29) to ![equation](https://latex.codecogs.com/png.latex?\inline%20O%28n%20%5Ccdot%20b%29) where $b$ is the effective neighborhood size (block size times number of neighbor blocks). For long sequences, this is the difference between "fits in memory" and "doesn't."

**When is this used?** BigBird (Google), Longformer (AI2), and various long-context models. In practice, pure sparse attention is less common in 2025-era LLMs. Most modern models use sliding window attention (next section) or combine sparse patterns with full attention in alternating layers.

## 8. Sliding Window Attention: Local Context, Global Reach

Sliding window attention is a specific, clean form of sparse attention. Each token attends to only a fixed window of $w$ tokens around it. Introduced in Longformer and popularized by [Mistral 7B](https://arxiv.org/abs/2310.06825).


https://gist.github.com/cmenguy/62eaa90cb5c047992b62b8592cd9ecca


The math is the same, just a different mask. For a window of size $w$:

![equation](https://latex.codecogs.com/png.latex?\dpi{150}\bg_white%20M_%7Bij%7D%20%3D%20%5Cbegin%7Bcases%7D%200%20%26%20%5Ctext%7Bif%20%7D%20i%20-%20w%20%3C%20j%20%5Cleq%20i%20%5C%5C%20-%5Cinfty%20%26%20%5Ctext%7Botherwise%7D%20%5Cend%7Bcases%7D)


https://gist.github.com/cmenguy/a312a70d334f5a0f4e01c97e51171899



https://gist.github.com/cmenguy/157e8f0fe43c542a0ae78951e17d3e10



https://gist.github.com/cmenguy/2c8b8f4abd912ce4092ccf502e67a747


"But wait, if token 7 can only see tokens 5-7 directly, how does it get information from token 0?" Through stacking layers. In a model with $L$ layers, information can propagate ![equation](https://latex.codecogs.com/png.latex?\inline%20L%20%5Ctimes%20w) positions. Mistral 7B uses ![equation](https://latex.codecogs.com/png.latex?\inline%20w%20%3D%204%7B%2C%7D096) with 32 layers, giving an effective receptive field of $32 \times 4{,}096 = 131{,}072$ tokens. That's how you get long context without quadratic scaling.

The memory savings are real. Instead of a KV cache that grows linearly with sequence length, you only need to store the last $w$ key-value pairs per layer. Once a token slides out of the window, its KV entry gets recycled. This is done with a rolling buffer:


https://gist.github.com/cmenguy/a0b0b8ef27c9a7815731c72c25aa797a


Fixed memory, regardless of how long the conversation gets.

**When is this used?** Mistral 7B was the model that popularized this approach. Mistral uses sliding window in some layers and full attention in others (alternating pattern). Mixtral follows the same strategy. Many newer models use a similar hybrid: full attention for some layers to maintain global context, sliding window for others to save memory.

## 9. Flash Attention: Same Math, Different Memory

Flash Attention, introduced by [Dao et al. (2022)](https://arxiv.org/abs/2205.14135), is the most impactful attention optimization of the past few years. The key insight: standard attention is bottlenecked not by compute, but by memory bandwidth.

Here's the problem. Standard attention materializes the full ![equation](https://latex.codecogs.com/png.latex?\inline%20n%20%5Ctimes%20n) attention matrix in GPU high-bandwidth memory (HBM):


https://gist.github.com/cmenguy/ba5f80d771812a6fa8c3d2b7fc658aa7


GPU SRAM (on-chip cache) is ~100x faster than HBM, but much smaller (20MB vs 40-80GB on an A100). Flash Attention restructures the computation to work in tiles that fit in SRAM, never materializing the full attention matrix:


https://gist.github.com/cmenguy/26a14b9aa43432c224f1ba9b4c4451c8


The math is identical to standard attention. Same inputs, same outputs, same gradients. The trick is an online softmax algorithm that processes the attention matrix in tiles without needing the full matrix at once.

The online softmax works like this. For standard softmax, you need the max over all elements (for numerical stability) before computing anything. The tiled version keeps a running max and a running sum, correcting for new tiles as they arrive:


https://gist.github.com/cmenguy/5e51517ebe89bb1cece23a674843caa3



https://gist.github.com/cmenguy/6bc5e9025ee32fe85372bc5a36ae4379



https://gist.github.com/cmenguy/a39c991bc623605188a51b6ec27d4ce8


Exact same result, computed without ever storing all scores simultaneously.

In practice, you don't implement Flash Attention yourself. It's a CUDA kernel that PyTorch exposes via `F.scaled_dot_product_attention`:


https://gist.github.com/cmenguy/9aaeeb67e7798656e6bf8408cb6fa7e9


The speedup numbers:


https://gist.github.com/cmenguy/6837a8691e59e772b93d5fea29d5ca2f


The longer the sequence, the bigger the win, because HBM bandwidth becomes more of a bottleneck relative to compute. Flash Attention 2 improved on the original by better partitioning work across GPU warps and reducing non-matmul operations.

**When is this used?** Basically everywhere in 2025-2026. PyTorch's `scaled_dot_product_attention` uses Flash Attention by default when available. Every major framework (vLLM, TGI, llama.cpp) uses it. If you're running inference or training on a modern GPU with sequences longer than a few hundred tokens, Flash Attention is what's running under the hood.

## 10. Putting It All Together: What Modern Models Actually Use

Let's map these attention variants to the models you're actually working with:


https://gist.github.com/cmenguy/6acc56c96bdcc755a954cc64d56ea8ac


A few patterns to notice:

**GQA won.** After Llama 2 showed that GQA matches MHA quality with a fraction of the KV cache, everyone adopted it. The 8 KV heads number appears repeatedly because it's a sweet spot: enough capacity for quality, small enough for efficient serving.

**Flash Attention is everywhere.** It went from research paper to industry standard in about a year. Any model released after 2023 assumes Flash Attention is available.

**Window attention is a practical choice.** Mistral proved that you don't need full quadratic attention for every layer. Alternating full and windowed attention across layers gives you global reach (through stacking) with local efficiency. Gemma 2 followed the same idea.

**Context lengths exploded.** From GPT-2's 1,024 tokens to Llama 3's 128,000. This happened because of three compounding factors: Flash Attention making long sequences computationally feasible, GQA/MQA reducing KV cache memory, and techniques like RoPE (Rotary Position Embeddings) that generalize to unseen positions.

## 11. The Decision Tree

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

## 12. Beyond Attention: What's Coming

The attention mechanism has been the core of the transformer since 2017, and every model released since then uses some variant of it. But the design space keeps evolving.

Multi-head Latent Attention (MLA), used in DeepSeek V2, compresses the KV cache into a learned low-rank latent space, getting even smaller than GQA. Linear attention variants like RWKV and Mamba replace the softmax attention entirely with recurrent-style computations that scale linearly with sequence length. These aren't mainstream yet for LLMs, but they're competitive on several benchmarks and have O(1) memory per token during inference.

The 2017 paper asked if attention is all you need. Nine years later, the answer is nuanced: attention is definitely what you need, but which attention and how you compute it matters more than anyone expected. The Q, K, V math from the original paper is still there in every model. Everything built on top of it (multi-head, causal masking, GQA, Flash Attention, sliding windows) is the engineering that makes it work at scale. If you understand each layer of that stack, you can make informed decisions about which pieces to use and when. That's the whole point.


---

*Originally published on [AI Terminal](https://ai-terminal.net/deep-learning/llm/2026/02/28/attention-mechanisms/).*

Tags: gqa, mqa, attention, transformer, from-scratch
