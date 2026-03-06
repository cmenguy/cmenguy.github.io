# LoRA From the Ground Up: The Math, the Matrices, and the Merge

---

**Hands-on LoRA implementation from scratch: building, training, merging, and inspecting low-rank adapters with PyTorch.**

[Run in Google Colab](https://colab.research.google.com/github/cmenguy/cmenguy.github.io/blob/main/notebooks/2025-12-28-lora-deep-dive.ipynb) | [View on GitHub](https://github.com/cmenguy/cmenguy.github.io/blob/main/notebooks/2025-12-28-lora-deep-dive.ipynb)

---

In my [last post](https://ai-terminal.net/llm/fine-tuning/2025/12/22/fine-tuning-llms-sft-dpo-rlhf/), I walked through SFT, DPO, and RLHF for fine-tuning LLMs. Throughout that entire post, LoRA kept showing up in every code example, every training config, every `LoraConfig(r=16, lora_alpha=32)` call. I used it the way most of us do: copy the config from a tutorial, set `r=16` because that's what everyone uses, set `lora_alpha` to double the rank because... reasons, and move on. The model trains, the loss goes down, the outputs improve. Ship it.

But a few days ago I got into a discussion with a colleague about fine-tuning efficiency: how much memory we were actually saving with LoRA, whether we could push the rank lower without hurting quality, whether it even mattered which layers we targeted. I had opinions on all of this, but when I tried to back them up with anything beyond "it worked last time," I realized I was hand-waving. I knew *what* LoRA did at a high level (low-rank matrices, fewer parameters, memory efficient), but I couldn't actually explain *why* those specific numbers mattered. What does rank even mean in this context? Why does `lora_alpha` scale the way it does? What's actually happening to the weight matrices during training? I'd been treating LoRA like a black box with good defaults, and that bothered me.

So I blocked out a weekend, pulled up the [original paper](https://arxiv.org/abs/2106.09685), and went through the math line by line. What follows is what I wish someone had explained to me before I started using LoRA in production.

## 1. The Problem LoRA Solves

Let's start with why LoRA exists. A model like Llama 3.1 8B has roughly 8 billion parameters. Full fine-tuning means updating all of them: every weight in every layer gets a gradient, an optimizer state, and a momentum term. For Adam, that's 3x the model size in memory just for the optimizer. On a Llama 8B in float32, that's:

![equation](https://latex.codecogs.com/png.latex?\dpi{150}\bg_white%20%5Ctext%7BMemory%7D_%7B%5Ctext%7Bfull%7D%7D%20%3D%208%5Ctext%7BB%7D%20%5Ctimes%204%5Ctext%7B%20bytes%7D%20%5Ctimes%203%20%3D%2096%5Ctext%7B%20GB%20%28optimizer%20alone%29%7D)

Add the model weights, gradients, and activations, and you're looking at needing multiple A100 80GB GPUs just for fine-tuning. For most teams, that's impractical.

LoRA's insight: when you fine-tune a large model on a specific task, the weight updates don't use the full dimensionality of the weight matrices. The *change* in weights during fine-tuning is low-rank. It lies in a much smaller subspace than the original weights. So instead of updating a giant matrix, you can decompose the update into two small matrices and only train those.

## 2. The Core Idea: Low-Rank Decomposition

Here's the key equation. For a pretrained weight matrix ![equation](https://latex.codecogs.com/png.latex?\inline%20W_0%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bd%20%5Ctimes%20k%7D), LoRA constrains the update ![equation](https://latex.codecogs.com/png.latex?\inline%20%5CDelta%20W) to be a low-rank decomposition:

![equation](https://latex.codecogs.com/png.latex?\dpi{150}\bg_white%20W%20%3D%20W_0%20%2B%20%5CDelta%20W%20%3D%20W_0%20%2B%20BA)

Where ![equation](https://latex.codecogs.com/png.latex?\inline%20B%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bd%20%5Ctimes%20r%7D) and ![equation](https://latex.codecogs.com/png.latex?\inline%20A%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Br%20%5Ctimes%20k%7D), with rank ![equation](https://latex.codecogs.com/png.latex?\inline%20r%20%5Cll%20%5Cmin%28d%2C%20k%29).

That's it. That's the whole trick. Instead of learning a ![equation](https://latex.codecogs.com/png.latex?\inline%20d%20%5Ctimes%20k) matrix of updates (potentially millions of parameters), you learn two smaller matrices whose product has the same shape but far fewer total parameters.

Let me make this concrete with a picture. Say you have a weight matrix in a transformer attention layer with $d = 4096$ and $k = 4096$:


https://gist.github.com/cmenguy/c27c1db10428480df5d6a325329c4878


With $r = 8$, you're training 65,536 parameters instead of 16.7 million — a **256x reduction** for this single layer. Across the entire model, LoRA typically trains 0.1-1% of the total parameters.

## 3. The Forward Pass: How It Actually Computes

During a forward pass, the original weight and the LoRA update combine like this. For an input $x$:

![equation](https://latex.codecogs.com/png.latex?\dpi{150}\bg_white%20h%20%3D%20W_0%20x%20%2B%20%5CDelta%20W%20x%20%3D%20W_0%20x%20%2B%20BAx)

Here's what that looks like step by step:


https://gist.github.com/cmenguy/3475a427558abec8902bef0307e2e148


The pretrained weights ![equation](https://latex.codecogs.com/png.latex?\inline%20W_0) stay **completely frozen**: no gradients, no optimizer states, no memory overhead. Only $B$ and $A$ receive gradients. This is why LoRA is so memory-efficient: you only store optimizer states for the tiny adapter matrices, not the full model.

Let's implement this from scratch in PyTorch so you can see exactly what's happening:


https://gist.github.com/cmenguy/3d9417d98a662f6a687c22ae32a61df8


A few things to notice here. The original layer is frozen (`requires_grad = False`). And there's a `scaling` factor that we'll come back to shortly. Now the adapter matrices:


https://gist.github.com/cmenguy/f50194f9ce5751aaffc700d1139e82c8


This initialization is critical. $B$ starts at zero, which means ![equation](https://latex.codecogs.com/png.latex?\inline%20%5CDelta%20W%20%3D%20BA%20%3D%200) at the beginning of training. The model starts producing exactly the same outputs as the pretrained model. Training then gradually learns the update. $A$ uses Kaiming uniform initialization to break symmetry.

The forward pass puts it all together:


https://gist.github.com/cmenguy/1f9a000e958e795b1685c95cf15a4c58


Two separate matrix multiplications through the bottleneck: ![equation](https://latex.codecogs.com/png.latex?\inline%20x%20%5Ccdot%20A%5ET) compresses to rank $r$, then ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Ccdot%20B%5ET) projects back up, plus the scaling factor. Let's see the parameter savings in action:


https://gist.github.com/cmenguy/9b19a3b762baacd1e8a925046b37c571



https://gist.github.com/cmenguy/91745082cbba6d1186451874d8544758


## 4. What Rank Actually Means

The rank $r$ is LoRA's most important hyperparameter, and it's worth building intuition about what it controls.

In linear algebra, the rank of a matrix is the number of linearly independent rows (or equivalently, columns). A rank-$r$ matrix can be expressed as the sum of $r$ rank-1 outer products. Think of it as the number of "independent directions" the matrix can push information through.

When we constrain ![equation](https://latex.codecogs.com/png.latex?\inline%20%5CDelta%20W%20%3D%20BA) with ![equation](https://latex.codecogs.com/png.latex?\inline%20B%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bd%20%5Ctimes%20r%7D) and ![equation](https://latex.codecogs.com/png.latex?\inline%20A%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Br%20%5Ctimes%20k%7D), the product $BA$ has rank at most $r$. This means the weight update can only modify the model's behavior along $r$ independent directions in the weight space.

The original LoRA paper found something surprising: even $r = 1$ or $r = 2$ works reasonably well for many tasks. The weight updates during fine-tuning really are low-rank. Here's an intuition for why: when you fine-tune on a specific task (like marketing copy), you're not rewiring the model's entire understanding of language. You're making a targeted adjustment: "write in this style" or "prefer these patterns." That adjustment occupies a small subspace of what the model's weights can represent.

Here's a practical way to see this. Let's create a weight update, compute its singular values, and see how the energy concentrates:


https://gist.github.com/cmenguy/a3c24fa21660b940b6d615ff732b967c



https://gist.github.com/cmenguy/278c20dfae46e7985d6d1e381ba8ed89


A random matrix spreads its energy uniformly across all singular values, which is why even $r = 64$ only captures ~2%. But real fine-tuning updates aren't random. They concentrate on a few directions that matter for the task. In practice, $r = 8$ or $r = 16$ captures the meaningful signal while ignoring noise.

#### 4.1 Choosing Rank in Practice


https://gist.github.com/cmenguy/fbeef349cd77b522c3f445ee77ff9e1a


The sweet spot for most tasks is ![equation](https://latex.codecogs.com/png.latex?\inline%20r%20%5Cin%20%5B8%2C%2016%5D). Going higher adds parameters without proportional improvement. Going lower risks underfitting complex tasks.

## 5. The Scaling Factor: Why lora_alpha Exists

If you've ever stared at `lora_alpha=32` in a config and wondered what it does, here's the answer. The LoRA forward pass applies a scaling factor:

![equation](https://latex.codecogs.com/png.latex?\dpi{150}\bg_white%20h%20%3D%20W_0%20x%20%2B%20%5Cfrac%7B%5Calpha%7D%7Br%7D%20%5Ccdot%20BAx)

Where ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Calpha) is `lora_alpha` and $r$ is the rank. This ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Cfrac%7B%5Calpha%7D%7Br%7D) scaling serves a critical purpose: it **decouples the learning rate from the rank**.

Without this scaling, changing the rank would change the magnitude of the LoRA update. If you double $r$, you'd roughly double the norm of $BA$ (more parameters contributing to the output), and you'd need to halve the learning rate to compensate. The ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Cfrac%7B%5Calpha%7D%7Br%7D) factor normalizes this away.

Here's the practical implication. When `lora_alpha = 2 * r` (the common convention), the scaling factor is ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Cfrac%7B2r%7D%7Br%7D%20%3D%202). The LoRA update gets amplified by 2x. This means:


https://gist.github.com/cmenguy/cd0edecc0b9aed432bc26bbad0de6b2a


You can think of `lora_alpha` as a "volume knob" for the LoRA update. Higher alpha amplifies the adapter's effect. The convention of `alpha = 2 * r` works well in practice, but you can tune it, especially if you notice training instability (lower alpha) or the model not learning fast enough (higher alpha).

Let's see this in action:


https://gist.github.com/cmenguy/d2224c3f6717e69633d164302a1e9624



https://gist.github.com/cmenguy/9bd537e9fd0b92af57474398b3578a3a


Linear relationship: double the alpha, double the output magnitude. The learning rate and scaling factor interact, which is why the convention of fixing `alpha = 2r` and tuning only the learning rate is the pragmatic approach.

## 6. Which Layers Get LoRA?

In a transformer, LoRA is typically applied to the attention projection matrices. Looking at a standard multi-head attention block:


https://gist.github.com/cmenguy/61a0b29d8fd667ed0eb0523c80c18948


The original paper applied LoRA only to ![equation](https://latex.codecogs.com/png.latex?\inline%20W_q) and ![equation](https://latex.codecogs.com/png.latex?\inline%20W_v), but modern practice targets all four attention projections. Some people also include the MLP layers (`gate_proj`, `up_proj`, `down_proj`), though the marginal benefit varies.

Here's the config you'll see in most production setups:


https://gist.github.com/cmenguy/8349a9815533ee8e47fee7332dad633a


And if you want to be more aggressive:


https://gist.github.com/cmenguy/a8890c2b0457a085e0b4786b6fc66836


Let's count the parameter difference across a full model:


https://gist.github.com/cmenguy/ab35ef07d4ec0c29dffec1ce0c27e161



https://gist.github.com/cmenguy/ee21a687d7e78c9d2428cbc820ac62f7


Even the aggressive "all projections" approach trains less than 1% of the model. That's LoRA's superpower.

## 7. A Complete Training Example

Let's put all the pieces together with a real training example. We'll fine-tune a small model so you can actually run this, and inspect the LoRA matrices at each stage.

First, let's create a minimal dataset and load a model with LoRA:


https://gist.github.com/cmenguy/fd7d42fedb11fac02484e90bf21deb99



https://gist.github.com/cmenguy/7beda87422356e47358edc00b1dd2418


Only 0.34% of parameters are trainable. Let's inspect what the LoRA matrices look like before training:


https://gist.github.com/cmenguy/6bc281ed38a89b630de0bbb90b407d29



https://gist.github.com/cmenguy/fb8d4fd42ebed490e58eaadd95eeeba8


Exactly as expected. $A$ is initialized with random values, $B$ is all zeros, so ![equation](https://latex.codecogs.com/png.latex?\inline%20%5CDelta%20W%20%3D%20BA%20%3D%200). The model starts as if no adapter exists.

Now let's train it on a few examples and see how the matrices change:


https://gist.github.com/cmenguy/d25a1b4797fae8387a6bf8299b6531b9


After training, let's check the matrices again:


https://gist.github.com/cmenguy/41ef74d1e1b8760b86a1af41dc3df0e4


$B$ is no longer zero. Training has learned a low-rank update. The model's behavior has shifted, but only along 8 independent directions in the weight space.

## 8. Merging: Collapsing the Adapter Into the Model

This is where things get practically interesting. You've trained your LoRA adapter. Now what? You have two options: keep the adapter separate, or merge it into the base model. The choice has real implications for serving.

#### 8.1 What Merging Means

Merging is just matrix addition. You take the pretrained weight ![equation](https://latex.codecogs.com/png.latex?\inline%20W_0) and permanently add the LoRA update:

![equation](https://latex.codecogs.com/png.latex?\dpi{150}\bg_white%20W_%7B%5Ctext%7Bmerged%7D%7D%20%3D%20W_0%20%2B%20%5Cfrac%7B%5Calpha%7D%7Br%7D%20%5Ccdot%20BA)

After merging, the model is a regular model again: no adapter, no separate matrices, no extra computation at inference time.


https://gist.github.com/cmenguy/9e82b5ba57d7a839ac5bbc7252df0b3e


Here's how you do it in code:


https://gist.github.com/cmenguy/35dece0358cb3c42d03ed86b81cc260e


Let's verify the merge is mathematically correct:


https://gist.github.com/cmenguy/57ecd181a892692ef569bd73eb13583d



https://gist.github.com/cmenguy/d86d83662892149c2472049255c43b61


Just matrix addition with scaling. Nothing mysterious.

#### 8.2 What Happens If You Don't Merge

If you skip the merge, the LoRA adapter stays separate from the base model. This isn't just an academic distinction; it affects both performance and flexibility.

**Inference overhead.** Without merging, every forward pass computes two paths: the base model path and the LoRA path. For a single request, the overhead is small. But at scale, those extra matrix multiplications add up:


https://gist.github.com/cmenguy/0d315c41f7fd134e68b78978e446012c


The exact overhead depends on hardware, but expect 5-15% extra latency on the forward pass. Not catastrophic, but not free.

**Multi-adapter serving.** Here's the flip side: not merging is actually a *feature* when you need to serve multiple adapters. If you have one base model and 50 brand-specific LoRA adapters (like the marketing scenario from the previous post), you can:


https://gist.github.com/cmenguy/a16979ee34a349ee31ef71eef7b20466


Each adapter is a few megabytes. The base model is tens of gigabytes. Without merging, you store one base model + N tiny adapters instead of N full model copies. That's the difference between needing 1 GPU and needing 50.


https://gist.github.com/cmenguy/1c979eaeb18943125168ee7d4c74faa2


#### 8.3 When to Merge vs. Keep Separate


https://gist.github.com/cmenguy/1cdfb4d36896e318526a2cefddacc6f1


#### 8.4 The Merge in Detail

Let's trace exactly what `merge_and_unload` does under the hood. It's simple but worth understanding:


https://gist.github.com/cmenguy/871259e16b936f5f2e3227f0d3d98b22


The merged weight is a regular matrix. No special structure, no adapter overhead. But you lose the ability to "un-merge"; the adapter's contribution is baked into the weights permanently.

## 9. Practical Tips From Production

A few things I've learned the hard way that the paper doesn't tell you:

**Start with r=8 and alpha=16.** This is a good default for 7B-13B parameter models on most tasks. Only increase rank if you see clear signs of underfitting (training loss not decreasing fast enough despite reasonable learning rate).

**Learning rate matters more than rank.** The learning rate for LoRA should typically be 5-10x higher than what you'd use for full fine-tuning. This is because you're only updating a small subset of parameters, so they need to move more per step to have the same overall effect. Start with `2e-4` and adjust from there.

**Dropout is your friend for small datasets.** `lora_dropout=0.05` is the default, but if you're training on fewer than 1000 examples, bump it to `0.1`. The low-rank bottleneck is already a form of regularization, but it's not always enough.

**Save adapters, not merged models**, at least during development. A LoRA adapter for a 7B model is ~10-50 MB. A merged model is ~14 GB. When you're running dozens of experiments, that storage difference matters.

**Double-check your target modules.** Different model families have different linear layer names. Llama uses `q_proj`, `k_proj`, `v_proj`, `o_proj`. Other models might use `query`, `key`, `value`, or `qkv_proj`. Check with:


https://gist.github.com/cmenguy/2a3272058df9748308ef359786db7306


## 10. Wrapping Up

LoRA's elegance is in how simple it actually is once you see the math. Freeze the pretrained weights, learn a low-rank update decomposed into two small matrices, and add it to the forward pass with a scaling factor. That's the whole algorithm. The rest is engineering: choosing which layers to target, setting the rank and scaling, deciding whether to merge for serving or keep adapters separate for flexibility.

The next time you write `LoraConfig(r=16, lora_alpha=32)`, you'll know exactly what those numbers mean and why they matter. And when someone on your team asks "can we make r bigger?" you'll be able to explain *what* it actually changes in the weight space, not just *whether* to do it.


---

*Originally published on [AI Terminal](https://ai-terminal.net/llm/fine-tuning/2025/12/28/lora-deep-dive/).*

Tags: lora, peft, merging, low-rank, fine-tuning
