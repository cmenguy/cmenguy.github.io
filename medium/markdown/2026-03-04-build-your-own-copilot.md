# Build Your Own Copilot in Pure Python

---

**A complete, runnable AI code completion tool: FIM inference, LSP-style server, and LoRA fine-tuning on a custom codebase.**

[Run in Google Colab](https://colab.research.google.com/github/cmenguy/cmenguy.github.io/blob/main/notebooks/2026-03-04-build-your-own-copilot.ipynb) | [View on GitHub](https://github.com/cmenguy/cmenguy.github.io/blob/main/notebooks/2026-03-04-build-your-own-copilot.ipynb)

---

In the [skills post](https://ai-terminal.net/llm/ai-engineering/agents/2026/02/05/skills-deep-dive/) I mentioned we'd been building agent skills at work, mostly for marketing ML workflows. One of those workflows is NL2Code: a user describes what they want in plain English ("build me a lookalike audience from this seed list using cosine similarity"), and the agent writes the Python code. It works surprisingly well for self-contained scripts. But it fell apart the moment we needed the generated code to fit into our existing codebase. The agent would write perfectly valid Python that imported libraries we don't use, called APIs that don't exist in our stack, and followed patterns that look nothing like the code we actually ship.

That got me thinking about how tools like Copilot actually work. Not the product, but the machinery underneath: how do you take a general-purpose code model and make it write code that feels like it belongs in *your* repo? I spent a few weekends digging into the full pipeline, from how code completion models are structured, to how fill-in-the-middle works, to how you fine-tune on a specific codebase. This post is what came out of that: a working AI coding tool, built from scratch in Python, that you can point at your own code.

This is **part 1 of a 3-part series** on AI-assisted code generation. This post covers the inline completion side: FIM, code models, fine-tuning, and building a working Copilot clone. Part 2 will tackle NL2Code: instruction-following code generation, where you describe what you want in English and the model writes the full implementation. Part 3 will go after the other end of the lifecycle: an AI-assisted bug detection and fix suggestion system that reads your code, spots problems, and proposes patches.

## 1. How Code Completion Actually Works

Before building anything, let's understand what's happening when Copilot suggests code in your editor.

A code completion model is a language model trained on code. Same transformer architecture as GPT, same next-token prediction objective, just trained on GitHub repos instead of web text. Models like CodeLlama, StarCoder, and DeepSeek-Coder are all in this family. They predict: given the code written so far, what token comes next?

But code completion has a problem that regular text doesn't. When you're typing in an editor, the cursor isn't always at the end of the file. You might be in the middle of a function, with code above *and* below. A model trained purely on left-to-right next-token prediction can only see what's above the cursor. It has no idea what comes after.

This is where **Fill-in-the-Middle (FIM)** comes in.

#### 1.1 Fill-in-the-Middle: The Key Trick

FIM was introduced in the [Bavarian et al. 2022](https://arxiv.org/abs/2207.14255) paper from OpenAI. The idea is simple: during training, you take a code file, split it into three parts (prefix, middle, suffix), and train the model to predict the middle given the prefix and suffix.

Here's how a normal training example looks:


https://gist.github.com/cmenguy/a13b5a06533d6eb1486fd80d1a8533fe


And here's the same example reformatted for FIM training:


https://gist.github.com/cmenguy/cd7aa551c0df0b297926a8358a257aca


The model learns to take the prefix (everything before the cursor) and the suffix (everything after), and generate the middle (what should go where the cursor is). Three special tokens mark the boundaries: `<fim_prefix>`, `<fim_suffix>`, and `<fim_middle>`.

There are two formats for arranging these pieces:

**PSM (Prefix-Suffix-Middle):** `<fim_prefix>PREFIX<fim_suffix>SUFFIX<fim_middle>MIDDLE`

**SPM (Suffix-Prefix-Middle):** `<fim_suffix>SUFFIX<fim_prefix>PREFIX<fim_middle>MIDDLE`

Most production models (StarCoder, CodeLlama, DeepSeek-Coder) use PSM. The key insight from the paper: FIM can be added to training with almost no cost. You take a fraction of your training examples (typically 50-90%), reformat them as FIM, and the model learns both regular left-to-right completion and fill-in-the-middle, with no degradation on either task.

#### 1.2 What Makes Code Models Different

Code LLMs aren't just GPT trained on code. A few things change:

**Tokenizer.** Code has different statistical patterns than English. Indentation matters, variable names repeat, and syntax tokens like `def`, `class`, `(`, `)` need to be handled efficiently. Code tokenizers typically treat common indent levels (2 spaces, 4 spaces, tab) as single tokens and handle code-specific punctuation better than a general-purpose BPE tokenizer.

**Context length.** Code files are long. A typical Python file might be 500 lines, and you often need cross-file context (imports from other modules, class definitions in other files). Production code models typically support 8k-16k tokens, with some (DeepSeek-Coder-V2, CodeLlama) going up to 100k+.

**Training data.** The training set is filtered, deduplicated code from GitHub (or similar sources), often with license filtering, quality scoring, and language balancing. StarCoder was trained on The Stack, roughly 6TB of permissively licensed source code across 300+ languages.

## 2. Building a Minimal Code Completion Engine

Let's build the core of a Copilot-like tool. We'll use a small code model and build the FIM inference pipeline from scratch.

#### 2.1 Picking a Model

For this post, I'm using [bigcode/tiny_starcoder_py](https://huggingface.co/bigcode/tiny_starcoder_py). It's a 164M parameter model trained on Python code with FIM support. Small enough to run on a laptop CPU (slowly) or a free Colab GPU (quickly), but large enough to generate real Python code.

For production work, you'd use something bigger: StarCoder2-15B, DeepSeek-Coder-33B, or CodeLlama-34B. The pipeline we're building is identical regardless of model size.


https://gist.github.com/cmenguy/6dec20daf49ab61daf386964fd19f38f


#### 2.2 FIM Inference

Here's the core function: given a prefix and suffix (the code before and after the cursor), generate the missing middle.


https://gist.github.com/cmenguy/f7a3906229b5e3ae3b8447f61adcb544


Low temperature (0.2) makes completions more deterministic, which is what you want for code. You're not looking for creativity; you're looking for the most likely correct code. `top_p=0.95` with nucleus sampling filters out very unlikely tokens.

Let's test it:


https://gist.github.com/cmenguy/f46bb9de6897e721174a86eec40a7701


The model should generate something like `if arr[mid] == target:\n            return mid`, which is the missing condition in a binary search. It sees the `elif` below and understands the branching structure.

#### 2.3 Left-to-Right Completion

FIM is for when there's code on both sides of the cursor. When the cursor is at the end of the file (or end of a function), regular left-to-right completion works:


https://gist.github.com/cmenguy/a4312183fbbb8d7da4b5560feed5bb22



https://gist.github.com/cmenguy/75ac60d83127ce1ee7fe88b7c65bb053


This should produce something reasonable: open the file, load the JSON, return the result. The docstring gives the model a clear signal about what the function should do.

#### 2.4 Putting It Together: An Editor-Like Interface

A real Copilot integration triggers on keystrokes, debounces requests, and sends completions back to the editor via LSP (Language Server Protocol) or a proprietary protocol. Here's a simplified version that simulates the decision logic:


https://gist.github.com/cmenguy/98269c675db730419da9f8d7e5cb1ec1



https://gist.github.com/cmenguy/512906b1ef86eab84a32d85ba51cc484


The function detects there's code below (`get_history` method) and uses FIM. It should fill in the `subtract` body in a way that's consistent with the `add` pattern above it, something like `result = a - b` followed by the history append.

## 3. Building an LSP-Style Completion Server

The completion engine above is the brains, but it needs a body. In the real world, Copilot talks to your editor through a server. Let's build a minimal HTTP server that acts like a code completion API. Any editor with an HTTP-capable plugin can talk to this.


https://gist.github.com/cmenguy/1824af07f7d2fc87e263f7c129fea85f



https://gist.github.com/cmenguy/5b1e013d8311d5049418a12ab2c9df64


Test it with curl:


https://gist.github.com/cmenguy/2de666d2c10dc4b7260dc59ce1477420


That's it. You now have a code completion server running locally. In production, you'd add request batching (queue up multiple keystrokes and only process the latest), caching (if the prefix hasn't changed, return the cached completion), and streaming (send tokens as they're generated rather than waiting for the full completion).

## 4. How Production Copilots Are Actually Built

Our toy server works, but production tools like Copilot, Cursor, and Cody have layers of optimization on top. Here's what the real architecture looks like:

#### 4.1 Context Gathering

The biggest gap between our toy and production is context. We only use the current file. Production tools gather context from:

**Open tabs.** If you have `utils.py` and `models.py` open, the completion engine sees relevant snippets from both.

**Import graph.** When the current file imports from `mypackage.data_loader`, the tool fetches the signatures and docstrings from that module.

**Repository-level retrieval.** Some tools (Sourcegraph Cody, Continue) use embedding-based search to find the most relevant code snippets across the entire repo and inject them into the prompt.


https://gist.github.com/cmenguy/1ffea440a1d546a3e96ab0a07964dffe



https://gist.github.com/cmenguy/636a7fc56ded01be0ffce7dff389e092


#### 4.2 Post-Processing

Raw model output often needs cleanup before showing it to the user:


https://gist.github.com/cmenguy/7428e172f8aa98cee5a566a155581bca


This handles three common issues: stopping the completion before it starts generating a new function (the model doesn't know where to stop), fixing indentation to match the surrounding code, and deduplicating against code that already exists below the cursor.

#### 4.3 Latency Budget

Here's the latency breakdown for a production code completion:


https://gist.github.com/cmenguy/0b60f38b535e7a662ce37168b15ad05c


The debounce is the most important optimization. You don't fire the model on every keystroke. You wait for the user to pause, then trigger. If they start typing again before the completion arrives, you cancel it and wait for the next pause.

Most completions are short: a single line, maybe two. The model generates 20-50 tokens, not 200. This is why code completion can feel instant even though the underlying model is the same one that takes 30 seconds to write a full function in a chat interface.

## 5. Fine-Tuning on Your Codebase

This is the part I actually set out to understand. A general code model writes generic Python. It doesn't know about your internal libraries, your coding conventions, or your domain-specific patterns. Fine-tuning bridges that gap.

#### 5.1 Why Fine-Tune (and When Not To)

There are three approaches to making a code model work with your codebase, and they exist on a spectrum of effort vs. payoff:

**Retrieval-Augmented Generation (RAG).** Fetch relevant code snippets from your repo and inject them into the prompt. Zero training required. Works well when your codebase follows standard patterns and the model just needs examples to follow.

**Fine-tuning with LoRA.** Train the model's weights on your codebase. Takes a few hours on a single GPU. The model learns your patterns, naming conventions, internal APIs. Works well when your codebase has strong conventions that differ from generic Python.

**Full fine-tuning.** Update all model weights. Expensive, requires multiple GPUs. Rarely necessary for code completion. Only makes sense if you have a very large, very distinctive codebase (hundreds of thousands of files) and the model needs to deeply learn a new programming language or paradigm.

For most teams, RAG gets you 70% of the way. Fine-tuning with LoRA gets you to 90%. Full fine-tuning is almost never worth the cost for code completion.

Here's a quick comparison:


https://gist.github.com/cmenguy/38afc9890f59cb0e97509b5ee5c69d33


#### 5.2 Preparing Training Data

This is where most people make mistakes. The quality of your training data determines the quality of your fine-tuned model. Let's build the data preparation pipeline.


https://gist.github.com/cmenguy/d0d2a3fe550a114a9a2322a4b9aa6316



https://gist.github.com/cmenguy/1b775970b73b438fa5c5629c0c33179f


#### 5.3 Creating FIM Training Examples

For fine-tuning a code completion model, you want FIM-formatted examples. Each example takes a file, picks a random split point, and creates a prefix-suffix-middle triple.


https://gist.github.com/cmenguy/bc5ca483224e69e0f50446264efbcffa



https://gist.github.com/cmenguy/c230d6a854d5c7f3b9e41e3728ad583a


For a medium-sized repo (500 Python files), this generates roughly 1500 training examples. That's plenty for LoRA fine-tuning.

#### 5.4 The LoRA Fine-Tuning Loop

If you read the [LoRA post](https://ai-terminal.net/llm/fine-tuning/2025/12/28/lora-deep-dive/), you know the mechanics. Here's the application to code completion. We're using `peft` and `trl` to keep it concise, but the underlying math is the same.


https://gist.github.com/cmenguy/ecef204a594785ef7e5f7734ab730ef7


This should print something like: `trainable params: 3,481,600 || all params: 167,625,728 || trainable%: 2.08%`. About 2% of the parameters, trained on your code. On a larger model like StarCoder2-15B, that percentage drops below 0.5%.

Why these target modules? In the StarCoder architecture, `c_attn` handles the query/key/value projections (this is where the model learns *what to attend to* in your code), `c_proj` is the attention output (how it combines attended information), and `c_fc` is the MLP (where pattern recognition happens). Targeting these three catches the most important weights for learning code patterns without touching the embedding layers.


https://gist.github.com/cmenguy/ca7166d903ba8d3ce6fbf55e687efdf7



https://gist.github.com/cmenguy/0f98b52e959c321a2922843277ff10e0


A few things about the hyperparameters. Learning rate `2e-4` is standard for LoRA fine-tuning. Much higher and you'll overfit fast; much lower and three epochs won't be enough. Cosine scheduler with warmup is the safest default. Batch size 4 with gradient accumulation 4 gives an effective batch size of 16. For code completion, you want small effective batch sizes because each example is fairly long (a full file context).

#### 5.5 Loading and Using the Fine-Tuned Model

After training, your LoRA adapter weights are saved separately from the base model. Loading them is straightforward:


https://gist.github.com/cmenguy/7869fe8d4b34a70f018c28484a899c05


The `merge_and_unload()` call is important for production. During training, LoRA adds a side path to each targeted layer: the output is `base_output + lora_output`. At inference time, you can merge the LoRA weights directly into the base weights (![equation](https://latex.codecogs.com/png.latex?\inline%20W%20%3D%20W_0%20%2B%20BA)) and run the model as if it were never LoRA-trained. Same quality, zero overhead.

#### 5.6 The Multi-Repo Problem: Adapters vs. Merging

This is the question I kept circling back to at work. We don't have one repo. We have a dozen. The data engineering team has their repo with Spark pipelines and Airflow DAGs. The ML team has a separate repo full of PyTorch training code and custom metrics. Marketing analytics has a repo that's mostly pandas and SQL generation. Each one has its own conventions, internal libraries, and idioms.

RAG handles this naturally. You point the embedding index at whichever repo the user is currently working in, and the retrieval layer fetches relevant snippets from that repo. Switch repos, switch index. No retraining, no weight changes.

LoRA is different. If you train a single adapter on all repos pooled together, the model learns a blurry average of everyone's patterns. The Spark repo uses `snake_case` everywhere and imports from `pyspark.sql.functions`. The ML repo uses short variable names and imports from custom internal modules. Training on both means the model sometimes suggests Spark imports when you're writing PyTorch code. Not great.

The better approach: **train a separate LoRA adapter per repo** (or per team, or per domain, whatever boundary makes sense). This is where LoRA's architecture actually shines. Each adapter is tiny. For the `tiny_starcoder_py` model we're using, the adapter is about 14MB. For a production 15B model with `r=16`, each adapter is around 50-100MB. You can store dozens of them.

Here's what that looks like in practice:


https://gist.github.com/cmenguy/e6ee680d7d2a0ea9a3ff36feb44dbb37


At serving time, you detect which repo the user is in (from their editor workspace, git remote, or an explicit config) and load the matching adapter. The base model stays in memory; you're just swapping a small set of low-rank matrices on top.

The question of **merge vs. keep separate** depends on your serving setup:

**Keep adapters separate** when you need to switch between repos during a session. PEFT supports loading multiple adapters onto the same base model and switching between them at inference time with zero reloading:


https://gist.github.com/cmenguy/539885245862a93de5d919e3a5a57634


The switching is near-instant because all adapters are already in memory. The memory overhead is small: each adapter adds about 2% to the base model's footprint with our config. Three adapters means ~6% extra memory, which is negligible.

**Merge into the base weights** when you're deploying a dedicated instance per team or per repo. Merging eliminates the LoRA forward pass overhead (the extra matrix multiplications through B and A), which saves a few milliseconds of latency per completion. If the data engineering team has their own completion server, merge their adapter and serve a single clean model.


https://gist.github.com/cmenguy/585e41faf4276ef605e39985608d74fd


There's a third option that I haven't tried in production but is worth knowing about: **adapter merging**. You can combine multiple LoRA adapters into one by weighted averaging their B and A matrices. This is useful if you want a single adapter that captures patterns from, say, three closely related repos:


https://gist.github.com/cmenguy/af4aeab8b0b71546118c9a75500cafb7


The weights control how much each repo's patterns contribute. If the data engineering repo is the primary codebase and the ML repo shares some common utilities, a 70/30 split makes sense. The `density` parameter in TIES-Merging controls sparsity: at 0.5, it keeps only the top 50% of adapter parameters by magnitude, which reduces interference between the two adapters.

In my experience, the cleanest setup for most companies is: **separate adapters, loaded on demand, no merging**. The memory is cheap, the switching is fast, and you avoid the blurring problem entirely. Reserve adapter merging for cases where repos genuinely share patterns and you've measured that the combined adapter outperforms either individual one.

Here's the decision tree I've landed on:


https://gist.github.com/cmenguy/95d11e4eb806017f32e5e54e4349c3ee


## 6. Evaluation: How Do You Know It's Working?

Fine-tuning without evaluation is guesswork. Here's how to measure whether your fine-tuned model is actually better at completing your code.

#### 6.1 Held-Out File Completion

The simplest test: hold out 10-20% of your repo's files, then measure how well the model completes code from those files.


https://gist.github.com/cmenguy/1f83f49c9aa216bcfa05e1834723ba78



https://gist.github.com/cmenguy/6ad01d9a2986da09037d7e212d8b7c8b


Exact match rates for single-line completion on a fine-tuned model typically land between 25-40% for internal codebases. That sounds low until you consider that there are often multiple valid ways to write a line of code. Edit distance is a better proxy: a fine-tuned model should average 3-5 characters of edit distance where a base model averages 15-20.

#### 6.2 The Real Test: Side-by-Side

Numbers are useful, but the most telling evaluation is qualitative. Take 10 examples from your codebase and compare the base model's completion against the fine-tuned model's:


https://gist.github.com/cmenguy/88304b374a807326cf5206a4b25d120e


What you're looking for: does the fine-tuned model use your internal function names? Does it follow your naming conventions (snake_case vs camelCase)? Does it import from your internal modules instead of suggesting `import pandas as pd` for everything? These qualitative differences are often more valuable than the quantitative metrics.

## 7. Techniques and Tradeoffs: A Practical Guide

Here's a summary of the main approaches to building a code completion tool, with honest assessments of when each one works.

#### 7.1 Approach 1: Off-the-Shelf Model + RAG

**What:** Use StarCoder2 or DeepSeek-Coder out of the box. Build a retrieval layer that fetches relevant code snippets from your repo and injects them into the prompt.

**How it works:** Embed all files in your repo with a code embedding model (like `nomic-embed-code` or `voyage-code-3`). At completion time, embed the current context, find the nearest neighbors, and prepend them to the prompt.

**Pros:** No training. Works immediately. Easy to keep up-to-date (just re-embed when code changes).

**Cons:** Limited context window means you can only inject a few snippets. The model still doesn't "know" your patterns; it just has examples in context. Retrieval quality is the bottleneck.

**Best for:** Small teams, codebases under 100k lines, standard Python patterns.

#### 7.2 Approach 2: LoRA Fine-Tuning (What We Built)

**What:** Fine-tune the model on your codebase with LoRA. The model learns your patterns, conventions, and internal APIs as weight updates.

**How it works:** As described above. Prepare FIM training data from your repo, train LoRA adapters for a few hours, merge and deploy.

**Pros:** Model internalizes your patterns. Better completions for internal APIs. Works with any context window since the knowledge is in the weights.

**Cons:** Stales over time as code changes. Need to retrain periodically (weekly or monthly). Requires a GPU for training.

**Best for:** Teams with strong coding conventions, internal libraries, or domain-specific patterns. Codebases over 50k lines.

#### 7.3 Approach 3: RAG + LoRA Together

**What:** Combine both approaches. Fine-tune for style and conventions, use RAG for specific API signatures and recent code.

**This is what most production Copilot-like tools actually do.** The fine-tuned model knows your general patterns; the RAG layer provides specific, up-to-date context. They're complementary.

**Best for:** Any team that's serious about code completion quality.

#### 7.4 Approach 4: Continued Pretraining

**What:** Before LoRA, do a round of continued pretraining on your full codebase. This teaches the model your "language" at a deeper level than LoRA can.

**How it works:** Standard causal language model training on your code, at a low learning rate, for 1-2 epochs. Then apply LoRA on top for task-specific fine-tuning.

**Pros:** Deeper knowledge of your codebase. Better for very large, distinctive codebases.

**Cons:** Expensive. Needs multiple GPUs and days of training. Risk of catastrophic forgetting (the model gets worse at general Python while getting better at your specific code).

**Best for:** Companies with 1M+ lines of highly distinctive code (custom DSLs, unusual frameworks).

## 8. What's Actually Running in Production

Let me close the loop on the original question: how does this relate to the NL2Code agent I'm building at work?

The Copilot pipeline we built here is for **inline code completion**: short, fast, triggered on keystrokes. The NL2Code use case is different: it's **instruction-following code generation**, where you describe what you want in natural language and the model writes a complete function or script.

But the fine-tuning pipeline is almost identical. The difference is in the training data format:


https://gist.github.com/cmenguy/739ef1a83f67847ce19f1c8325239929


Same base model, same LoRA setup, different data format. For the NL2Code agent, I swapped FIM examples for instruction/response pairs extracted from our codebase (using docstrings and function signatures as the "instruction" and the function body as the "response"). The fine-tuning loop didn't change at all.

The key takeaway from this whole exercise: there's no magic in Copilot. It's a code LLM with FIM training, a context-gathering layer, and (probably) fine-tuning on accepted completions. Each piece is understandable, buildable, and improvable. The competitive moat isn't the model architecture. It's the data flywheel: every accepted completion becomes a training signal that makes the next completion better. That's harder to replicate than any of the code in this post.

Next up in part 2: an NL2Code agent that takes a plain-English description and generates a complete function or script that fits into your codebase. Same fine-tuning pipeline, different data format, very different evaluation problem. And in part 3, we'll flip the direction entirely: instead of writing new code, we'll build a system that reads existing code, finds bugs, and suggests fixes.


---

*Originally published on [AI Terminal](https://ai-terminal.net/llm/ai-engineering/code-gen/2026/03/04/build-your-own-copilot/).*

Tags: lora, copilot, nl2code, code-llm, fine-tuning
