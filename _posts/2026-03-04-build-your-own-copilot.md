---
layout: post
title: "Build Your Own Copilot in Pure Python"
date: 2026-03-04 10:00:00 -0800
categories: [llm, ai-engineering, code-gen]
tags: [copilot, code-completion, fill-in-the-middle, fine-tuning, lora, code-llm, nl2code, from-scratch]
author: cmenguy
image: /assets/images/posts/2026-03-04-build-your-own-copilot.png
colab_url: "https://colab.research.google.com/github/cmenguy/cmenguy.github.io/blob/main/notebooks/2026-03-04-build-your-own-copilot.ipynb"
colab_embed: false
github_notebook: "https://github.com/cmenguy/cmenguy.github.io/blob/main/notebooks/2026-03-04-build-your-own-copilot.ipynb"
notebook_description: "A complete, runnable AI code completion tool: FIM inference, LSP-style server, and LoRA fine-tuning on a custom codebase."
---

In the [skills post](/llm/ai-engineering/agents/2026/02/05/skills-deep-dive/) I mentioned we'd been building agent skills at work, mostly for marketing ML workflows. One of those workflows is NL2Code: a user describes what they want in plain English ("build me a lookalike audience from this seed list using cosine similarity"), and the agent writes the Python code. It works surprisingly well for self-contained scripts. But it fell apart the moment we needed the generated code to fit into our existing codebase. The agent would write perfectly valid Python that imported libraries we don't use, called APIs that don't exist in our stack, and followed patterns that look nothing like the code we actually ship.

That got me thinking about how tools like Copilot actually work. Not the product, but the machinery underneath: how do you take a general-purpose code model and make it write code that feels like it belongs in *your* repo? I spent a few weekends digging into the full pipeline, from how code completion models are structured, to how fill-in-the-middle works, to how you fine-tune on a specific codebase. This post is what came out of that: a working AI coding tool, built from scratch in Python, that you can point at your own code.

This is **part 1 of a 3-part series** on AI-assisted code generation. This post covers the inline completion side: FIM, code models, fine-tuning, and building a working Copilot clone. Part 2 will tackle NL2Code: instruction-following code generation, where you describe what you want in English and the model writes the full implementation. Part 3 will go after the other end of the lifecycle: an AI-assisted bug detection and fix suggestion system that reads your code, spots problems, and proposes patches.

## How Code Completion Actually Works

Before building anything, let's understand what's happening when Copilot suggests code in your editor.

A code completion model is a language model trained on code. Same transformer architecture as GPT, same next-token prediction objective, just trained on GitHub repos instead of web text. Models like CodeLlama, StarCoder, and DeepSeek-Coder are all in this family. They predict: given the code written so far, what token comes next?

But code completion has a problem that regular text doesn't. When you're typing in an editor, the cursor isn't always at the end of the file. You might be in the middle of a function, with code above *and* below. A model trained purely on left-to-right next-token prediction can only see what's above the cursor. It has no idea what comes after.

This is where **Fill-in-the-Middle (FIM)** comes in.

### Fill-in-the-Middle: The Key Trick

FIM was introduced in the [Bavarian et al. 2022](https://arxiv.org/abs/2207.14255) paper from OpenAI. The idea is simple: during training, you take a code file, split it into three parts (prefix, middle, suffix), and train the model to predict the middle given the prefix and suffix.

Here's how a normal training example looks:

```
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

And here's the same example reformatted for FIM training:

```
<fim_prefix>def fibonacci(n):
    if n <= 1:
        return n
    <fim_suffix>
<fim_middle>return fibonacci(n-1) + fibonacci(n-2)
```

The model learns to take the prefix (everything before the cursor) and the suffix (everything after), and generate the middle (what should go where the cursor is). Three special tokens mark the boundaries: `<fim_prefix>`, `<fim_suffix>`, and `<fim_middle>`.

There are two formats for arranging these pieces:

**PSM (Prefix-Suffix-Middle):** `<fim_prefix>PREFIX<fim_suffix>SUFFIX<fim_middle>MIDDLE`

**SPM (Suffix-Prefix-Middle):** `<fim_suffix>SUFFIX<fim_prefix>PREFIX<fim_middle>MIDDLE`

Most production models (StarCoder, CodeLlama, DeepSeek-Coder) use PSM. The key insight from the paper: FIM can be added to training with almost no cost. You take a fraction of your training examples (typically 50-90%), reformat them as FIM, and the model learns both regular left-to-right completion and fill-in-the-middle, with no degradation on either task.

### What Makes Code Models Different

Code LLMs aren't just GPT trained on code. A few things change:

**Tokenizer.** Code has different statistical patterns than English. Indentation matters, variable names repeat, and syntax tokens like `def`, `class`, `(`, `)` need to be handled efficiently. Code tokenizers typically treat common indent levels (2 spaces, 4 spaces, tab) as single tokens and handle code-specific punctuation better than a general-purpose BPE tokenizer.

**Context length.** Code files are long. A typical Python file might be 500 lines, and you often need cross-file context (imports from other modules, class definitions in other files). Production code models typically support 8k-16k tokens, with some (DeepSeek-Coder-V2, CodeLlama) going up to 100k+.

**Training data.** The training set is filtered, deduplicated code from GitHub (or similar sources), often with license filtering, quality scoring, and language balancing. StarCoder was trained on The Stack, roughly 6TB of permissively licensed source code across 300+ languages.

## Building a Minimal Code Completion Engine

Let's build the core of a Copilot-like tool. We'll use a small code model and build the FIM inference pipeline from scratch.

### Picking a Model

For this post, I'm using [bigcode/tiny_starcoder_py](https://huggingface.co/bigcode/tiny_starcoder_py). It's a 164M parameter model trained on Python code with FIM support. Small enough to run on a laptop CPU (slowly) or a free Colab GPU (quickly), but large enough to generate real Python code.

For production work, you'd use something bigger: StarCoder2-15B, DeepSeek-Coder-33B, or CodeLlama-34B. The pipeline we're building is identical regardless of model size.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "bigcode/tiny_starcoder_py"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

### FIM Inference

Here's the core function: given a prefix and suffix (the code before and after the cursor), generate the missing middle.

```python
import torch

FIM_PREFIX = "<fim_prefix>"
FIM_SUFFIX = "<fim_suffix>"
FIM_MIDDLE = "<fim_middle>"

def fim_complete(prefix, suffix, max_new_tokens=64,
                 temperature=0.2, top_p=0.95):
    """Generate code to fill between prefix and suffix."""
    prompt = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    generated = outputs[0, inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(generated, skip_special_tokens=True)

    # Stop at end-of-middle token if present
    if "<fim_middle>" in completion:
        completion = completion[:completion.index("<fim_middle>")]
    if "<|endoftext|>" in completion:
        completion = completion[:completion.index("<|endoftext|>")]

    return completion.rstrip()
```

Low temperature (0.2) makes completions more deterministic, which is what you want for code. You're not looking for creativity; you're looking for the most likely correct code. `top_p=0.95` with nucleus sampling filters out very unlikely tokens.

Let's test it:

```python
prefix = """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        """

suffix = """
        elif arr[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    return -1
"""

print(fim_complete(prefix, suffix))
```

The model should generate something like `if arr[mid] == target:\n            return mid`, which is the missing condition in a binary search. It sees the `elif` below and understands the branching structure.

### Left-to-Right Completion

FIM is for when there's code on both sides of the cursor. When the cursor is at the end of the file (or end of a function), regular left-to-right completion works:

```python
def complete_code(code, max_new_tokens=128,
                  temperature=0.2, top_p=0.95):
    """Standard left-to-right code completion."""
    inputs = tokenizer(code, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0, inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(generated, skip_special_tokens=True)

    if "<|endoftext|>" in completion:
        completion = completion[:completion.index("<|endoftext|>")]

    return completion.rstrip()
```

```python
code = """import json

def parse_config(filepath):
    \"\"\"Read a JSON config file and return a dict.\"\"\"
"""

print(complete_code(code))
```

This should produce something reasonable: open the file, load the JSON, return the result. The docstring gives the model a clear signal about what the function should do.

### Putting It Together: An Editor-Like Interface

A real Copilot integration triggers on keystrokes, debounces requests, and sends completions back to the editor via LSP (Language Server Protocol) or a proprietary protocol. Here's a simplified version that simulates the decision logic:

```python
def get_completion(document, cursor_line, cursor_col):
    """
    Given a document (list of lines), cursor position,
    decide whether to use FIM or left-to-right completion.
    """
    lines = document.split("\n") if isinstance(document, str) else document

    # Split at cursor position
    prefix_lines = lines[:cursor_line]
    current_line = lines[cursor_line][:cursor_col]
    suffix_lines = lines[cursor_line + 1:]

    prefix = "\n".join(prefix_lines + [current_line])
    suffix = "\n".join(suffix_lines)

    # If there's meaningful code below, use FIM
    has_suffix = any(line.strip() for line in suffix_lines)

    if has_suffix:
        return fim_complete(prefix, suffix)
    else:
        return complete_code(prefix)
```

```python
document = """class Calculator:
    def __init__(self):
        self.history = []

    def add(self, a, b):
        result = a + b
        self.history.append(('add', a, b, result))
        return result

    def subtract(self, a, b):

    def get_history(self):
        return self.history
"""

# Cursor is inside the empty subtract method (line 11, col 0)
completion = get_completion(document, cursor_line=11, cursor_col=0)
print(completion)
```

The function detects there's code below (`get_history` method) and uses FIM. It should fill in the `subtract` body in a way that's consistent with the `add` pattern above it, something like `result = a - b` followed by the history append.

## Building an LSP-Style Completion Server

The completion engine above is the brains, but it needs a body. In the real world, Copilot talks to your editor through a server. Let's build a minimal HTTP server that acts like a code completion API. Any editor with an HTTP-capable plugin can talk to this.

```python
from http.server import HTTPServer, BaseHTTPRequestHandler
import json as json_mod
import threading

class CompletionHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        body = json_mod.loads(self.rfile.read(content_length))

        document = body["document"]
        line = body["line"]
        col = body["col"]

        completion = get_completion(document, line, col)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response = json_mod.dumps({"completion": completion})
        self.wfile.write(response.encode())

    def log_message(self, format, *args):
        pass  # Suppress default logging
```

```python
def start_server(port=8765):
    server = HTTPServer(("localhost", port), CompletionHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Completion server running on http://localhost:{port}")
    return server
```

Test it with curl:

```bash
curl -s http://localhost:8765 -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "document": "def greet(name):\n    ",
    "line": 1,
    "col": 4
  }' | python -m json.tool
```

That's it. You now have a code completion server running locally. In production, you'd add request batching (queue up multiple keystrokes and only process the latest), caching (if the prefix hasn't changed, return the cached completion), and streaming (send tokens as they're generated rather than waiting for the full completion).

## How Production Copilots Are Actually Built

Our toy server works, but production tools like Copilot, Cursor, and Cody have layers of optimization on top. Here's what the real architecture looks like:

### Context Gathering

The biggest gap between our toy and production is context. We only use the current file. Production tools gather context from:

**Open tabs.** If you have `utils.py` and `models.py` open, the completion engine sees relevant snippets from both.

**Import graph.** When the current file imports from `mypackage.data_loader`, the tool fetches the signatures and docstrings from that module.

**Repository-level retrieval.** Some tools (Sourcegraph Cody, Continue) use embedding-based search to find the most relevant code snippets across the entire repo and inject them into the prompt.

```python
def gather_context(filepath, project_root, max_context_tokens=2048):
    """
    Collect relevant context beyond the current file.
    This is a simplified version of what production tools do.
    """
    import os
    import ast

    with open(filepath) as f:
        source = f.read()

    context_parts = []

    # Parse imports and fetch their signatures
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            # Convert module path to file path
            module_path = os.path.join(
                project_root,
                node.module.replace(".", "/") + ".py"
            )
            if os.path.exists(module_path):
                sigs = extract_signatures(module_path)
                if sigs:
                    context_parts.append(
                        f"# From {node.module}:\n{sigs}"
                    )

    return "\n\n".join(context_parts)
```

```python
def extract_signatures(filepath):
    """Pull function/class signatures from a Python file."""
    import ast

    with open(filepath) as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return ""

    sigs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = ast.dump(node.args)
            # Reconstruct a simple signature
            arg_names = [a.arg for a in node.args.args]
            sig = f"def {node.name}({', '.join(arg_names)}): ..."
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)):
                docstring = node.body[0].value.value
                first_line = docstring.strip().split("\n")[0]
                sig += f'  # {first_line}'
            sigs.append(sig)
        elif isinstance(node, ast.ClassDef):
            sigs.append(f"class {node.name}: ...")

    return "\n".join(sigs)
```

### Post-Processing

Raw model output often needs cleanup before showing it to the user:

```python
def postprocess_completion(completion, indent_level,
                           existing_lines=None):
    """
    Clean up a raw model completion for display.
    """
    lines = completion.split("\n")
    result = []

    for line in lines:
        # Stop at a line that starts a new top-level definition
        stripped = line.lstrip()
        if stripped.startswith(("class ", "def ")) and not line.startswith(" "):
            break

        # Fix indent: ensure consistent with surrounding code
        if line.strip():
            result.append(" " * indent_level + line.lstrip())
        else:
            result.append("")

    # Remove trailing empty lines
    while result and not result[-1].strip():
        result.pop()

    # Deduplicate against existing code
    if existing_lines:
        while (result and existing_lines
               and result[-1].strip() == existing_lines[0].strip()):
            result.pop()

    return "\n".join(result)
```

This handles three common issues: stopping the completion before it starts generating a new function (the model doesn't know where to stop), fixing indentation to match the surrounding code, and deduplicating against code that already exists below the cursor.

### Latency Budget

Here's the latency breakdown for a production code completion:

| Component | Time | Notes |
|-----------|------|-------|
| Keystroke debounce | 200-500ms | Wait for user to pause typing |
| Context gathering | 10-50ms | Parse imports, fetch signatures |
| Tokenization | 5-10ms | Convert text to token IDs |
| Model inference | 100-500ms | Generate 20-50 tokens |
| Post-processing | 1-5ms | Clean up, deduplicate |
| **Total** | **~300-1000ms** | User sees completion after typing pause |

The debounce is the most important optimization. You don't fire the model on every keystroke. You wait for the user to pause, then trigger. If they start typing again before the completion arrives, you cancel it and wait for the next pause.

Most completions are short: a single line, maybe two. The model generates 20-50 tokens, not 200. This is why code completion can feel instant even though the underlying model is the same one that takes 30 seconds to write a full function in a chat interface.

## Fine-Tuning on Your Codebase

This is the part I actually set out to understand. A general code model writes generic Python. It doesn't know about your internal libraries, your coding conventions, or your domain-specific patterns. Fine-tuning bridges that gap.

### Why Fine-Tune (and When Not To)

There are three approaches to making a code model work with your codebase, and they exist on a spectrum of effort vs. payoff:

**Retrieval-Augmented Generation (RAG).** Fetch relevant code snippets from your repo and inject them into the prompt. Zero training required. Works well when your codebase follows standard patterns and the model just needs examples to follow.

**Fine-tuning with LoRA.** Train the model's weights on your codebase. Takes a few hours on a single GPU. The model learns your patterns, naming conventions, internal APIs. Works well when your codebase has strong conventions that differ from generic Python.

**Full fine-tuning.** Update all model weights. Expensive, requires multiple GPUs. Rarely necessary for code completion. Only makes sense if you have a very large, very distinctive codebase (hundreds of thousands of files) and the model needs to deeply learn a new programming language or paradigm.

For most teams, RAG gets you 70% of the way. Fine-tuning with LoRA gets you to 90%. Full fine-tuning is almost never worth the cost for code completion.

Here's a quick comparison:

| Approach | Training Cost | Infra Needs | Quality Gain | When to Use |
|----------|--------------|-------------|-------------|-------------|
| RAG only | None | Embedding index | Good | Standard patterns, small codebase |
| LoRA fine-tune | ~$10-50 (cloud GPU) | 1 GPU, 4-8 hours | Very good | Strong conventions, internal APIs |
| Full fine-tune | ~$500-5000 | Multi-GPU, days | Marginal over LoRA | New language/paradigm, huge codebase |

### Preparing Training Data

This is where most people make mistakes. The quality of your training data determines the quality of your fine-tuned model. Let's build the data preparation pipeline.

```python
import os
import random

def collect_python_files(repo_path, max_file_size=50000):
    """Walk a repo and collect Python files."""
    files = []
    skip_dirs = {
        ".git", "__pycache__", "node_modules",
        ".venv", "venv", ".tox", "build", "dist",
        ".eggs", "*.egg-info",
    }

    for root, dirs, filenames in os.walk(repo_path):
        # Skip hidden and build directories
        dirs[:] = [
            d for d in dirs
            if d not in skip_dirs and not d.startswith(".")
        ]

        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            filepath = os.path.join(root, fname)
            size = os.path.getsize(filepath)
            if size > max_file_size or size < 50:
                continue
            files.append(filepath)

    return files
```

```python
def read_and_filter(filepath):
    """Read a file and apply quality filters."""
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Skip auto-generated files
    first_line = content.split("\n")[0] if content else ""
    if "auto-generated" in first_line.lower():
        return None
    if "do not edit" in first_line.lower():
        return None

    # Skip test files that are mostly boilerplate
    lines = content.split("\n")
    code_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
    if len(code_lines) < 5:
        return None

    return content
```

### Creating FIM Training Examples

For fine-tuning a code completion model, you want FIM-formatted examples. Each example takes a file, picks a random split point, and creates a prefix-suffix-middle triple.

```python
def create_fim_examples(content, num_splits=3):
    """
    Create FIM training examples from a single file.
    Split at random points to create prefix/middle/suffix triples.
    """
    lines = content.split("\n")
    if len(lines) < 5:
        return []

    examples = []
    for _ in range(num_splits):
        # Pick a random split point (avoid very start/end)
        split_start = random.randint(1, max(1, len(lines) - 3))
        # Middle is 1-5 lines
        mid_len = random.randint(1, min(5, len(lines) - split_start))
        split_end = split_start + mid_len

        prefix = "\n".join(lines[:split_start])
        middle = "\n".join(lines[split_start:split_end])
        suffix = "\n".join(lines[split_end:])

        # Skip if any part is empty
        if not middle.strip():
            continue

        fim_text = (
            f"<fim_prefix>{prefix}\n"
            f"<fim_suffix>\n{suffix}"
            f"<fim_middle>{middle}"
        )
        examples.append(fim_text)

    return examples
```

```python
def build_dataset(repo_path, output_path="train_data.jsonl"):
    """Build a JSONL training dataset from a repo."""
    import json

    files = collect_python_files(repo_path)
    print(f"Found {len(files)} Python files")

    examples = []
    for filepath in files:
        content = read_and_filter(filepath)
        if content is None:
            continue
        file_examples = create_fim_examples(content)
        examples.extend(file_examples)

    random.shuffle(examples)

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps({"text": ex}) + "\n")

    print(f"Created {len(examples)} training examples")
    return examples
```

For a medium-sized repo (500 Python files), this generates roughly 1500 training examples. That's plenty for LoRA fine-tuning.

### The LoRA Fine-Tuning Loop

If you read the [LoRA post](/llm/fine-tuning/2025/12/28/lora-deep-dive/), you know the mechanics. Here's the application to code completion. We're using `peft` and `trl` to keep it concise, but the underlying math is the same.

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "c_attn",   # attention QKV projection
        "c_proj",   # attention output projection
        "c_fc",     # MLP first layer
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
```

This should print something like: `trainable params: 3,481,600 || all params: 167,625,728 || trainable%: 2.08%`. About 2% of the parameters, trained on your code. On a larger model like StarCoder2-15B, that percentage drops below 0.5%.

Why these target modules? In the StarCoder architecture, `c_attn` handles the query/key/value projections (this is where the model learns *what to attend to* in your code), `c_proj` is the attention output (how it combines attended information), and `c_fc` is the MLP (where pattern recognition happens). Targeting these three catches the most important weights for learning code patterns without touching the embedding layers.

```python
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# Load our prepared dataset
dataset = load_dataset(
    "json", data_files="train_data.jsonl", split="train"
)

training_config = SFTConfig(
    output_dir="./copilot-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,  # Use bfloat16 if GPU supports it
    max_seq_length=1024,
    dataset_text_field="text",
)
```

```python
trainer = SFTTrainer(
    model=peft_model,
    args=training_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()
```

A few things about the hyperparameters. Learning rate `2e-4` is standard for LoRA fine-tuning. Much higher and you'll overfit fast; much lower and three epochs won't be enough. Cosine scheduler with warmup is the safest default. Batch size 4 with gradient accumulation 4 gives an effective batch size of 16. For code completion, you want small effective batch sizes because each example is fairly long (a full file context).

### Loading and Using the Fine-Tuned Model

After training, your LoRA adapter weights are saved separately from the base model. Loading them is straightforward:

```python
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(model_name)

# Load the LoRA adapter on top
finetuned_model = PeftModel.from_pretrained(
    base_model, "./copilot-finetuned/checkpoint-final"
)

# For production: merge the adapter into the base weights
# This eliminates the LoRA overhead at inference time
merged_model = finetuned_model.merge_and_unload()
```

The `merge_and_unload()` call is important for production. During training, LoRA adds a side path to each targeted layer: the output is `base_output + lora_output`. At inference time, you can merge the LoRA weights directly into the base weights ($W = W_0 + BA$) and run the model as if it were never LoRA-trained. Same quality, zero overhead.

### The Multi-Repo Problem: Adapters vs. Merging

This is the question I kept circling back to at work. We don't have one repo. We have a dozen. The data engineering team has their repo with Spark pipelines and Airflow DAGs. The ML team has a separate repo full of PyTorch training code and custom metrics. Marketing analytics has a repo that's mostly pandas and SQL generation. Each one has its own conventions, internal libraries, and idioms.

RAG handles this naturally. You point the embedding index at whichever repo the user is currently working in, and the retrieval layer fetches relevant snippets from that repo. Switch repos, switch index. No retraining, no weight changes.

LoRA is different. If you train a single adapter on all repos pooled together, the model learns a blurry average of everyone's patterns. The Spark repo uses `snake_case` everywhere and imports from `pyspark.sql.functions`. The ML repo uses short variable names and imports from custom internal modules. Training on both means the model sometimes suggests Spark imports when you're writing PyTorch code. Not great.

The better approach: **train a separate LoRA adapter per repo** (or per team, or per domain, whatever boundary makes sense). This is where LoRA's architecture actually shines. Each adapter is tiny. For the `tiny_starcoder_py` model we're using, the adapter is about 14MB. For a production 15B model with `r=16`, each adapter is around 50-100MB. You can store dozens of them.

Here's what that looks like in practice:

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(model_name)

# Each repo gets its own adapter directory
adapters = {
    "data-eng": "./adapters/data-engineering-repo",
    "ml-training": "./adapters/ml-training-repo",
    "marketing": "./adapters/marketing-analytics-repo",
}

def load_adapter_for_repo(repo_name):
    """Load the right LoRA adapter based on which repo is active."""
    adapter_path = adapters[repo_name]
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model
```

At serving time, you detect which repo the user is in (from their editor workspace, git remote, or an explicit config) and load the matching adapter. The base model stays in memory; you're just swapping a small set of low-rank matrices on top.

The question of **merge vs. keep separate** depends on your serving setup:

**Keep adapters separate** when you need to switch between repos during a session. PEFT supports loading multiple adapters onto the same base model and switching between them at inference time with zero reloading:

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(model_name)

# Load multiple adapters onto the same base model
model = PeftModel.from_pretrained(
    base_model, "./adapters/data-engineering-repo",
    adapter_name="data_eng",
)
model.load_adapter(
    "./adapters/ml-training-repo",
    adapter_name="ml_training",
)
model.load_adapter(
    "./adapters/marketing-analytics-repo",
    adapter_name="marketing",
)

# Switch based on which repo the user is editing
model.set_adapter("data_eng")
# ... serve completions for the data engineering repo ...

model.set_adapter("ml_training")
# ... now completions use ML repo patterns ...
```

The switching is near-instant because all adapters are already in memory. The memory overhead is small: each adapter adds about 2% to the base model's footprint with our config. Three adapters means ~6% extra memory, which is negligible.

**Merge into the base weights** when you're deploying a dedicated instance per team or per repo. Merging eliminates the LoRA forward pass overhead (the extra matrix multiplications through B and A), which saves a few milliseconds of latency per completion. If the data engineering team has their own completion server, merge their adapter and serve a single clean model.

```python
# For a dedicated per-repo deployment
base = AutoModelForCausalLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(base, "./adapters/data-engineering-repo")
merged = model.merge_and_unload()
merged.save_pretrained("./deployed-models/data-eng-copilot")
```

There's a third option that I haven't tried in production but is worth knowing about: **adapter merging**. You can combine multiple LoRA adapters into one by weighted averaging their B and A matrices. This is useful if you want a single adapter that captures patterns from, say, three closely related repos:

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(
    base_model, "./adapters/data-engineering-repo",
    adapter_name="data_eng",
)
model.load_adapter(
    "./adapters/ml-training-repo",
    adapter_name="ml_training",
)

# Merge two adapters with weighted combination
# "ties" = TIES-Merging (Yadav et al., 2023)
model.add_weighted_adapter(
    adapters=["data_eng", "ml_training"],
    weights=[0.7, 0.3],
    adapter_name="combined",
    combination_type="ties",
    density=0.5,
)
model.set_adapter("combined")
```

The weights control how much each repo's patterns contribute. If the data engineering repo is the primary codebase and the ML repo shares some common utilities, a 70/30 split makes sense. The `density` parameter in TIES-Merging controls sparsity: at 0.5, it keeps only the top 50% of adapter parameters by magnitude, which reduces interference between the two adapters.

In my experience, the cleanest setup for most companies is: **separate adapters, loaded on demand, no merging**. The memory is cheap, the switching is fast, and you avoid the blurring problem entirely. Reserve adapter merging for cases where repos genuinely share patterns and you've measured that the combined adapter outperforms either individual one.

Here's the decision tree I've landed on:

| Scenario | Strategy |
|----------|----------|
| 1 repo, 1 team | Train one adapter, merge into base |
| 3-5 repos, shared server | Separate adapters, switch at runtime |
| 10+ repos, dedicated infra per team | Merge per-repo adapter into dedicated model |
| Closely related repos with shared libs | Experiment with TIES-Merging, measure quality |

## Evaluation: How Do You Know It's Working?

Fine-tuning without evaluation is guesswork. Here's how to measure whether your fine-tuned model is actually better at completing your code.

### Held-Out File Completion

The simplest test: hold out 10-20% of your repo's files, then measure how well the model completes code from those files.

```python
def eval_completion_accuracy(model, tokenizer, test_files,
                             num_samples=100):
    """
    For each test file, mask a random chunk and measure
    whether the model's completion matches the original.
    """
    exact_matches = 0
    edit_distances = []

    for filepath in test_files[:num_samples]:
        content = read_and_filter(filepath)
        if content is None:
            continue

        lines = content.split("\n")
        if len(lines) < 5:
            continue

        # Pick a random line to mask
        mask_line = random.randint(1, len(lines) - 2)
        prefix = "\n".join(lines[:mask_line])
        expected = lines[mask_line]
        suffix = "\n".join(lines[mask_line + 1:])

        generated = fim_complete(prefix, suffix, max_new_tokens=64)
        first_line = generated.split("\n")[0]

        if first_line.strip() == expected.strip():
            exact_matches += 1

        ed = edit_distance(first_line.strip(), expected.strip())
        edit_distances.append(ed)

    accuracy = exact_matches / min(num_samples, len(test_files))
    avg_edit_dist = sum(edit_distances) / len(edit_distances)
    return {
        "exact_match": accuracy,
        "avg_edit_distance": avg_edit_dist,
    }
```

```python
def edit_distance(s1, s2):
    """Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]
```

Exact match rates for single-line completion on a fine-tuned model typically land between 25-40% for internal codebases. That sounds low until you consider that there are often multiple valid ways to write a line of code. Edit distance is a better proxy: a fine-tuned model should average 3-5 characters of edit distance where a base model averages 15-20.

### The Real Test: Side-by-Side

Numbers are useful, but the most telling evaluation is qualitative. Take 10 examples from your codebase and compare the base model's completion against the fine-tuned model's:

```python
def compare_models(base_model, finetuned_model, test_cases):
    """Side-by-side comparison of completions."""
    for prefix, suffix in test_cases:
        base_completion = fim_complete(
            prefix, suffix, model=base_model
        )
        ft_completion = fim_complete(
            prefix, suffix, model=finetuned_model
        )

        print(f"PREFIX: {prefix[-100:]}...")
        print(f"SUFFIX: ...{suffix[:100]}")
        print(f"BASE:   {base_completion}")
        print(f"TUNED:  {ft_completion}")
        print("-" * 60)
```

What you're looking for: does the fine-tuned model use your internal function names? Does it follow your naming conventions (snake_case vs camelCase)? Does it import from your internal modules instead of suggesting `import pandas as pd` for everything? These qualitative differences are often more valuable than the quantitative metrics.

## Techniques and Tradeoffs: A Practical Guide

Here's a summary of the main approaches to building a code completion tool, with honest assessments of when each one works.

### Approach 1: Off-the-Shelf Model + RAG

**What:** Use StarCoder2 or DeepSeek-Coder out of the box. Build a retrieval layer that fetches relevant code snippets from your repo and injects them into the prompt.

**How it works:** Embed all files in your repo with a code embedding model (like `nomic-embed-code` or `voyage-code-3`). At completion time, embed the current context, find the nearest neighbors, and prepend them to the prompt.

**Pros:** No training. Works immediately. Easy to keep up-to-date (just re-embed when code changes).

**Cons:** Limited context window means you can only inject a few snippets. The model still doesn't "know" your patterns; it just has examples in context. Retrieval quality is the bottleneck.

**Best for:** Small teams, codebases under 100k lines, standard Python patterns.

### Approach 2: LoRA Fine-Tuning (What We Built)

**What:** Fine-tune the model on your codebase with LoRA. The model learns your patterns, conventions, and internal APIs as weight updates.

**How it works:** As described above. Prepare FIM training data from your repo, train LoRA adapters for a few hours, merge and deploy.

**Pros:** Model internalizes your patterns. Better completions for internal APIs. Works with any context window since the knowledge is in the weights.

**Cons:** Stales over time as code changes. Need to retrain periodically (weekly or monthly). Requires a GPU for training.

**Best for:** Teams with strong coding conventions, internal libraries, or domain-specific patterns. Codebases over 50k lines.

### Approach 3: RAG + LoRA Together

**What:** Combine both approaches. Fine-tune for style and conventions, use RAG for specific API signatures and recent code.

**This is what most production Copilot-like tools actually do.** The fine-tuned model knows your general patterns; the RAG layer provides specific, up-to-date context. They're complementary.

**Best for:** Any team that's serious about code completion quality.

### Approach 4: Continued Pretraining

**What:** Before LoRA, do a round of continued pretraining on your full codebase. This teaches the model your "language" at a deeper level than LoRA can.

**How it works:** Standard causal language model training on your code, at a low learning rate, for 1-2 epochs. Then apply LoRA on top for task-specific fine-tuning.

**Pros:** Deeper knowledge of your codebase. Better for very large, distinctive codebases.

**Cons:** Expensive. Needs multiple GPUs and days of training. Risk of catastrophic forgetting (the model gets worse at general Python while getting better at your specific code).

**Best for:** Companies with 1M+ lines of highly distinctive code (custom DSLs, unusual frameworks).

## What's Actually Running in Production

Let me close the loop on the original question: how does this relate to the NL2Code agent I'm building at work?

The Copilot pipeline we built here is for **inline code completion**: short, fast, triggered on keystrokes. The NL2Code use case is different: it's **instruction-following code generation**, where you describe what you want in natural language and the model writes a complete function or script.

But the fine-tuning pipeline is almost identical. The difference is in the training data format:

```python
# FIM format (for code completion):
"<fim_prefix>def process(data):\n    <fim_suffix>\n    return result<fim_middle>filtered = [x for x in data if x > 0]"

# Instruction format (for NL2Code):
"### Instruction:\nWrite a function that filters positive numbers from a list.\n\n### Response:\ndef process(data):\n    filtered = [x for x in data if x > 0]\n    return filtered"
```

Same base model, same LoRA setup, different data format. For the NL2Code agent, I swapped FIM examples for instruction/response pairs extracted from our codebase (using docstrings and function signatures as the "instruction" and the function body as the "response"). The fine-tuning loop didn't change at all.

The key takeaway from this whole exercise: there's no magic in Copilot. It's a code LLM with FIM training, a context-gathering layer, and (probably) fine-tuning on accepted completions. Each piece is understandable, buildable, and improvable. The competitive moat isn't the model architecture. It's the data flywheel: every accepted completion becomes a training signal that makes the next completion better. That's harder to replicate than any of the code in this post.

Next up in part 2: an NL2Code agent that takes a plain-English description and generates a complete function or script that fits into your codebase. Same fine-tuning pipeline, different data format, very different evaluation problem. And in part 3, we'll flip the direction entirely: instead of writing new code, we'll build a system that reads existing code, finds bugs, and suggests fixes.
