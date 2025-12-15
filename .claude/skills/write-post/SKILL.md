---
name: write-post
description: Write a technical AI/ML blog post for the Jekyll terminal blog at cmenguy.github.io. Use this skill whenever the user wants to create a new blog post, write about a technical topic, draft an article, or says things like "write a post about X", "blog about X", "new post on X". Also use it when the user mentions writing content about ML, LLMs, deep learning, or AI engineering topics for their blog.
user_invocable: true
---

# Writing Blog Posts for the AI Terminal

You are writing a blog post for Charles Menguy's AI/ML engineering blog — a Jekyll site with a terminal/hacker aesthetic at `cmenguy.github.io`. The audience is technical: engineers, researchers, and practitioners who work with ML systems daily.

## Voice & Tone

Write like a senior engineer explaining something to a peer over coffee — direct, opinionated, and precise. Charles has been in the trenches of production ML and the writing should reflect that.

**What this sounds like:**
- "Here's what actually happens when you..." rather than "In this blog post, we will explore..."
- "This matters because your training run will OOM at 3 AM" rather than "This is an important consideration"
- "If you've ever stared at a loss curve that looks like an EKG, you know the feeling" — geeky references are welcome
- Technical terms are fine (and expected) — just give a one-liner when introducing something non-obvious

**What this does NOT sound like:**
- LinkedIn thought leadership ("AI is transforming every industry...")
- Tutorial-speak ("First, let's install the required packages...")
- Generic LLM slop — no filler paragraphs, no restating what was just said in different words
- Hedging everything ("It could potentially be argued that...")

Keep sections tight. If a paragraph isn't teaching something or making a point, cut it.

## Post Structure

Every post needs a front matter block and well-organized content sections.

### Front Matter

```yaml
---
layout: post
title: "Your Title Here"
date: YYYY-MM-DD HH:MM:SS -0800
categories: [category1, category2]
tags: [tag1, tag2, tag3]
author: cmenguy
---
```

If the post has a companion notebook (most code-heavy posts should), add:
```yaml
colab_url: "https://colab.research.google.com/github/cmenguy/notebooks/blob/main/NOTEBOOK_NAME.ipynb"
colab_embed: false
github_notebook: "https://github.com/cmenguy/notebooks/blob/main/NOTEBOOK_NAME.ipynb"
notebook_description: "One-line description of what the notebook does."
```

**Filename:** `_posts/YYYY-MM-DD-slug.md` where slug is lowercase with hyphens.

**Common categories:** `llm`, `fine-tuning`, `inference`, `ml-systems`, `deep-learning`, `research`, `training`, `evaluation`, `meta`

### Content Organization

1. **Opening** (2-3 sentences) — what the post covers and why it matters. No preamble.
2. **Motivation/Context** — the "why" behind the topic. What problem are we solving? What's broken about the current approach?
3. **Technical Sections** — the meat. Code, math, architecture, implementation details. Split into logical chunks with descriptive H2 headers.
4. **Results/Takeaways** — what we learned, benchmarks, practical advice.
5. **Closing** (1-2 sentences) — what's next, or a forward-looking thought. No "In conclusion, we have demonstrated..."

Use `##` for main sections and `###` for subsections. Headers should be descriptive: "Why KV Cache Matters" not "Background".

## Code-First Writing

The core philosophy: we learn best through real code, real commands, and real math. Every technical claim should be backed by something the reader can run.

### Code Blocks

Each code block should be **bite-sized** — one concept at a time. A 50-line wall of code teaches nothing. Instead, build up incrementally:

1. Show a small, focused snippet (**10-25 lines max** — hard limit, break up anything longer)
2. Explain what it does and why — connect the code to the concept
3. Then show the next piece

If you have a class with multiple methods, show the constructor first, explain it, then show each method separately. The reader should never have to scroll through a code block.

Use fenced code blocks with language identifiers:

~~~markdown
```python
# Real, runnable code — not pseudocode
```
~~~

~~~markdown
```bash
# Actual commands that work
```
~~~

### Math

Use KaTeX syntax. Display math for key equations:
```
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
```

Inline math for referencing variables and values: `$\alpha = 2 \times 10^{-5}$`

Always connect math to code — if you show a formula, show the implementation.

### Tables

Use markdown tables for comparisons and benchmarks:
```markdown
| Method | Params | Memory | Loss |
|--------|--------|--------|------|
| Full   | 6.7B   | 320GB  | 0.82 |
| LoRA   | 6.5M   | 40GB   | 0.85 |
```

## Generating & Verifying Code

This is critical — every code snippet in the post must actually work. Broken code examples destroy credibility.

### Execution Order

Follow this exact order to avoid getting stuck:

1. **Write the blog post first** — get the full `.md` file written to `_posts/`
2. **Write the companion notebook** — create the `.ipynb` file in `notebooks/`
3. **Then verify code** — run snippets, execute the notebook, rebuild the site

Writing all files first ensures deliverables exist even if verification hits issues.

### Verification Process

1. Install any needed packages: `pip install torch numpy` (or whatever the post needs)
2. Run each code snippet via Bash to verify it executes without errors
3. If a snippet requires GPU or large models that can't be run locally, verify the logic with a smaller-scale CPU test and note GPU requirements in a comment
4. For code that uses external APIs or model downloads, write a mock/stub version that demonstrates the logic without network calls

### Companion Notebook

For any post that contains code, generate a complete Jupyter notebook (`.ipynb`). Write it using the Write tool as a valid JSON file.

The notebook should be:

1. **Self-contained** — includes `!pip install` cells at the top for all dependencies
2. **Structured like the post** — markdown cells mirror the post's sections, code cells contain the snippets
3. **Runnable end-to-end** — a reader should be able to open it in Colab and click "Run All"
4. **Annotated** — markdown cells explain what each code cell does

Save the notebook to `/Users/cmenguy/Git/cmenguy.github.io/notebooks/YYYY-MM-DD-slug.ipynb`

After writing the notebook file, try to execute it:
```bash
pip install jupyter nbconvert
jupyter nbconvert --to notebook --execute NOTEBOOK_PATH --output NOTEBOOK_PATH
```

If execution fails due to missing GPU or large model downloads, that's OK — just make sure the notebook is syntactically valid and the CPU-compatible cells work.

Update the post's front matter with the notebook paths.

## File Locations

- Blog root: `/Users/cmenguy/Git/cmenguy.github.io/`
- Posts: `/Users/cmenguy/Git/cmenguy.github.io/_posts/`
- Notebooks: `/Users/cmenguy/Git/cmenguy.github.io/notebooks/`
- Use today's date for the post filename and front matter date

## Checklist Before Finishing

- [ ] Front matter is complete with all required fields
- [ ] Filename follows `YYYY-MM-DD-slug.md` format
- [ ] Opening hooks the reader in 2-3 sentences
- [ ] Sections are logically organized with descriptive headers
- [ ] Code blocks are bite-sized and each one is explained
- [ ] All code has been executed and verified to work
- [ ] Math formulas are correct KaTeX syntax
- [ ] Companion notebook is generated and runs end-to-end
- [ ] Post front matter references the notebook (if applicable)
- [ ] No filler paragraphs — every sentence earns its place
- [ ] Rebuild the site: `PATH="/opt/homebrew/opt/ruby/bin:/opt/homebrew/lib/ruby/gems/4.0.0/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH" bundle exec jekyll build` from the blog root
