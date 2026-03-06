# The Skill is the Unit of Reuse

---

**Hands-on examples of building, loading, and evaluating AI agent skills across Claude Code, Mistral Vibe, and the Claude API.**

[Run in Google Colab](https://colab.research.google.com/github/cmenguy/cmenguy.github.io/blob/main/notebooks/2026-02-05-skills-deep-dive.ipynb) | [View on GitHub](https://github.com/cmenguy/cmenguy.github.io/blob/main/notebooks/2026-02-05-skills-deep-dive.ipynb)

---

In the [last post](https://ai-terminal.net/llm/ai-engineering/agents/2026/01/28/agentic-protocols/) I talked about A2A, MCP, and where multi-agent systems break down. I ended with a teaser about the "single agent with skills" pattern being the sweet spot most teams should reach for. This post is the follow-through on that promise.

At work, we'd been running a multi-agent setup loosely inspired by A2A. Multiple agents, each with their own context, coordinating over internal APIs. It worked, mostly. But every time we wanted to add a new capability (a new type of analysis, a new data source) it meant spinning up a new service, defining new endpoints, and dealing with exactly the context fragmentation problems I'd been writing about.

Someone on the team brought up skills. Not as a vague concept, but as a concrete pattern: what if instead of a new agent, we wrote a markdown file? A file that tells the existing agent *how* to do the new thing, what tools it needs, and when to activate. I was skeptical at first. A markdown file replacing a microservice felt like bringing a napkin to a knife fight. But I tried it, and within a few days I had a pretty complex agent rebuilt as a skill. It worked about 90% as well as the dedicated service and was already a lot more flexible. That got my attention.

## 1. Why Not Just Context Engineering?

Before we get into skills, let's address the elephant: why can't you just stuff everything into the system prompt?

You can, and many teams do. Here's what that looks like in practice:


https://gist.github.com/cmenguy/533e4e828c6ce28e23dd66befd6c69a1


This works at three capabilities. At fifteen, you start noticing the model sometimes confuses which tool sequence to use. At thirty, you're managing a 10,000-token system prompt where changing one capability risks breaking another. And testing? You're testing the entire system prompt every time you touch a single capability.

The failure mode is subtle. The model doesn't crash, it just starts blending instructions. It'll use the billing escalation flow for a technical issue, or skip a step in onboarding because the adjacent capability's instructions bled in. I've seen this happen in production, and it's the worst kind of bug because the output looks plausible.

Skills solve this by **scoping instructions to a single capability** and **loading only what's relevant**. Instead of a 10K-token system prompt where everything is always present, you have focused instruction sets that get loaded on demand. The model sees the billing skill *or* the technical support skill, never a soup of both.

## 2. Anatomy of a Skill

A skill is a folder. At minimum it contains one file: `SKILL.md`. That's it. Here's the structure:


https://gist.github.com/cmenguy/4e09f901a66d5786b5d2c26112fd4621


The `SKILL.md` has two parts: frontmatter (metadata) and body (instructions). Let me walk through a real example. I'll use a "deploy review" skill, something that reviews a deployment plan before pushing to production. This is a skill I actually built at work, simplified for this post.

#### 2.1 The Frontmatter


https://gist.github.com/cmenguy/3f749c331f44cc733622f5e80d89c65c


Each field matters:

- **`name`**: Lowercase, hyphens only. This becomes the `/deploy-review` slash command.
- **`description`**: This is the skill's résumé. The agent reads this to decide whether to activate the skill. Good descriptions include *trigger words*, the phrases from the user that should activate this skill.
- **`user-invocable`**: Can a human type `/deploy-review` to trigger it? Set to `false` for background skills that only the agent should activate.
- **`allowed-tools`**: What tools the skill can use. This is a security boundary. A documentation skill shouldn't need `Bash`, and a deploy review skill shouldn't need `Write`.

#### 2.2 The Body

The body is markdown instructions. Here's a simplified version:


https://gist.github.com/cmenguy/e7333542fd006d838d98dc95e0251359


Notice what this is: *scoped instructions for a specific task*. Not a general system prompt trying to do everything. The model loads this when it needs to review a deployment, follows the steps, and produces the expected output format. When it's doing something else, like writing code or answering questions, these instructions aren't in context and can't interfere.

#### 2.3 Progressive Disclosure

Here's a design principle that took me a while to appreciate: **the agent doesn't load the full skill upfront**. It works in two phases:

**Phase 1 — Discovery (~100 tokens):** The agent reads the `name` and `description` of every available skill. This is cheap. For 20 skills, that's ~2,000 tokens of metadata.

**Phase 2 — Activation (<5,000 tokens):** When the agent decides a skill is relevant (based on the description matching the user's request), it loads the full `SKILL.md`. This is the full instruction set, but only for the activated skill.


https://gist.github.com/cmenguy/f61870ae694e4a7e6748fc2fde8b3857


This is why the description matters so much. A bad description means the skill never gets activated:


https://gist.github.com/cmenguy/c8de433359eb859c0a5019159e3d1491


#### 2.4 Supporting Files

For complex skills, the `SKILL.md` stays focused and references separate files for the heavy stuff:


https://gist.github.com/cmenguy/a05eda110ffa46f81378cb50a9e39fd8


The agent loads `RUNBOOK.md` only when it needs the detailed steps, not every time the skill activates. This keeps the token budget tight and avoids drowning the model in irrelevant detail.

## 3. Skills Across Agentic Harnesses

The [Agent Skills specification](https://agentskills.io) is an open standard, which means the same `SKILL.md` file works across multiple tools. In practice, there are quirks. Let me walk through how skills integrate with three major agentic harnesses, using our deploy-review skill as the running example.

#### 3.1 Claude Code

Claude Code has the most mature skills implementation. Skills live in a few places, in priority order:


https://gist.github.com/cmenguy/68965cb7e76fe9fbeca158be87bd9b55


Project-level skills are the most common. They travel with the repo, so your whole team gets them via `git pull`.

Claude Code adds some extensions beyond the base spec:


https://gist.github.com/cmenguy/b1273059fb0b33bf8d8edf4871968128


**`context: fork`** is particularly useful. It runs the skill in a subagent with its own context, so a long skill execution doesn't pollute the main conversation. The tradeoff is that the subagent can't see prior conversation history.

**Dynamic context injection** is another Claude Code extension. You can execute shell commands at skill load time:


https://gist.github.com/cmenguy/ba97b5eff7454c41db99e8bf02363db5


The `` !`command` `` syntax runs the command and injects its output into the skill context before the model sees it. So the model gets live infrastructure state without needing to run those commands itself.

#### 3.2 Mistral Vibe

Mistral's Vibe agent implements the same Agent Skills spec. Skills go in:


https://gist.github.com/cmenguy/67eee1e6d01dbfcfbbe40d01884e23db


The frontmatter and body format are identical. Your `SKILL.md` from Claude Code works in Vibe without changes. Vibe adds its own agent system on top:


https://gist.github.com/cmenguy/e236e262a465ea2eea144653e022296f


In Vibe, you can define custom agents (separate from skills) that reference skills as part of their behavior. The distinction is: a **skill** is a reusable capability, an **agent** is a persona with specific tools and approval settings that might *use* skills.

#### 3.3 OpenAI Codex

Here's where it gets interesting. Codex doesn't have a formal skills system as of early 2026. Instead, it uses two mechanisms that approximate skills:

**AGENTS.md files** serve a similar purpose to skills. They're markdown instructions that configure agent behavior:


https://gist.github.com/cmenguy/d5df36f82a83eadab4f181025221f7ac


The key difference: AGENTS.md is monolithic. All instructions live in one file (or a few files), always loaded. There's no progressive disclosure, no activation-based loading, no isolated context. It's closer to the "stuff it in the system prompt" approach, just organized as a file.

**Custom instructions** in Codex's settings are the other mechanism: persistent behavioral instructions. But again, they're always on, not loaded on-demand.


https://gist.github.com/cmenguy/f75d16561e23de7d6dacf761f523c9d9


This doesn't mean Codex is worse. It's a different design philosophy. Codex optimizes for simplicity (one file, always loaded), while Claude Code and Vibe optimize for modularity (many skills, loaded on demand). For small projects with 2-3 capabilities, the Codex approach is simpler. For large projects with dozens of capabilities, the skill-based approach scales better.

## 4. Skills via the Claude API

Skills in Claude Code and Vibe are great for interactive development. But what if you're building a backend service (a customer support bot, an internal tool, a CI/CD pipeline) that needs skill-like behavior? You need skills at the API level.

The Claude API supports this through two mechanisms: the **Agent SDK** and the **container-based skills API**.

#### 4.1 The Agent SDK Approach

The Claude Agent SDK lets you load skills from the filesystem, just like Claude Code does:


https://gist.github.com/cmenguy/14dbc553a000fadb6c904e01b0fa93a3


This reads your `SKILL.md` files from disk, handles the progressive disclosure, and manages the tool loop automatically. The agent decides which skill to activate based on the prompt, just like the interactive experience.

You can also define skills programmatically, which is useful when your skills come from a database or API rather than a filesystem:


https://gist.github.com/cmenguy/c941c37cd9acff0cdad8477382892908


#### 4.2 The Container-Based Approach

For skills that need to produce files (generate reports, create spreadsheets, build presentations), the Claude API supports a container model where skills execute in a sandboxed environment with filesystem access:


https://gist.github.com/cmenguy/351879e29ccc1ceae0c22260ae8d2a22


The `container` parameter spins up an execution environment on Anthropic's servers. The skill (`xlsx` in this case) gives the model the knowledge and tools to create Excel files. The model writes files inside the container, and you get back `file_id` references you can download.

What makes this powerful for custom skills is the pattern: **read a skill definition, inject it as context, give the model filesystem access in a container, let it execute**. You can build your own skill runner:


https://gist.github.com/cmenguy/fcb74114285991863758718ad4ef8c27


This is a minimal skill runner. It loads a `SKILL.md` as the system prompt and runs the user's request against it. In production, you'd add tool result handling (the response might contain `tool_use` blocks that need execution), multi-turn conversation support, and error handling. But the core pattern is this simple: **skill instructions become system prompt, user intent becomes the message**.

#### 4.3 Building a Multi-Skill Router

Here's the pattern I've converged on for services that need multiple skills: a router that mimics the progressive disclosure from Claude Code.


https://gist.github.com/cmenguy/ad2a0cb52ca4e76da3aee8de740d283f



https://gist.github.com/cmenguy/2a517a517465b481c8fb126a45291762



https://gist.github.com/cmenguy/958b3ee0b0bab3f0a448559462a37458


Two things to notice. First, routing uses a cheap, fast model (Haiku) because it's just matching text to a description. You don't need Opus for that. Second, only the matched skill's full instructions get loaded for execution. This is progressive disclosure implemented at the API level: Phase 1 costs ~2K tokens across all skills, Phase 2 costs ~5K tokens for the one skill that fires.

## 5. Best Practices: What I've Learned

After building about a dozen skills across different projects, here's what actually matters:

#### 5.1 One Skill, One Job

The biggest mistake is building a Swiss Army knife skill. A skill called `code-helper` that handles code review, test generation, refactoring, and documentation will be mediocre at all four. Split it into four skills. The progressive disclosure system means there's no cost to having many small skills, since only the active one loads.

#### 5.2 Description Is Everything

The description field is the most important part of the frontmatter. It's the search index for your skill. I've seen skills that do exactly the right thing but never activate because their description doesn't include the words a user would actually say.


https://gist.github.com/cmenguy/c4137746d16ba94123973f56e3ac9396


#### 5.3 Keep SKILL.md Under 500 Lines

If your skill instructions exceed 500 lines, you're doing too much in one skill. Split it, or move detailed reference material into `references/` files that get loaded on demand.

#### 5.4 Define the Output Format

Skills that specify their output format get dramatically better results. Don't leave it to the model's judgment. Tell it what the output looks like:


https://gist.github.com/cmenguy/a3a0547bc0aa84ea64e2a8c4c5a93a13


#### 5.5 Test with the Description, Not Just the Skill

Most people test whether a skill works *after* it activates. That's only half the problem. You also need to test whether it activates *when it should* and *doesn't* activate when it shouldn't. I'll come back to this in the evaluation section.

## 6. The Honest Take: Markdown All the Way Down

Here's where I get into opinion territory.

**I think skills are the future of how we compose AI capabilities.** The reasons are almost boringly practical: they're just files. They version control with `git`. They diff, they merge, they code review. A junior engineer can read a `SKILL.md` and understand what the AI is going to do — try doing that with a fine-tuned model or a 500-line Python orchestration class. The composability is natural: add a skill by dropping a folder, remove it by deleting the folder, share it by copying it to another project.

The convergence is real. Claude Code and Mistral Vibe both implement the same Agent Skills spec. Other tools (Cursor, OpenHands, Goose) are adopting it. This feels like the early days of REST APIs: not because anyone mandated it, but because the pattern is obvious once you see it.

**And yet, I'm genuinely unsettled by how well this works.** Let me explain. A skill is a markdown file. Not code. Not a formal specification. Not a schema. Markdown. With headings and bullet points and plain English instructions. And the model *follows it*. Not perfectly, but well enough that I'm shipping production systems where the "business logic" is a markdown file that says things like "if the user seems frustrated, escalate to a human."

This shouldn't work as well as it does. I spent years building systems with strict schemas, type checking, validation layers. All the machinery you need because software is brittle and does exactly what you tell it, not what you mean. Skills flip this on its head: you write what you mean in natural language, and the model figures out the "what you tell it" part. It's powerful and it makes me uncomfortable in equal measure.

**Where I'm genuinely unsure: complex multi-step skills.** Our deploy-review skill is ~50 lines of markdown and works great. But I've tried building skills that are essentially runbooks: 200+ lines of conditional logic, branching paths, error handling. At that complexity, the model starts dropping steps, confusing sequences, and making creative interpretations of instructions that were meant to be followed literally.

There seems to be a ceiling, somewhere around 100-150 lines of instruction, where the model transitions from "reliably following the recipe" to "improvising on a theme." Below that ceiling, skills are magic. Above it, they're a gamble. I don't know exactly where that ceiling is, and I suspect it varies by model, by task complexity, and by how well the instructions are written. But it exists, and anyone building complex skills will hit it.

## 7. The Missing Piece: Skill-First Evaluation

This brings me to what I think is the most important unsolved problem in the skills ecosystem: **how do you test a skill?**

Right now, most teams test skills manually. They activate the skill, run it against a few inputs, eyeball the output, and call it good. This is the "we tested in prod" of the skills world, and it scales exactly as well as that phrase implies.

What we need is skill-first evaluation, automated tests that verify:

1. **Activation accuracy**: Does the skill fire for the right inputs and stay silent for the wrong ones?
2. **Instruction adherence**: Does the model follow the steps in order? Does it produce the expected output format?
3. **Edge case handling**: What happens with ambiguous inputs? Empty inputs? Adversarial inputs?

This is starting to emerge. The Agent Skills spec includes an `evals.json` concept, a file that lives alongside `SKILL.md` and defines test cases:


https://gist.github.com/cmenguy/f74f16d626e29616735bf88fb04a64cb


This is promising but early. A few things I'd want to see mature:

**Activation precision/recall metrics.** Not just "did it fire" but "across 100 diverse phrasings of deploy-related requests, how often does it fire?" and "across 100 non-deploy requests, how often does it incorrectly fire?" This is a classic information retrieval problem, and we have good frameworks for measuring it.

**Multi-step verification.** For complex skills, checking the final output isn't enough. You need to verify intermediate steps. Did it actually run the test suite before approving, or did it skip to the approval? This requires either tool call tracing or structured intermediate outputs.

**Regression testing across model updates.** When Sonnet 4.7 drops, does your skill still work? Model updates can change how instructions are interpreted. Skills need regression tests the same way code needs them. Run the eval suite before and after, compare results.


https://gist.github.com/cmenguy/17a26dc326ebfd250a59dc4ad014b99d


This is a sketch, not production code. But the shape of the solution is clear: treat skill evaluation like information retrieval evaluation. Measure precision and recall for activation. Measure adherence for execution. Run it automatically, track it over time, flag regressions.

## 8. What's Next

Skills are the single-agent answer to the multi-agent question. For most teams, most of the time, a well-designed set of skills on a single capable model will outperform a constellation of specialized agents. Less complexity, less latency, better debugging, and critically, the ability for anyone on the team to read the skill file and understand what the AI is supposed to do.

The tooling is converging, the spec is open, and the pattern works. What's missing is the evaluation layer, and I suspect that's where the next wave of investment will go. Right now we're in the "vibes-based testing" era of skills. The teams that move to rigorous, automated skill evaluation first will ship more reliably and iterate faster. The `evals.json` pattern is the seed of that future, and I'm watching it closely.


---

*Originally published on [AI Terminal](https://ai-terminal.net/llm/ai-engineering/agents/2026/02/05/skills-deep-dive/).*

Tags: codex, skills, agents, evaluation, claude-code
