---
layout: post
title: "SFT, DPO, RLHF — Picking the Right Fine-Tuning Strategy for Your LLM"
date: 2025-12-22 10:00:00 -0800
categories: [llm, fine-tuning]
tags: [sft, dpo, rlhf, trl, marketing, preference-learning]
author: cmenguy
colab_url: "https://colab.research.google.com/github/cmenguy/notebooks/blob/main/2025-12-22-fine-tuning-llms-sft-dpo-rlhf.ipynb"
colab_embed: false
github_notebook: "https://github.com/cmenguy/notebooks/blob/main/2025-12-22-fine-tuning-llms-sft-dpo-rlhf.ipynb"
notebook_description: "End-to-end code for SFT, DPO, and RLHF fine-tuning with synthetic marketing data."
---

I've been spending a lot of time with Llama models at work lately — mostly prototyping, seeing where the edges are, getting a feel for what a mid-size open model can actually do in production. A few weeks ago, someone on the marketing team asked if we could get the model to write copy that sounds less like a robot and more like *us*. The kind of brand-specific voice that takes a human copywriter months to internalize. That question got stuck in my head.

Then December hit and things started winding down at work — the way they always do when half the company is already mentally on PTO. I flew out to France to spend the holidays with my parents, and somewhere between the jet lag and my mom's cooking, I found myself with actual uninterrupted thinking time for the first time in months. Long walks through the village, bad Wi-Fi that kept me off Slack, and a laptop I'd told myself I wouldn't open but obviously did. What follows is the result of that Christmas week exploration — fueled less by coffee and more by vin chaud.

## Why Not Just Context Engineering?

The first thing I tried — and probably what you'd try too — was cramming everything into the prompt. System message with brand guidelines, a few examples of approved copy via few-shot, maybe RAG over a library of past campaigns. No training, no GPUs, ship it by Friday.

And honestly? For a single brand with a straightforward voice, this works fine. Here's the kind of prompt I was playing with:

```text
You are a copywriter for Acme Outdoor Gear. Brand voice: rugged,
no-nonsense, subtly humorous. Never use exclamation marks. Prefer
short sentences. Reference the outdoors without being cliché.

Examples of approved copy:
- "Rain doesn't cancel adventures. Neither does this jacket."
- "Built for the trail. Tolerated at the office."
- "Your gear should work harder than you do."

Write an email subject line for our winter boot launch.
```

Llama 3.1 8B does a decent job with this. You get something like "Cold feet aren't part of the plan." — not bad, sounds roughly on-brand. So why bother fine-tuning?

Three reasons kept pushing me past the prompt-engineering ceiling:

**Context window budget.** When you're serving 50 different brands, each with its own voice guidelines and example library, the prompt gets heavy fast. A 2,000-token brand context block times 50 brands means every request is burning tokens on instructions instead of thinking. Fine-tuning bakes the knowledge into the weights — the prompt stays lean, inference is cheaper, and latency drops.

**Consistency at the margins.** Prompting gets you 80% of the way there, but the last 20% is where brand voice actually lives. The subtle things — whether the brand uses Oxford commas, whether contractions are OK, whether humor is dry or playful — those nuances drift across generations with prompting alone. I ran 100 generations with the same prompt and kept seeing the model slip into its default "helpful assistant" voice on maybe 15-20% of them. After SFT on 500 brand-specific examples, that drop rate went to near zero.

**Preference capture.** This is the big one. Prompting can tell a model *what* to write, but it can't teach it *what's better*. In marketing, you have A/B test data — version A got 3x the click-through rate of version B. That signal is gold, and there's no way to feed it into a prompt. You need DPO or RLHF to actually learn from preference pairs, and that requires training.

So the rule of thumb I landed on: if you have one brand and simple requirements, prompt engineering is the right call. If you have many brands, need tight consistency, or have preference data you want to exploit — that's when fine-tuning starts paying for itself.

## The Landscape

Here's the scenario I keep coming back to. You're building an AI-powered marketing platform. Customers upload their brand guidelines, past campaigns, and tone-of-voice docs. They want the LLM to generate email subject lines, ad copy, social media posts, and product descriptions that are indistinguishable from what their in-house team would write.

This is a preference-heavy domain. "Good" marketing copy isn't just grammatically correct — it's persuasive, on-brand, and sometimes deliberately breaks rules for effect. That matters a lot for which fine-tuning approach you choose.

Here's the quick mental model before we go deep:

| Method | What It Learns From | Core Idea | Complexity |
|--------|-------------------|-----------|------------|
| SFT | Gold-standard examples | "Write like this" | Low |
| RLHF | Human preference rankings + reward model | "A human liked this better" | High |
| DPO | Preference pairs directly | "This is better than that" | Medium |

## Supervised Fine-Tuning (SFT) — Teaching by Example

SFT is the workhorse. You show the model examples of what good output looks like, and it learns to imitate them. If you've ever fine-tuned a model on a dataset of prompt-completion pairs, you've done SFT.

### The Math

SFT minimizes the standard language modeling cross-entropy loss over your curated dataset. Given a dataset of $(x, y)$ pairs where $x$ is the prompt and $y = (y_1, y_2, \ldots, y_T)$ is the target completion:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \sum_{t=1}^{T} \log \pi_\theta(y_t \mid x, y_{\lt t}) \right]$$

Where $\pi_\theta$ is your model parameterized by $\theta$. This is just next-token prediction on your curated data — nothing fancy, and that's the point.

### When SFT Makes Sense

SFT is your go-to when you have clean, high-quality examples of the output you want. For marketing, this means curated campaign copy that's already been approved by the brand team. Think of it as cloning your best copywriter's style.

The catch: SFT can only learn from positive examples. It knows what "good" looks like, but it has no concept of "this is better than that." If your training data has any noise — mediocre examples mixed in with great ones — the model learns to reproduce the average, not the best.

### The Code

Let's fine-tune a model on marketing copy using TRL's `SFTTrainer`. First, the data prep — structuring brand-specific examples as conversations:

```python
from datasets import Dataset

# Your curated marketing examples as conversations
raw_examples = [
    {
        "messages": [
            {"role": "system", "content": "You are a marketing copywriter for a luxury skincare brand. Tone: sophisticated, confident, never salesy."},
            {"role": "user", "content": "Write an email subject line for our new retinol serum launch."},
            {"role": "assistant", "content": "Your skin has been waiting for this."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a marketing copywriter for a luxury skincare brand. Tone: sophisticated, confident, never salesy."},
            {"role": "user", "content": "Write a 2-sentence product description for our vitamin C moisturizer."},
            {"role": "assistant", "content": "Radiance isn't something you chase — it's something you build. Our vitamin C complex works at the cellular level, so the glow you see is the glow you own."},
        ]
    },
]

sft_dataset = Dataset.from_list(raw_examples)
```

Now the training setup — LoRA keeps this feasible on a single GPU:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
```

And the trainer — note the `max_length` tuned for short-form marketing copy:

```python
training_args = SFTConfig(
    output_dir="./sft-marketing",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_length=512,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=sft_dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

trainer.train()
```

That's it. With 500-1000 curated examples per brand, you'll get a model that captures tone and style surprisingly well. SFT is the 80/20 of fine-tuning — it gets you most of the way there with the least effort.

## RLHF — The Full Pipeline

Reinforcement Learning from Human Feedback is the approach that made ChatGPT feel different from GPT-3. The idea: train a separate reward model on human preferences, then use it to guide the LLM's behavior via reinforcement learning.

### The Math

RLHF has three stages, each with its own objective.

**Stage 1: SFT** — same as above. You start with a supervised fine-tuned model $\pi_{\text{SFT}}$ as your starting point.

**Stage 2: Reward Model Training.** Given pairs of responses $(y_w, y_l)$ where a human preferred $y_w$ over $y_l$, train a reward model $r_\phi$ using the Bradley-Terry preference model:

$$\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right) \right]$$

Where $\sigma$ is the sigmoid function. The reward model learns to assign higher scores to preferred responses.

**Stage 3: RL Optimization.** Maximize the reward while staying close to the SFT policy via a KL penalty:

$$\mathcal{L}_{\text{RLHF}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot \mid x)} \left[ r_\phi(x, y) - \beta \, D_{\text{KL}}\left(\pi_\theta(\cdot \mid x) \| \pi_{\text{SFT}}(\cdot \mid x)\right) \right]$$

The classic approach used PPO (Proximal Policy Optimization) for this stage, but modern implementations have moved on. TRL now uses RLOO (REINFORCE Leave-One-Out), which is simpler and more stable than PPO while achieving comparable results. The KL term remains critical — without it, the model collapses to reward hacking, generating degenerate text that scores high on the reward model but is gibberish to humans.

### When RLHF Makes Sense

RLHF shines when preferences are nuanced and hard to capture with static examples alone. It's the gold standard for aligning models to complex human judgment — which is exactly what "good marketing copy" requires.

The problem: it's expensive. You need to train and maintain a separate reward model. The RL optimization stage adds instability — loss curves that look like seismographs are par for the course. And you need a *lot* of preference data to train a reward model that doesn't just memorize your training set.

For a marketing use case, RLHF makes sense at scale — when you have thousands of A/B tested campaigns and can build a reward model that genuinely captures what makes copy convert. For most teams, this is overkill.

### The Code

Let's build the full pipeline. First, the reward model — trained on pairs where one version of copy outperformed another:

```python
from datasets import Dataset

# Preference data: which copy performed better in A/B tests
preference_data = [
    {
        "prompt": "Write a flash sale email subject line for 30% off sneakers.",
        "chosen": "Your favorites. 30% off. Today only.",
        "rejected": "Big Sale Alert! Get 30% Off All Sneakers Now!!!",
    },
    {
        "prompt": "Write a push notification for a loyalty reward.",
        "chosen": "You've unlocked something. Check your rewards.",
        "rejected": "Congratulations! You have earned a new loyalty reward! Tap to claim.",
    },
]

reward_dataset = Dataset.from_list(preference_data)
```

Training the reward model using TRL's `RewardTrainer`:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

reward_model_name = "meta-llama/Llama-3.1-8B-Instruct"
reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
reward_tokenizer.pad_token = reward_tokenizer.eos_token

reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_name, num_labels=1, torch_dtype="auto", device_map="auto"
)
```

```python
reward_training_args = RewardConfig(
    output_dir="./reward-model-marketing",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    learning_rate=1e-5,
    max_length=512,
)

reward_trainer = RewardTrainer(
    model=reward_model,
    args=reward_training_args,
    processing_class=reward_tokenizer,
    train_dataset=reward_dataset,
)

reward_trainer.train()
```

Now the RL stage — this is where the model learns to generate copy that scores high on the reward model. TRL's `RLOOTrainer` (REINFORCE Leave-One-Out) handles generation, scoring, and policy updates in a unified loop. First, prepare the prompt dataset:

```python
from datasets import Dataset

prompts_data = [
    {"prompt": "Write a subject line for a summer collection launch."},
    {"prompt": "Write a 1-sentence Instagram caption for a new coffee blend."},
    {"prompt": "Write a CTA button label for a free trial signup."},
    {"prompt": "Write a 1-sentence product teaser for wireless earbuds."},
]

prompt_dataset = Dataset.from_list(prompts_data)
```

Configure and run the RLOO trainer — it generates multiple completions per prompt and uses the leave-one-out baseline to reduce variance:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import RLOOConfig, RLOOTrainer

rl_model = AutoModelForCausalLM.from_pretrained(
    "./sft-marketing", torch_dtype="auto", device_map="auto"
)
rl_tokenizer = AutoTokenizer.from_pretrained(model_name)
rl_tokenizer.pad_token = rl_tokenizer.eos_token

rloo_config = RLOOConfig(
    output_dir="./rloo-marketing",
    per_device_train_batch_size=4,
    num_generations=4,           # completions per prompt
    max_completion_length=64,
    learning_rate=1e-6,
    num_train_epochs=1,
    logging_steps=1,
    beta=0.05,                   # KL penalty coefficient
)

rloo_trainer = RLOOTrainer(
    config=rloo_config,
    model=rl_model,
    reward_model=reward_model,
    processing_class=rl_tokenizer,
    reward_processing_class=reward_tokenizer,
    train_dataset=prompt_dataset,
)

rloo_trainer.train()
```

That's still a lot of moving parts — two models, generation during training, and hyperparameters that interact in non-obvious ways. RLOO is more stable than the old PPO approach, but the fundamental complexity of reward-model-guided RL remains. This is why DPO exists.

## DPO — The Pragmatist's Choice

Direct Preference Optimization is the "what if we just... skipped the reward model?" approach. Published in 2023, DPO showed that you can optimize directly on preference pairs without ever training a separate reward model or running RL. It's mathematically equivalent to RLHF under certain assumptions, but dramatically simpler to implement.

### The Math

The key insight of DPO: the optimal policy under the RLHF objective has a closed-form relationship with the reward function. Instead of learning $r_\phi$ and then optimizing $\pi_\theta$ against it, DPO reparameterizes the reward in terms of the policy itself:

$$r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

Where $\pi_{\text{ref}}$ is the reference policy (your SFT model) and $Z(x)$ is a partition function that cancels out in the preference comparison. Substituting into the Bradley-Terry model gives the DPO loss:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

Read it like this: push up the probability of the preferred response $y_w$ relative to the reference model, while pushing down the probability of the rejected response $y_l$, all modulated by $\beta$.

### When DPO Makes Sense

DPO is the sweet spot for most production fine-tuning. You get preference learning without the infrastructure complexity of RLHF. No reward model to train and serve. No PPO instability. Just preference pairs and a single training run.

For marketing, this is particularly appealing. You likely have A/B test results, editorial feedback, and brand compliance reviews — all natural sources of "this version is better than that version" data. DPO consumes this directly.

### The Code

DPO needs an SFT model as the starting point (the reference policy) and a dataset of preference pairs. The data is the same format we used for the reward model:

```python
from datasets import Dataset

dpo_data = [
    {
        "prompt": "Write a homepage hero headline for an eco-friendly cleaning brand.",
        "chosen": "Clean home. Clean conscience.",
        "rejected": "Our All-Natural Cleaning Products Are Good for You and the Planet!",
    },
    {
        "prompt": "Write a cart abandonment email subject line.",
        "chosen": "Still thinking it over?",
        "rejected": "You Left Items in Your Cart! Complete Your Purchase Now!",
    },
    {
        "prompt": "Write a loyalty program welcome message.",
        "chosen": "You're in. Here's what that means.",
        "rejected": "Welcome to Our Amazing Rewards Program! Start Earning Points Today!",
    },
]

dpo_dataset = Dataset.from_list(dpo_data)
```

The training setup is refreshingly simple — one model, one trainer, one loss function:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
```

```python
dpo_training_args = DPOConfig(
    output_dir="./dpo-marketing",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    beta=0.1,  # controls KL penalty strength
    max_length=512,
    logging_steps=10,
)

dpo_trainer = DPOTrainer(
    model=model,
    args=dpo_training_args,
    train_dataset=dpo_dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

dpo_trainer.train()
```

Compare this to the RLHF pipeline above. Same preference data, same end goal, a fraction of the complexity.

## The Comparison — What Actually Works for Marketing

Let's put these side by side for the marketing fine-tuning scenario:

| Dimension | SFT | RLHF | DPO |
|-----------|-----|------|-----|
| **Data needed** | Curated examples (500-1k) | Preference pairs (5k+) + SFT data | Preference pairs (1k+) + SFT data |
| **Training complexity** | Single training run | 3-stage pipeline | 2-stage pipeline (SFT then DPO) |
| **GPU memory** | 1 model | 3 models simultaneously | 2 models (policy + reference) |
| **Stability** | Very stable | PPO can diverge | Stable |
| **Captures preferences** | No — only positive examples | Yes — full preference modeling | Yes — direct preference optimization |
| **Time to production** | Days | Weeks | Days |
| **When to use** | You have gold examples, preferences don't matter | Scale with rich preference signal | You have preference data, want simplicity |

### My Recommendation for Marketing

**Start with SFT. Graduate to DPO. Reach for RLHF only if you have to.**

Here's why, stage by stage:

**Phase 1 — SFT to capture brand voice.** Collect 500-1000 examples of approved marketing copy per customer. Email subjects, product descriptions, social posts — all tagged with the prompt that generated them. Fine-tune with LoRA. This alone gets you 80% of the way to sounding like the brand.

**Phase 2 — DPO to learn preferences.** Once you have SFT running, start collecting preference data. This is natural in marketing: A/B test results (version A got 3x clicks), editorial reviews (the brand team picked option 2), compliance flags (this version violated guidelines). Feed these as preference pairs into DPO. The model learns not just *what* good copy looks like, but *why* one version is better than another.

**Phase 3 — RLHF only at scale.** If you're running millions of campaigns across hundreds of brands and have a genuine signal for what converts, a dedicated reward model starts to make sense. It can generalize across brands and capture patterns that individual preference pairs can't. But you need the data volume and the engineering team to support it.

For most teams starting out, SFT + DPO is the right answer. You get preference learning, training stability, and a pipeline that a single ML engineer can own. RLHF is a great system when you grow into it — it's a terrible starting point.

## The Hidden Variable — Data Quality

None of this matters if your data is garbage. A few hard-won lessons:

**For SFT data:** only include examples you'd be proud to show a client. One mediocre example teaches the model more bad habits than ten good examples can fix. Have your best copywriter review every training example.

**For preference pairs:** the margin matters. "Slightly better" pairs teach less than "clearly better" pairs. When collecting A/B test data, filter for cases where one version significantly outperformed the other. A 51/49 click split is noise, not signal.

**For both:** diversity beats volume. 500 examples across 10 different copy formats (emails, ads, social, product descriptions) will outperform 2000 examples of just email subject lines. The model needs to learn the brand voice, not just one template.

## Wrapping Up

SFT teaches your model what good looks like. DPO teaches it what *better* looks like. RLHF builds a full preference engine. For customer-specific marketing fine-tuning, the SFT-then-DPO pipeline gives you the best ratio of capability to complexity. Start there, measure what you're getting, and scale up the sophistication only when the data and the business justify it.

The code in this post (and the companion notebook) uses TRL throughout — it's become the de facto library for all three approaches, and keeping your entire fine-tuning stack in one framework means less glue code and fewer surprises when you upgrade.
