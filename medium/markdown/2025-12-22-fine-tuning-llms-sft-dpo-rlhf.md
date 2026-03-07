# SFT, DPO, RLHF: Picking the Right Fine-Tuning Strategy for Your LLM

---

**End-to-end code for SFT, DPO, and RLHF fine-tuning with synthetic marketing data.**

[Run in Google Colab](https://colab.research.google.com/github/cmenguy/cmenguy.github.io/blob/main/notebooks/2025-12-22-fine-tuning-llms-sft-dpo-rlhf.ipynb) | [View on GitHub](https://github.com/cmenguy/cmenguy.github.io/blob/main/notebooks/2025-12-22-fine-tuning-llms-sft-dpo-rlhf.ipynb)

---

I've been spending a lot of time with Llama models at work lately, mostly prototyping, seeing where the edges are, getting a feel for what a mid-size open model can actually do in production. A few weeks ago, someone on the marketing team asked if we could get the model to write copy that sounds less like a robot and more like *us*. The kind of brand-specific voice that takes a human copywriter months to internalize. That question got stuck in my head.

Then December hit and things started winding down at work, the way they always do when half the company is already mentally on PTO. I flew out to France to spend the holidays with my parents, and somewhere between the jet lag and my mom's cooking, I found myself with actual uninterrupted thinking time for the first time in months. Long walks through the village, bad Wi-Fi that kept me off Slack, and a laptop I'd told myself I wouldn't open but obviously did. What follows is the result of that Christmas week exploration, fueled less by coffee and more by vin chaud.

## 1. Why Not Just Context Engineering?

The first thing I tried (and probably what you'd try too) was cramming everything into the prompt. System message with brand guidelines, a few examples of approved copy via few-shot, maybe RAG over a library of past campaigns. No training, no GPUs, ship it by Friday.

And honestly? For a single brand with a straightforward voice, this works fine. Here's the kind of prompt I was playing with:


https://gist.github.com/cmenguy/dd3cff44ef39441eae97c33b322422e3


Llama 3.1 8B does a decent job with this. You get something like "Cold feet aren't part of the plan." Not bad, sounds roughly on-brand. So why bother fine-tuning?

Three reasons kept pushing me past the prompt-engineering ceiling:

**Context window budget.** When you're serving 50 different brands, each with its own voice guidelines and example library, the prompt gets heavy fast. A 2,000-token brand context block times 50 brands means every request is burning tokens on instructions instead of thinking. Fine-tuning bakes the knowledge into the weights. The prompt stays lean, inference is cheaper, and latency drops.

**Consistency at the margins.** Prompting gets you 80% of the way there, but the last 20% is where brand voice actually lives. The subtle things (whether the brand uses Oxford commas, whether contractions are OK, whether humor is dry or playful) drift across generations with prompting alone. I ran 100 generations with the same prompt and kept seeing the model slip into its default "helpful assistant" voice on maybe 15-20% of them. After SFT on 500 brand-specific examples, that drop rate went to near zero.

**Preference capture.** This is the big one. Prompting can tell a model *what* to write, but it can't teach it *what's better*. In marketing, you have A/B test data: version A got 3x the click-through rate of version B. That signal is gold, and there's no way to feed it into a prompt. You need DPO or RLHF to actually learn from preference pairs, and that requires training.

So the rule of thumb I landed on: if you have one brand and simple requirements, prompt engineering is the right call. If you have many brands, need tight consistency, or have preference data you want to exploit, that's when fine-tuning starts paying for itself.

## 2. The Setup

Here's the scenario I keep coming back to. You're building an AI-powered marketing platform. Customers upload their brand guidelines, past campaigns, and tone-of-voice docs. They want the LLM to generate email subject lines, ad copy, social media posts, and product descriptions that are indistinguishable from what their in-house team would write.

This is a preference-heavy domain. "Good" marketing copy isn't just grammatically correct; it's persuasive, on-brand, and sometimes deliberately breaks rules for effect. That matters a lot for which fine-tuning approach you choose.

Here's the quick mental model before we go deep:


https://gist.github.com/cmenguy/bdea3cd36a24a318501025363c5a82de


## 3. Supervised Fine-Tuning (SFT): Teaching by Example

SFT is the workhorse. You show the model examples of what good output looks like, and it learns to imitate them. If you've ever fine-tuned a model on a dataset of prompt-completion pairs, you've done SFT.

#### 3.1 The Math

SFT minimizes the standard language modeling cross-entropy loss over your curated dataset. Given a dataset of $(x, y)$ pairs where $x$ is the prompt and ![equation](https://latex.codecogs.com/png.latex?\inline%20y%20%3D%20%28y_1%2C%20y_2%2C%20%5Cldots%2C%20y_T%29) is the target completion:

![equation](https://latex.codecogs.com/png.latex?\dpi{150}\bg_white%20%5Cmathcal%7BL%7D_%7B%5Ctext%7BSFT%7D%7D%28%5Ctheta%29%20%3D%20-%5Cmathbb%7BE%7D_%7B%28x%2Cy%29%20%5Csim%20%5Cmathcal%7BD%7D%7D%20%5Cleft%5B%20%5Csum_%7Bt%3D1%7D%5E%7BT%7D%20%5Clog%20%5Cpi_%5Ctheta%28y_t%20%5Cmid%20x%2C%20y_%7B%5Clt%20t%7D%29%20%5Cright%5D)

Where ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Cpi_%5Ctheta) is your model parameterized by ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Ctheta). This is just next-token prediction on your curated data. Nothing fancy, and that's the point.

#### 3.2 When SFT Makes Sense

SFT is your go-to when you have clean, high-quality examples of the output you want. For marketing, this means curated campaign copy that's already been approved by the brand team. Think of it as cloning your best copywriter's style.

The catch: SFT can only learn from positive examples. It knows what "good" looks like, but it has no concept of "this is better than that." If your training data has any noise (mediocre examples mixed in with great ones), the model learns to reproduce the average, not the best.

#### 3.3 The Code

Let's fine-tune a model on marketing copy using TRL's `SFTTrainer`. First, the data prep: structuring brand-specific examples as conversations:


https://gist.github.com/cmenguy/f044adb91302e8596adff8ef809029e4


Now the training setup. LoRA keeps this feasible on a single GPU:


https://gist.github.com/cmenguy/33bc44404759c62a8a68d955776146a9


And the trainer. Note the `max_length` tuned for short-form marketing copy:


https://gist.github.com/cmenguy/abc05856190d642c24f36476f90eab6b


That's it. With 500-1000 curated examples per brand, you'll get a model that captures tone and style surprisingly well. SFT is the 80/20 of fine-tuning: it gets you most of the way there with the least effort.

## 4. RLHF: The Full Pipeline

Reinforcement Learning from Human Feedback is the approach that made ChatGPT feel different from GPT-3. The idea: train a separate reward model on human preferences, then use it to guide the LLM's behavior via reinforcement learning.

#### 4.1 The Math

RLHF has three stages, each with its own objective.

**Stage 1: SFT.** Same as above. You start with a supervised fine-tuned model ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Cpi_%7B%5Ctext%7BSFT%7D%7D) as your starting point.

**Stage 2: Reward Model Training.** Given pairs of responses ![equation](https://latex.codecogs.com/png.latex?\inline%20%28y_w%2C%20y_l%29) where a human preferred ![equation](https://latex.codecogs.com/png.latex?\inline%20y_w) over ![equation](https://latex.codecogs.com/png.latex?\inline%20y_l), train a reward model ![equation](https://latex.codecogs.com/png.latex?\inline%20r_%5Cphi) using the Bradley-Terry preference model:

![equation](https://latex.codecogs.com/png.latex?\dpi{150}\bg_white%20%5Cmathcal%7BL%7D_%7B%5Ctext%7BRM%7D%7D%28%5Cphi%29%20%3D%20-%5Cmathbb%7BE%7D_%7B%28x%2C%20y_w%2C%20y_l%29%20%5Csim%20%5Cmathcal%7BD%7D%7D%20%5Cleft%5B%20%5Clog%20%5Csigma%5Cleft%28r_%5Cphi%28x%2C%20y_w%29%20-%20r_%5Cphi%28x%2C%20y_l%29%5Cright%29%20%5Cright%5D)

Where ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Csigma) is the sigmoid function. The reward model learns to assign higher scores to preferred responses.

**Stage 3: RL Optimization.** Maximize the reward while staying close to the SFT policy via a KL penalty:

![equation](https://latex.codecogs.com/png.latex?\dpi{150}\bg_white%20%5Cmathcal%7BL%7D_%7B%5Ctext%7BRLHF%7D%7D%28%5Ctheta%29%20%3D%20%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20%5Cmathcal%7BD%7D%2C%20y%20%5Csim%20%5Cpi_%5Ctheta%28%5Ccdot%20%5Cmid%20x%29%7D%20%5Cleft%5B%20r_%5Cphi%28x%2C%20y%29%20-%20%5Cbeta%20%5C%2C%20D_%7B%5Ctext%7BKL%7D%7D%5Cleft%28%5Cpi_%5Ctheta%28%5Ccdot%20%5Cmid%20x%29%20%5C%7C%20%5Cpi_%7B%5Ctext%7BSFT%7D%7D%28%5Ccdot%20%5Cmid%20x%29%5Cright%29%20%5Cright%5D)

The classic approach used PPO (Proximal Policy Optimization) for this stage, but modern implementations have moved on. TRL now uses RLOO (REINFORCE Leave-One-Out), which is simpler and more stable than PPO while achieving comparable results. The KL term remains critical: without it, the model collapses to reward hacking, generating degenerate text that scores high on the reward model but is gibberish to humans.

#### 4.2 When RLHF Makes Sense

RLHF shines when preferences are nuanced and hard to capture with static examples alone. It's the gold standard for aligning models to complex human judgment, which is exactly what "good marketing copy" requires.

The problem: it's expensive. You need to train and maintain a separate reward model. The RL optimization stage adds instability. Loss curves that look like seismographs are par for the course. And you need a *lot* of preference data to train a reward model that doesn't just memorize your training set.

For a marketing use case, RLHF makes sense at scale: when you have thousands of A/B tested campaigns and can build a reward model that genuinely captures what makes copy convert. For most teams, this is overkill.

#### 4.3 The Code

Let's build the full pipeline. First, the reward model, trained on pairs where one version of copy outperformed another:


https://gist.github.com/cmenguy/7065cd395e8fc5c0d005b0c619b4f081


Training the reward model using TRL's `RewardTrainer`:


https://gist.github.com/cmenguy/1c94c2e1467e0fa312f8d77a9303364d



https://gist.github.com/cmenguy/cf8fc636069a1e649ff1a6e49f0ba291


Now the RL stage: this is where the model learns to generate copy that scores high on the reward model. TRL's `RLOOTrainer` (REINFORCE Leave-One-Out) handles generation, scoring, and policy updates in a unified loop. First, prepare the prompt dataset:


https://gist.github.com/cmenguy/93378edd9186219bd8344c63f5f176ba


Configure and run the RLOO trainer: it generates multiple completions per prompt and uses the leave-one-out baseline to reduce variance:


https://gist.github.com/cmenguy/e8725d3d1b93af8c85da659e6d7d457d


That's still a lot of moving parts: two models, generation during training, and hyperparameters that interact in non-obvious ways. RLOO is more stable than the old PPO approach, but the fundamental complexity of reward-model-guided RL remains. This is why DPO exists.

## 5. DPO: The Pragmatist's Choice

Direct Preference Optimization is the "what if we just... skipped the reward model?" approach. Published in 2023, DPO showed that you can optimize directly on preference pairs without ever training a separate reward model or running RL. It's mathematically equivalent to RLHF under certain assumptions, but dramatically simpler to implement.

#### 5.1 The Math

The key insight of DPO: the optimal policy under the RLHF objective has a closed-form relationship with the reward function. Instead of learning ![equation](https://latex.codecogs.com/png.latex?\inline%20r_%5Cphi) and then optimizing ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Cpi_%5Ctheta) against it, DPO reparameterizes the reward in terms of the policy itself:

![equation](https://latex.codecogs.com/png.latex?\dpi{150}\bg_white%20r%28x%2C%20y%29%20%3D%20%5Cbeta%20%5Clog%20%5Cfrac%7B%5Cpi_%5Ctheta%28y%20%5Cmid%20x%29%7D%7B%5Cpi_%7B%5Ctext%7Bref%7D%7D%28y%20%5Cmid%20x%29%7D%20%2B%20%5Cbeta%20%5Clog%20Z%28x%29)

Where ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Cpi_%7B%5Ctext%7Bref%7D%7D) is the reference policy (your SFT model) and $Z(x)$ is a partition function that cancels out in the preference comparison. Substituting into the Bradley-Terry model gives the DPO loss:

![equation](https://latex.codecogs.com/png.latex?\dpi{150}\bg_white%20%5Cmathcal%7BL%7D_%7B%5Ctext%7BDPO%7D%7D%28%5Ctheta%29%20%3D%20-%5Cmathbb%7BE%7D_%7B%28x%2C%20y_w%2C%20y_l%29%20%5Csim%20%5Cmathcal%7BD%7D%7D%20%5Cleft%5B%20%5Clog%20%5Csigma%20%5Cleft%28%20%5Cbeta%20%5Clog%20%5Cfrac%7B%5Cpi_%5Ctheta%28y_w%20%5Cmid%20x%29%7D%7B%5Cpi_%7B%5Ctext%7Bref%7D%7D%28y_w%20%5Cmid%20x%29%7D%20-%20%5Cbeta%20%5Clog%20%5Cfrac%7B%5Cpi_%5Ctheta%28y_l%20%5Cmid%20x%29%7D%7B%5Cpi_%7B%5Ctext%7Bref%7D%7D%28y_l%20%5Cmid%20x%29%7D%20%5Cright%29%20%5Cright%5D)

Read it like this: push up the probability of the preferred response ![equation](https://latex.codecogs.com/png.latex?\inline%20y_w) relative to the reference model, while pushing down the probability of the rejected response ![equation](https://latex.codecogs.com/png.latex?\inline%20y_l), all modulated by ![equation](https://latex.codecogs.com/png.latex?\inline%20%5Cbeta).

#### 5.2 When DPO Makes Sense

DPO is the sweet spot for most production fine-tuning. You get preference learning without the infrastructure complexity of RLHF. No reward model to train and serve. No PPO instability. Just preference pairs and a single training run.

For marketing, this is particularly appealing. You likely have A/B test results, editorial feedback, and brand compliance reviews, all natural sources of "this version is better than that version" data. DPO consumes this directly.

#### 5.3 The Code

DPO needs an SFT model as the starting point (the reference policy) and a dataset of preference pairs. The data is the same format we used for the reward model:


https://gist.github.com/cmenguy/837e66f0924adcca0a9e30285f327e25


The training setup is refreshingly simple: one model, one trainer, one loss function:


https://gist.github.com/cmenguy/6a06b8be8281cdff2a4e722fe2fe4d7f



https://gist.github.com/cmenguy/dcfb8c7f918eb6f79fd466395fa29a31


Compare this to the RLHF pipeline above. Same preference data, same end goal, a fraction of the complexity.

## 6. The Comparison: What Actually Works for Marketing

Let's put these side by side for the marketing fine-tuning scenario:


https://gist.github.com/cmenguy/64e2fda7ce188d879007515252056199


#### 6.1 My Recommendation for Marketing

**Start with SFT. Graduate to DPO. Reach for RLHF only if you have to.**

Here's why, stage by stage:

**Phase 1: SFT to capture brand voice.** Collect 500-1000 examples of approved marketing copy per customer. Email subjects, product descriptions, social posts, all tagged with the prompt that generated them. Fine-tune with LoRA. This alone gets you 80% of the way to sounding like the brand.

**Phase 2: DPO to learn preferences.** Once you have SFT running, start collecting preference data. This is natural in marketing: A/B test results (version A got 3x clicks), editorial reviews (the brand team picked option 2), compliance flags (this version violated guidelines). Feed these as preference pairs into DPO. The model learns *why* one version is better than another, not just *what* good copy looks like.

**Phase 3: RLHF only at scale.** If you're running millions of campaigns across hundreds of brands and have a genuine signal for what converts, a dedicated reward model starts to make sense. It can generalize across brands and capture patterns that individual preference pairs can't. But you need the data volume and the engineering team to support it.

For most teams starting out, SFT + DPO is the right answer. You get preference learning, training stability, and a pipeline that a single ML engineer can own. RLHF is a great system when you grow into it; it's a terrible starting point.

## 7. The Hidden Variable: Data Quality

None of this matters if your data is garbage. A few hard-won lessons:

**For SFT data:** only include examples you'd be proud to show a client. One mediocre example teaches the model more bad habits than ten good examples can fix. Have your best copywriter review every training example.

**For preference pairs:** the margin matters. "Slightly better" pairs teach less than "clearly better" pairs. When collecting A/B test data, filter for cases where one version significantly outperformed the other. A 51/49 click split is noise, not signal.

**For both:** diversity beats volume. 500 examples across 10 different copy formats (emails, ads, social, product descriptions) will outperform 2000 examples of just email subject lines. The model needs to learn the brand voice, not just one template.

## 8. Wrapping Up

SFT teaches your model what good looks like. DPO teaches it what *better* looks like. RLHF builds a full preference engine. For customer-specific marketing fine-tuning, the SFT-then-DPO pipeline gives you the best ratio of capability to complexity. Start there, measure what you're getting, and scale up the sophistication only when the data and the business justify it.

The code in this post (and the companion notebook) uses TRL throughout. It's become the de facto library for all three approaches, and keeping your entire fine-tuning stack in one framework means less glue code and fewer surprises when you upgrade.


---

*Originally published on [AI Terminal](https://ai-terminal.net/llm/fine-tuning/2025/12/22/fine-tuning-llms-sft-dpo-rlhf/).*

Tags: sft, dpo, trl, rlhf, marketing
