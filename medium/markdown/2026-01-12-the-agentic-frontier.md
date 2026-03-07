# Here Be Dragons

I spent the holidays in France with my family: the usual routine of too much food, bad Wi-Fi, and long walks through the countryside. But this time was different. Normally I use that downtime to tinker with side projects or catch up on papers. This year I couldn't focus on any of that. I kept circling the same question, somewhere between my second glass of Burgundy and my parents asking me to explain what I do for a living again: *what exactly is my job anymore?*

I've been a senior IC in ML engineering for years. I know what that identity feels like: you're the person who can hold the whole system in your head, who knows which abstractions are load-bearing, who mentors the junior folks and pushes back on product when the timeline is fantasy. That identity has been stable for a long time. But over the past few months, something shifted. I started using agentic coding tools (Claude Code, Cursor, Copilot) and at first it felt like a productivity boost. Then it started feeling like something else entirely. Like the ground was moving under my feet.

By the time I got on the flight back home in early January, I'd stopped calling it a "productivity tool" in my head. I was calling it a fundamental shift. And I was having a genuine identity crisis about what it means for me, for my team, and for this entire profession.

## 1. Landing on a New Continent

On medieval maps, when cartographers reached the edge of the known world (the parts no explorer had returned from yet), they'd scrawl a warning in the blank space: *Hic sunt dracones*. Here be dragons. It wasn't a statement about actual dragons. It was an admission: we don't know what's out there, and it might be dangerous.

The best analogy I've found for what's happening right now is the Age of Exploration. When European navigators first reached the Americas, they didn't immediately understand the scale of what they'd found. They thought they'd found a faster route to India. They mapped coastlines, set up small trading posts, and tried to fit the new territory into their existing mental models. It took decades before the full implications became clear: new economies, new power structures, new ways of organizing entire civilizations.

That's where we are with agentic AI. Most people (including me until recently) are still in the "faster route to India" phase. We think we've found a better autocomplete. A smarter Stack Overflow. A way to write boilerplate faster. But what we've actually stumbled onto is a new continent, and we're still standing on the beach drawing crude maps. The rest of the map says *here be dragons*.

The frontier isn't autocomplete. The frontier is *agency*: AI systems that can hold context across an entire codebase, make multi-step decisions, run commands, verify their own output, and iterate until the job is done. That changes everything, and not everyone has realized it yet.

## 2. The Code Analogy: From Functions to Orchestrators

Here's a way to think about what's happening, in terms we understand. For most of software engineering history, the job has been writing functions. You take a spec, you decompose it into components, you implement each piece, you wire them together. Senior engineers are good at this because they've built enough systems to know where the abstractions should go.

Think of the old model like this:


https://gist.github.com/cmenguy/f5b2e850fff72b688efd2587c729f06f


You are the loop. You decompose, implement, review, test, ship. Your value is in the quality of each step and your judgment about what to build.

Now here's what the agentic model looks like:


https://gist.github.com/cmenguy/e68d9ffa103dfacf3a5186910df3efbd


You're no longer the loop. You're the orchestrator. Your value shifts from "can you write this code" to "do you know what code should exist, why, and how to verify it's correct." The implementing is increasingly delegated. The judgment, architecture, and customer-facing intent are what remain uniquely yours.

This isn't hypothetical. Here's what my actual workflow looks like now with Claude Code:


https://gist.github.com/cmenguy/f50921a8f50784c5dfd3a6f1f10100b0


The time I save on implementation goes straight into customer discovery, architecture review, and cross-team coordination. That's the real shift.

## 3. What This Means for Everyone

This reorganization of labor ripples out differently depending on where you sit. Let me be honest about what I'm seeing: not the optimistic conference talk version, the real version.

#### 3.1 Senior ICs

This is us. The identity crisis crew. For a long time, senior ICs derived status and job security from being the person who could build the hard thing. You knew the codebase intimately. You could debug that race condition nobody else could reproduce. You had the muscle memory of ten thousand pull requests.

Agentic tools compress that. A staff engineer with Claude Code can now do what used to require a staff engineer *and* two senior engineers. The raw coding throughput advantage that comes with experience is getting arbitraged away.

What doesn't get arbitraged: knowing *what* to build. Knowing which of the five possible architectures will survive contact with production at scale. Knowing when the product spec is subtly wrong because you've seen this failure mode before. Knowing how to break a quarterly roadmap into a sequence of bets that compound.

The senior IC role doesn't disappear; it evolves. Less "I shipped this feature" and more "I made sure the right features got shipped correctly and we measured the impact." The scope expands from code to customer. If that sounds more like a tech lead or an architect, that's because the boundary between those roles is dissolving.

#### 3.2 Junior ICs

Here's the counterintuitive part: I think agentic AI is *better* for junior engineers than people expect, but in a specific way. The boring ramp-up period (learning the codebase, understanding build systems, writing your first CRUD endpoints) gets dramatically shorter. A junior engineer with Cursor can be productive on day one in a way that wasn't possible before.

The risk is that they skip the understanding phase. You can ship code you don't understand. You can land PRs that pass CI but that you couldn't explain in a code review. And that's where senior ICs matter more than ever: not as implementers, but as teachers and quality gates. Code review becomes less about style nits and more about "do you actually understand what this does and why."

#### 3.3 Product

Product managers are about to have a very interesting few years. The cost of building a prototype is dropping toward zero, which means the cost of *testing an idea* is dropping toward zero. A PM who can articulate a clear spec can now get a working prototype from an agentic coding session in hours, not sprint cycles.

This shifts the bottleneck from engineering capacity to product judgment. The question stops being "can we build this in time" and starts being "should we build this at all." PMs who are good at customer discovery and ruthless prioritization will thrive. PMs who were basically project managers tracking Jira velocity will struggle.

#### 3.4 Engineering Leadership

The org chart implications are real and uncomfortable. If one senior engineer with agentic tools can do the work of three, what does that mean for team size? For hiring plans? For career ladders built around managing larger teams?

I don't think the answer is "fire two-thirds of your engineers." I think the answer is "the same team can now attempt things that were previously out of scope." The limiting factor moves from "how many engineers do we have" to "how many good ideas do we have and how fast can we validate them." Engineering leaders who reorient around speed-of-learning rather than headcount-per-project will build better organizations.

## 4. The AI/ML Engineering Angle

This is where it gets personal for me. I'm an ML engineer. My specialty is supposed to be the hard stuff: training loops, model optimization, infrastructure, evaluation harnesses. For years, this was a high barrier-to-entry domain. You needed to understand linear algebra, distributed systems, GPU memory management, and the dark arts of hyperparameter tuning. That scarcity was part of our value.

Agentic AI is lowering that barrier fast. Here's an example. This is what deploying a model with vLLM used to require: understanding CUDA, memory management, quantization trade-offs.


https://gist.github.com/cmenguy/8728086bb703c6e141710b70a39167f6


Each of those flags represents a decision that used to require domain expertise. Tensor parallelism? You need to know your GPU topology. Memory utilization at 0.9? You need to understand KV cache sizing. AWQ quantization? You need to know the accuracy-latency trade-off for your specific use case.

Now someone can ask Claude Code "deploy Llama 3.1 8B for my API with the lowest latency on my 2xA100 setup" and get a working config. The barrier to entry just dropped from "months of learning" to "minutes of prompting."

But here's the thing: the barrier dropped for *getting started*. It didn't drop for *getting it right in production*. When that vLLM server OOMs under load at 2 AM, when the quantized model starts hallucinating on edge cases your eval suite didn't cover, when you need to debug why throughput dropped 40% after a library upgrade, that's where deep understanding still matters.

The net effect for ML engineers is actually a win if you lean into it. I can now prototype things in a day that used to take a week. I can explore architectures I wouldn't have had time for. I can build whole systems, data pipeline to model to API to frontend to metrics dashboard, because the agentic tools cover my weak spots (frontend, mainly) while I focus on the parts where my expertise matters most.

The identity shift for ML engineers specifically: less "I'm the person who knows how to train models" and more "I'm the person who knows how to build AI systems that reliably deliver value." The scope expands from model to product.

## 5. The Tools in January 2026

Everyone I talk to is using *something*. Things are moving so fast that what's best today might not be best next month, but here's my honest read of where things stand as I write this:

**Claude Code (Anthropic)**: this is my daily driver and it's not close. The agentic loop is the key differentiator. It doesn't just suggest code; it reads your codebase, runs commands, checks its own work, and iterates. For complex multi-file changes, refactors, and "build me this feature" type work, nothing else I've used comes close. Claude's ability to hold architectural intent across a long session (understanding not just the file you're in but the whole system you're building) is genuinely impressive. With Sonnet 3.5 and the newer Claude 3.5 Opus under the hood, the quality of the code it produces has taken a real step up from even six months ago.

**Cursor**: still the best IDE-integrated experience. If you're an "I live in my editor" person, Cursor's inline completions and chat are excellent. It's better than Claude Code for small, targeted edits: the "fix this function" or "add a docstring here" kind of work. Where it falls short is multi-file orchestration. It thinks in files, not in systems. That said, the Composer feature has gotten significantly better at multi-step tasks.

**GitHub Copilot**: still solid for line-by-line completions and increasingly good with its chat and workspace agent features. The advantage is ubiquity; it works everywhere, the VS Code integration just works, and for many engineers it's the gateway drug to AI-assisted coding. Copilot has been leaning harder into agentic capabilities lately, but it still feels a step behind Claude Code and Cursor for deep codebase understanding.

**Antigravity**: the dark horse that had its moment this winter. What caught my attention is the generous free tier. You get access to both Gemini and Claude models without burning through your own API credits, which makes it a fantastic way to experiment. The UX is clean, it handles multi-file edits well, and for someone who wants to try agentic coding without committing to a $20/month subscription on every tool, it's the obvious starting point. I don't think it's unseated Claude Code or Cursor for power users yet, but it's lowered the floor in a way that matters, especially for junior engineers or anyone just getting their feet wet.

My personal stack: Claude Code for big tasks and architecture-level work, Cursor for in-editor flow state, Antigravity when I want to quickly test something with a different model without fiddling with API keys. Your mileage will vary. The best tool is the one that fits your workflow.

## 6. Where I Go From Here

I came back from the holidays with a decision. I'm going to stop defining my value by the code I write and start defining it by the outcomes I drive. That sounds like corporate-speak, so let me be specific.

Before, a good week looked like: merged 5 PRs, fixed that gnarly caching bug, got the new endpoint into staging. Concrete, tangible, mine.

Now, a good week looks like: identified that our model serving costs are 3x higher than they need to be because we're over-provisioning, used Claude Code to prototype three different optimization approaches in a day instead of a sprint, got the best one into production, and showed the finance team we saved $40K/month. The code is almost a byproduct. The value is the insight, the speed, and the business impact.

This is uncomfortable. I got into engineering because I love building things with my hands (metaphorically). There's a deep satisfaction in writing an elegant solution to a hard problem that no AI can replicate for me emotionally, even if it can replicate the code. I'm still processing that loss.

But the new continent is out there, and the map still says *here be dragons*. The people who thrive won't be the ones who insist on doing everything by hand. They'll be the ones who learn the new terrain, figure out what's actually valuable, and move fast while everyone else is still arguing about whether the map is real.

The dragons are real. The territory is unknown. Go anyway.


---

*Originally published on [AI Terminal](https://ai-terminal.net/meta/ai-engineering/2026/01/12/the-agentic-frontier/).*

Tags: cursor, copilot, senior-ic, agentic-ai, claude-code
