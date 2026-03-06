# The Agentic Protocol Zoo

---

**Interactive examples of A2A agent cards, MCP tool definitions, and multi-agent communication patterns.**

[Run in Google Colab](https://colab.research.google.com/github/cmenguy/cmenguy.github.io/blob/main/notebooks/2026-01-28-agentic-protocols.ipynb) | [View on GitHub](https://github.com/cmenguy/cmenguy.github.io/blob/main/notebooks/2026-01-28-agentic-protocols.ipynb)

---

I've been building agents at work for the past few months. Not the "wrap an LLM in a while loop and call it an agent" kind, but actual multi-step systems that read data, call APIs, make decisions, and hand off work to other systems. The kind where you wake up on a Monday and realize you now have six different services that need to talk to each other, and none of them agree on what a "task" is.

It started innocuously enough. We had one agent handling a customer-facing workflow, and it needed to call out to a specialized retrieval system. Easy: just make an API call, right? Then another team built their own agent for a different use case, and suddenly someone in a design review is asking "can your agent talk to their agent?" and the room goes quiet because nobody has a good answer. By December I had a whiteboard full of arrows and a growing conviction that we were reinventing the wheel badly.

## 1. Why Not Just Wing It?

The temptation when you're iterating fast is to skip the protocol question entirely. Just define a REST endpoint, pass some JSON, move on. I've done this. Everyone has done this. And it works — until it doesn't.

Here's where it breaks down. Say you have Agent A (a planning agent) that needs to delegate a research task to Agent B (a retrieval agent). The naive approach:


https://gist.github.com/cmenguy/d498869071d738b2babb124b7d326c3c


This works for request-response. But what happens when the research takes 45 seconds? Or 5 minutes? Now you need timeouts, polling, maybe webhooks. What happens when you want to swap out the retrieval agent for a different one? Now you need discovery. What about auth? What about streaming partial results back so the user isn't staring at a spinner? Each of these is a custom integration, and you end up writing more glue code than actual agent logic.

That's the problem protocols solve. Not because they're elegant (they're not, they're specifications with committee fingerprints all over them), but because they let you stop reinventing the boring parts.

## 2. The Two Protocols That Matter

As of early 2026, two protocols have emerged with real traction: **MCP** (Model Context Protocol, from Anthropic) and **A2A** (Agent-to-Agent, originally from Google, now under the Linux Foundation). There are others (OpenAI's function calling pattern, the Agent Protocol from AGI Inc., LangChain's various abstractions), but MCP and A2A are the ones I see actually getting adopted in production systems.

They solve fundamentally different problems, and understanding that distinction is the whole game.

**MCP** is about giving a single AI application access to tools and data. Think of it as a USB-C port for your LLM: you plug in servers that provide capabilities (read a database, call an API, search files), and the AI host uses them. The architecture is a star: one host at the center, many servers around it.

**A2A** is about agents talking to other agents as peers. Think of it as a phone system for autonomous services: each agent advertises what it can do, and other agents can discover and delegate work to it. The architecture is a mesh: any agent can talk to any other.


https://gist.github.com/cmenguy/ce921aabd74b2cd96755ed9af709ab8c


They're not competing. They're complementary layers. An A2A agent might internally use MCP to access tools. Let me break each one down.

## 3. MCP: The Context Layer

MCP's mental model is simple: an AI application (the "host") connects to multiple MCP servers, each of which provides some combination of three primitives: **tools**, **resources**, and **prompts**.

**Tools** are functions the model can call. A weather tool, a database query tool, a code execution tool. The model decides when to use them.


https://gist.github.com/cmenguy/0174f03f4bfc3bc5f58df32f5f519434


**Resources** are data the application can pull in for context: files, database records, API responses. Unlike tools, resources are typically application-driven (the UI exposes them for selection) rather than model-driven.


https://gist.github.com/cmenguy/81b5b743434424bba2f8ca7dd4cfe0e2


**Prompts** are reusable templates: think slash commands. The user explicitly selects them.

The transport layer is either **stdio** (for local servers where your MCP server runs as a subprocess and communicates over stdin/stdout) or **Streamable HTTP** (for remote servers that use HTTP POST for requests and Server-Sent Events for streaming responses).

Here's what a minimal MCP server looks like in practice:


https://gist.github.com/cmenguy/88e3d517057cbe9d8ee5c37eba7cb840


The key thing about MCP: **it's not agent-to-agent communication**. It's one AI application accessing capabilities. The AI host is the brain; the MCP servers are the hands. There's a clear hierarchy. The host decides what to do, the servers execute.

This works phenomenally well for the single-agent case. Claude Code connecting to your filesystem, your database, your Jira instance: that's MCP's sweet spot. The protocol handles discovery (servers advertise their tools via `tools/list`), capability negotiation (during initialization, client and server agree on what each supports), and tool execution (the model calls a tool, the result comes back).

Where MCP falls short is when you need two autonomous systems to collaborate. MCP servers don't have agency. They don't make decisions, they don't hold state across interactions in a meaningful way, they don't negotiate with the host about *how* to accomplish something. They're tools, not peers.

## 4. A2A: The Agent Coordination Layer

A2A starts from a different premise: agents are autonomous entities that need to discover each other, negotiate capabilities, delegate tasks, and coordinate on long-running work.

The foundational concept is the **Agent Card**: a JSON document that describes what an agent can do. Think of it as a résumé that other agents can read programmatically.


https://gist.github.com/cmenguy/58697457db67be75a3e2a1bf7e0bdd6b


This is more than a tool definition: it's an agent's identity. The `skills` field tells other agents what tasks this agent can handle, with human-readable examples. The `capabilities` field advertises protocol features (can it stream? does it support push notifications?). The `securitySchemes` field tells clients how to authenticate.

#### 4.1 Agent Discovery

In a complex ecosystem with dozens of agents, discovery matters. Agent Cards are typically hosted at a well-known URL (like `/.well-known/agent.json`), and client agents can fetch them to understand what's available. In enterprise settings, you might have an agent registry: a directory service that indexes Agent Cards across your organization.


https://gist.github.com/cmenguy/1d323b7bcc573339c9620976acf5ce5a


#### 4.2 The Task Lifecycle

Where A2A really separates itself from "just make an API call" is its task model. When Agent A sends a message to Agent B, the interaction is wrapped in a **Task** that progresses through well-defined states:


https://gist.github.com/cmenguy/26f933804fa7600fe436dbf13da621dd


This matters because real agent work isn't always instant. A research agent might need minutes. A code generation agent might need to iterate. A data pipeline agent might need hours. The task model gives you a shared vocabulary for "what's happening right now" and "what happened."


https://gist.github.com/cmenguy/f9b1eb2f7220fdd3bec7955b527a86cf


#### 4.3 Short-Lived vs Long-Running Tasks

A2A handles the temporal spectrum gracefully through three execution patterns:

**Immediate (synchronous):** For quick operations, set `blocking: True` and get a response directly. The agent processes the request and returns a completed task or message in the same HTTP response.


https://gist.github.com/cmenguy/ad13da8f7c4c6df382109b17f090df8f


**Streaming (Server-Sent Events):** For tasks with incremental output, use `SendStreamingMessage`. You get a stream of events: `TaskStatusUpdateEvent` for state changes, `TaskArtifactUpdateEvent` for partial results.


https://gist.github.com/cmenguy/06d9a00c829fe5b039d33a5546a597f3


**Polling / Push (asynchronous):** For long-running work, send the message with `blocking: False`, get back a task ID, and either poll with `GetTask` or set up a webhook for push notifications.


https://gist.github.com/cmenguy/88b191e709e4edd24f8ace3fc594081e


This three-tier approach is what makes A2A practical for production systems. A simple lookup agent responds in milliseconds (synchronous). A summarization agent streams tokens as it generates (streaming). A data pipeline agent takes 20 minutes and pushes a notification when it's done (async with webhooks).

#### 4.4 Extensions

A2A has an extension mechanism for capabilities beyond the core spec. Agents declare supported extensions in their Agent Card, and clients signal which extensions they want to activate per-request.


https://gist.github.com/cmenguy/da69cdd485d376aa5ab7efdce8b59666


This is how you get domain-specific behavior without bloating the core protocol. A financial services agent might support a compliance extension. A healthcare agent might support a HIPAA audit trail extension. The core protocol stays lean.

#### 4.5 Protocol Bindings

A2A doesn't lock you into one transport. The spec defines three protocol bindings:

- **JSON-RPC 2.0 over HTTP(S):** The default. Method-based invocation, familiar to anyone who's used Ethereum APIs or LSP.
- **gRPC:** For high-throughput, low-latency scenarios. Protobuf-defined messages, bi-directional streaming.
- **HTTP+JSON (REST):** For teams that just want RESTful endpoints and don't want to think about JSON-RPC.


https://gist.github.com/cmenguy/f18f05c536a19887f1eae1979d79b934


The protobuf definitions are the normative source of truth, with JSON Schema auto-generated for broader compatibility. In practice, most teams I've talked to are using the JSON-RPC binding: it's the path of least resistance.

## 5. The Comparison Matrix

Here's my honest breakdown of when to reach for which:


https://gist.github.com/cmenguy/875cdf175be94e029ef3ccdcc5b42a3d


**Use MCP when** you're building one AI application that needs to access external tools and data. If you're building a coding assistant that needs filesystem access, database queries, and API calls: MCP.

**Use A2A when** you have multiple autonomous agents that need to discover each other and coordinate work. If you're building a system where a planning agent delegates to a research agent, which delegates to a data agent: A2A.

**Use both when** your agents need to coordinate with each other *and* each agent needs access to tools. The A2A agent uses A2A to talk to peer agents and MCP to access its own tools.


https://gist.github.com/cmenguy/b87d1178f3e5d66eef67bdf87ad621cd


## 6. The Honest Take: Complexity vs Speed

Here's where I stop being an impartial protocol explainer and start being an engineer who's shipped multi-agent systems.

**Multi-agent systems are seductive and treacherous.** The architecture diagrams look beautiful on a whiteboard. Agent A handles planning! Agent B handles research! Agent C handles execution! Clean separation of concerns! And then you actually build it and discover:

**The context fragmentation problem.** When you split work across agents, you split context. Agent A knows the user's intent. Agent B knows what it found during research. Agent C knows what happened during execution. But no single agent has the full picture. When something goes wrong (and it will), debugging requires reconstructing context across multiple systems. I've spent more time building tracing and logging infrastructure for multi-agent systems than building the actual agents.


https://gist.github.com/cmenguy/b3e0721b9043d635d081e0f1666db0e5


**The feedback loop problem.** Good agents learn from their mistakes within a session. They try something, see it fail, adjust. When you split an agent into multiple agents, each sub-agent optimizes locally. The research agent finds great sources but doesn't know they're irrelevant to the execution step. The execution agent does its best with what it got. The planning agent never finds out that its plan was subtly wrong because the error manifests two agents downstream.

**The latency tax.** Every agent-to-agent call adds latency. HTTP round trips. Serialization. Potentially cold starts if your agents are serverless. A task that takes 3 seconds in a single agent might take 15 seconds across three agents. The user doesn't care about your architecture; they care that it's slow.

**My honest recommendation:** Start with a single agent that does everything. Seriously. One agent, one context window, one feedback loop. Only split into multiple agents when you hit a concrete wall: the context window is literally too small, the latency of a single model call is unacceptable, or you need genuinely different models for different sub-tasks (e.g., a cheap fast model for triage and an expensive smart model for analysis).

When you do split, A2A gives you a real protocol instead of bespoke REST endpoints. That's valuable. But don't let the existence of a nice protocol convince you that you *need* multiple agents. The protocol is a tool, not an architecture mandate.

## 7. The Spectrum: Agents vs Skills

This brings me to something I've been thinking about a lot lately. The industry frames things as "single agent" vs "multi-agent," but there's a more nuanced spectrum:

**Single agent, no tools**: a raw LLM. Limited to what's in its weights.

**Single agent with tools (MCP)**: one agent that can call external functions. This is where most production systems should start.

**Single agent with skills**: one agent that has multiple specialized capabilities, each with their own instructions, tools, and context management. The agent itself decides which skill to invoke. This looks like multi-agent from the outside but maintains a single context and feedback loop.

**Multi-agent (A2A)**: genuinely separate agents coordinating over a protocol. Each has its own context, its own model, potentially its own infrastructure.


https://gist.github.com/cmenguy/709f77240587eb4d15503b0e600dbc8b


The "single agent with skills" pattern is the one I'm most excited about. You get the specialization benefits of multi-agent without the context fragmentation. The agent routes to the right skill based on the task, the skill has focused instructions and tools, but everything stays in one context window. When the skill needs to reference something from earlier in the conversation, it can, because the context is right there.

I'll be writing a deep dive on the skills pattern in an upcoming post, because I think it's the architecture that most teams should reach for before going full multi-agent.

## 8. Putting It Together: A Real Example

Let me sketch out a concrete architecture that uses both protocols. Say you're building a customer support system. Here's how the pieces fit:


https://gist.github.com/cmenguy/eee9b90808d06016698c6811e74ba765


The support agent uses MCP to look up customer info, check ticket history, and search the knowledge base. When it determines the issue needs specialized handling (a billing dispute, a complex technical problem), it delegates to a specialized A2A agent.


https://gist.github.com/cmenguy/aa66545e254e16a23ea91a5b67f5fd82


This is the pattern I keep coming back to: MCP for tools, A2A for delegation. Most requests get handled by the primary agent with its MCP tools. The expensive A2A delegation only happens when truly needed.

## 9. Where This Is All Going

The protocol ecosystem is still early. A2A hit v0.3 in late 2025 and is iterating fast under the Linux Foundation. MCP has broader adoption thanks to Claude and VS Code but is still evolving (the Streamable HTTP transport is relatively new). Neither protocol has "won" because they're not in the same race.

What I expect to happen:

1. **MCP becomes the standard context layer.** Every AI application will speak MCP the way every web app speaks HTTP. This is already happening: Claude, VS Code, Cursor, and others already support it.

2. **A2A becomes the standard for enterprise multi-agent systems.** As organizations deploy more specialized agents, they'll need A2A or something like it. The Google/Linux Foundation backing gives it credibility in enterprise.

3. **The "skills" pattern emerges as the pragmatic middle ground.** Most teams don't need multi-agent. They need one good agent with well-defined skills. I think we'll see frameworks coalesce around this pattern.

4. **Someone will build the "API gateway for agents"**: a proxy that handles A2A discovery, auth, rate limiting, and observability. This is the missing infrastructure piece.

For now, my advice: learn MCP first (you'll use it immediately), understand A2A conceptually (you'll need it eventually), and resist the urge to build a multi-agent system until a single agent with good tools genuinely can't do the job. The boring architecture is usually the right one.

Next up, I'll be writing about the skills pattern: how to give a single agent multiple specialized capabilities without the overhead of a full multi-agent system. Stay tuned.


---

*Originally published on [AI Terminal](https://ai-terminal.net/llm/ai-engineering/agents/2026/01/28/agentic-protocols/).*

Tags: a2a, mcp, agents, google, protocols
