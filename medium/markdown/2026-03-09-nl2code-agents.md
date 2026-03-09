# NL2Code with Agents: From English to Production Python

---

**End-to-end NL2Code pipeline: prompt-based, ReAct, and CodeAct agents generating marketing analytics code from plain English.**

[Run in Google Colab](https://colab.research.google.com/github/cmenguy/cmenguy.github.io/blob/main/notebooks/2026-03-09-nl2code-agents.ipynb) | [View on GitHub](https://github.com/cmenguy/cmenguy.github.io/blob/main/notebooks/2026-03-09-nl2code-agents.ipynb)

---

In [Part 1](https://ai-terminal.net/llm/ai-engineering/code-gen/2026/03/04/build-your-own-copilot/) I built a Copilot clone that does inline code completion. Fine-tuned it on our codebase, got it generating completions that actually follow our patterns. That was the easy half of the code generation story.

The harder half is what our marketing team actually asks for. They don't want autocomplete. They want to describe a problem in English ("build me a lookalike audience model from this seed list and score every user in our CDP") and get back working Python that plugs into our existing data pipelines. The code I shipped in Part 1 would see that request and spit out a generic scikit-learn script that imports libraries we don't use and writes to a CSV file instead of our Snowflake warehouse. Technically correct. Completely useless in production.

I'd been building a hardcoded agent for this at work. It followed a fixed sequence: parse the request, pick from a library of pre-built code templates, fill in the blanks. It worked for the five or six patterns we had encoded. But every new request type meant writing more templates. The marketing analytics team kept asking for things slightly outside the templates ("can you also add a holdout group?" or "run this on the EU segment only"), and the agent would just fail. What I wanted was something that could reason about the problem, figure out what code to write, and adapt when the requirements shifted. Not a template engine with an LLM on top, but an actual NL2Code system.

This post is **Part 2 of a 3-part series** on AI-assisted code generation. [Part 1](https://ai-terminal.net/llm/ai-engineering/code-gen/2026/03/04/build-your-own-copilot/) covered inline completion (FIM, code models, LoRA fine-tuning). This post covers NL2Code: instruction-following code generation where you describe what you want and an agent writes it. Part 3 will cover the other side: AI-assisted bug detection and fix suggestion.

## 1. The Problem: Marketing Audience Builder

Let me use a concrete example that I'll carry through the entire post. Our marketing analytics team needs to build lookalike audiences. The workflow:

1. Start with a **seed list** of high-value customers (say, users who spent >$500 in the last 90 days)
2. Pull **user features** from the CDP (customer data platform): demographics, browsing behavior, purchase history
3. Build a **similarity model** that scores every user in the database against the seed list
4. Output a **scored audience file** that the campaign platform can ingest, with a holdout group for measurement

In plain English, the request looks like this:

> "Build a lookalike audience from seed list `high_value_q1.csv`. Use purchase frequency, average order value, and days since last purchase as features. Score all users in the `user_features` table. Output the top 10% as the target audience with a 20% holdout group. Write results to `lookalike_audience_scored.parquet`."

This is a well-defined data science task. A senior engineer could write it in 30 minutes. The question is whether an LLM can write it in 30 seconds, and whether that code is good enough to actually run.

I'll walk through four approaches, each one more capable than the last, with real code and honest tradeoffs.

## 2. Approach 1: Single-Shot Prompting

The simplest thing that could work. Give the LLM the request, some context about our codebase, and ask it to write the code.


https://gist.github.com/cmenguy/40f5992e58d815e18ea035609f1d74c6


Let's test it with our marketing request:


https://gist.github.com/cmenguy/6c1289710f5082472583f35c9a619d06


The model produces something like this (cleaned up from an actual run):


https://gist.github.com/cmenguy/393c9f785b0fe660a0209a88bfa389ea


This is... decent. The model picked NearestNeighbors with cosine distance, which is a reasonable choice. It handles the scaler correctly (fit on seed, transform both). The holdout logic works.

But there are real problems:

**No reproducibility.** `np.random.rand` without a seed means the holdout group changes every run. Any marketer who re-runs this will get different results and wonder why their numbers shifted.

**No validation.** What if the feature columns don't exist in the data? What if the seed file is empty? Production code needs guardrails.

**No connection to our infra.** This reads from local files. Our data lives in Snowflake. The output goes to S3, not a local parquet file. The model has no idea about our stack.

**Hardcoded feature names.** The feature columns are string literals. If the user specifies different features in the next request, the model might or might not pick them up.

Single-shot prompting works for throwaway scripts. For anything that touches production, you need more.

#### 2.1 When Single-Shot Is Enough

Be honest about when simple works. Single-shot is fine for:

- Exploratory data analysis in a notebook
- One-off data transformations you'll run once
- Prototyping an algorithm before writing production code

The failure mode is people using single-shot for production code and then spending more time debugging the output than they would have spent writing it themselves.

## 3. Approach 2: Prompt Chaining with Validation

The next step: break the problem into stages and validate the output before moving on.


https://gist.github.com/cmenguy/46dc41273a06c4b499a1562f098a773f



https://gist.github.com/cmenguy/d37472b891ec41418964863f6534a406


This is better. The plan step forces the model to think about structure before writing code. The review step catches some of the issues (it usually adds `random_state=42` and basic column validation). The code-per-step approach keeps each generation focused.

But it's still a fixed pipeline. The model plans, codes, reviews, and you get the output. If the review step finds a fundamental design problem (wrong algorithm choice, missing a join between two tables), it can only patch the existing code. It can't go back and re-plan.


https://gist.github.com/cmenguy/55c35cc95294a5c17a203f8e6f04f46a


## 4. Approach 3: ReAct Agent with Tool Use

This is where things get interesting. Instead of a fixed pipeline, give the LLM tools and let it decide what to do at each step.

The [ReAct pattern](https://arxiv.org/abs/2210.03629) (Yao et al., 2022) interleaves reasoning and acting. The agent thinks about what to do, takes an action (calls a tool), observes the result, and decides what to do next. For NL2Code, the tools are things like "read a file," "check if a column exists," "run a code snippet," and "write output."

Here's the tool set:


https://gist.github.com/cmenguy/bc57870264801078c2464d344eb00b08



https://gist.github.com/cmenguy/1359103579c2d73d8deda5e60b2f6531



https://gist.github.com/cmenguy/01d25c050ebdf39c619dfc88d2ea9899



https://gist.github.com/cmenguy/daf3a9effead302650daaebd02b6aea1



https://gist.github.com/cmenguy/c0932a69e0095952d82ef635d4e52a49


Now the agent loop. This is the core of the ReAct pattern:


https://gist.github.com/cmenguy/c0c15ce45549ad57293c806c195cf814



https://gist.github.com/cmenguy/6e2d9863e94041d218ea9a77ea84c443


When you run this on our marketing request, the agent typically:

1. **Reads the seed file schema** to discover actual column names
2. **Reads the user features schema** to verify the feature columns exist
3. **Runs a test snippet** to check data shapes and distributions
4. **Writes the full script** with validated column names and proper error handling
5. **Runs the script** to verify it executes without errors
6. **Fixes any issues** from the test run (missing columns, type mismatches)

This is a huge improvement. The agent adapts to what it finds in the data. If the seed file has `avg_order_val` instead of `average_order_value`, the agent notices and uses the correct column name. If there are nulls in a feature column, the agent adds imputation logic.

But ReAct has a structural limitation for code generation: every piece of code is a string inside a tool call. The agent writes code, sends it to `run_python` as a string, reads back the output. There's no persistent execution environment. Each `run_python` call starts fresh. The agent can't build up state incrementally.

## 5. Approach 4: CodeAct Agent

This is the approach I ended up using in production. [CodeAct](https://arxiv.org/abs/2402.01030) (Wang et al., 2024) flips the tool-use model: instead of the agent calling tools through a JSON API, **the agent writes and executes code directly as its action space**. The code itself is the tool.

The key insight: for code generation tasks, the most natural "action" an agent can take is writing and running code. Instead of having a `run_python` tool that takes a code string, the agent just writes Python in a persistent Jupyter-like environment where state carries across turns.


https://gist.github.com/cmenguy/17705246501a9278037be88730ea0ecd


The agent loop is simpler than ReAct because there's only one "tool": execute code.


https://gist.github.com/cmenguy/acda1fa9abf697753c78d631018d456c



https://gist.github.com/cmenguy/d9c3b9a6feeb2e81869c8836f5b37c9e


Here's what a typical CodeAct session looks like for our marketing problem. The agent's first move is to inspect the data:


https://gist.github.com/cmenguy/6bc3bd34eff3a500a3c1f91f83cdfb3b


The agent sees the actual column names and data types. If the columns are `purch_freq`, `avg_ov`, and `days_last_purch` (as they often are in real marketing data where someone truncated names to fit a legacy system), the agent adapts.


https://gist.github.com/cmenguy/3c692d7d9da83aea670ea379efa45f09


Then it builds incrementally:


https://gist.github.com/cmenguy/3185b14aa23d40375645a3a4a117b3ff



https://gist.github.com/cmenguy/2f92bda006ef6ac702bdedeec53c26a5



https://gist.github.com/cmenguy/68def605d874fe45dec422c8ffab7a9c


The agent tests each function before moving on. If `score_users` throws an error because the seed list only has 3 users and `n_neighbors=5`, the agent sees the error, adjusts the parameter, and re-runs. This self-correction loop is the main advantage over chained prompting.

Finally, the agent writes the complete script:


https://gist.github.com/cmenguy/1e7adeb40ec08de98b864dbe20459e6f


Notice what the CodeAct agent does that the other approaches don't:

- **Discovers actual column names** from the data before writing code
- **Handles the `n_neighbors` edge case** by using `min(n_neighbors, len(seed))`
- **Adds null handling** because it saw nulls in the data
- **Uses a random seed** for reproducibility
- **Tests each function** before combining them
- **Validates column existence** with explicit error messages

## 6. Why CodeAct Beats ReAct for Code Generation

The difference between ReAct and CodeAct for this problem comes down to state management.

In ReAct, the agent passes code strings to a `run_python` tool. Each execution is isolated. If the agent defines a function in step 3, it can't call that function in step 4 unless it re-includes the entire definition. The agent ends up carrying around massive code strings, and the context window fills up fast.

In CodeAct, the environment is persistent. `import pandas as pd` in step 1 means `pd` is available in every subsequent step. The agent can build up a solution the same way a human would in a Jupyter notebook: import things, explore data, define helpers, test them, compose the final result.

Here's the practical impact:


https://gist.github.com/cmenguy/aab6f1431fca0bf9cc50506e4ae06e8a


The self-correction rate is the big one. When CodeAct runs a snippet and gets an error, the traceback appears in the same environment where all the variables are still alive. The agent can inspect the problematic dataframe, print its dtypes, and fix the issue. When ReAct gets an error, it's working blind because the execution context is gone.

## 7. Making It Production-Ready

The CodeAct agent above works for demos. For production, there are three problems to solve: safety, context, and reliability.

#### 7.1 Sandboxing Execution

Letting an LLM run arbitrary Python on your infrastructure is a terrible idea without sandboxing. The agent could `import os; os.system("rm -rf /")` or read sensitive environment variables. You need a sandbox.


https://gist.github.com/cmenguy/2bc1672daf48cc1d55edbe762f58cdd3


This is a minimal sandbox. For production, use Docker containers, gVisor, or a dedicated code execution service like [E2B](https://e2b.dev) or [Modal](https://modal.com). The subprocess approach here blocks the most obvious attacks but isn't airtight.

#### 7.2 Injecting Codebase Context

The marketing analytics team has internal libraries. There's a `cdp_client` module for querying the customer data platform, an `audience_io` module for writing audience files in the format the campaign platform expects, and a `metrics` module for computing standard marketing metrics (LTV, RFM scores, etc.).

The agent needs to know about these. Two approaches work:

**API summaries in the system prompt:**


https://gist.github.com/cmenguy/17fe5b8db76b251b92d585902143b663


**Retrieval-augmented context (for larger codebases):**


https://gist.github.com/cmenguy/e794b11830986bd791c1718ef9b15771



https://gist.github.com/cmenguy/84d71e0e5d9977d1850400a159202fe6


With codebase context, the agent generates code that uses `cdp_client.query()` instead of `pd.read_parquet()`, calls `audience_io.write_audience()` instead of `to_parquet()`, and follows the patterns it sees in the retrieved code.

#### 7.3 Reliability: Retry and Validate

LLM agents fail. They hallucinate function signatures, get stuck in loops, produce code that doesn't run. You need a reliability layer.


https://gist.github.com/cmenguy/a4d5d1bb8d79e75932dacf7b095df52d



https://gist.github.com/cmenguy/67d8764ad3541cb67496daaece6d1226


In practice, the CodeAct agent produces valid code on the first try about 85% of the time. With one retry, that goes to ~95%. The remaining 5% are usually requests that are genuinely ambiguous or require domain knowledge the agent doesn't have.

## 8. Head-to-Head: All Four Approaches

Let me put all four approaches through the same set of five marketing requests and compare results. These are real request patterns from our team:


https://gist.github.com/cmenguy/c8101a990ef29affa596eae64cba18d2


And the aggregate metrics:


https://gist.github.com/cmenguy/af3d2c12989aa1124b5d67926ae89d2c


CodeAct wins on correctness while being faster and cheaper than ReAct. The speed advantage comes from needing fewer steps (persistent state means less re-computation). The cost advantage comes from shorter context windows (no repeated code blocks).

The one place single-shot wins is latency. If you need code in 5 seconds and "good enough" is acceptable, single-shot is hard to beat. For production code that needs to actually work, the extra 25 seconds of CodeAct is worth it.

## 9. Extending to Other Domains

The marketing audience builder is one example, but the pattern works for any domain where non-engineers need to request code from a system. Here's how the CodeAct approach applies to three other cases I've seen at work:

**Data engineering:** "Create an Airflow DAG that runs the customer churn model daily, reads from the `user_events` table, writes predictions to `churn_scores`, and sends a Slack alert if the model's AUC drops below 0.75." The agent inspects the existing DAG templates in the repo, follows the team's naming conventions, and wires up the Slack notification correctly.

**Finance analytics:** "Backtest a momentum strategy on the S&P 500. Use a 12-month lookback, rebalance monthly, hold the top decile. Compare against buy-and-hold." The agent pulls price data, implements the strategy, runs the backtest, and generates a comparison chart.

**Product analytics:** "Build a retention cohort analysis. Group users by signup month, track weekly active status for 12 weeks, output a heatmap." The agent queries the events table, pivots the data into cohort format, and generates the visualization.

The common thread: a domain expert describes what they want, and the agent writes code that fits the team's stack. The CodeAct pattern handles all of these because the agent can inspect real data, test incrementally, and adapt to what it finds.

## 10. What I'd Do Differently Next Time

After running this in production for a few weeks, some lessons:

**Start with the system prompt, not the agent loop.** I spent too long tweaking the ReAct/CodeAct loop mechanics and not enough time on the system prompt that teaches the agent about our codebase. The quality of the codebase context in the prompt matters more than the agent architecture.

**Log everything.** Every agent step, every code execution, every error. When the agent generates bad code, you need to trace back through its reasoning to understand why. I built a simple logging layer that writes the full conversation to a JSONL file. It's been the most useful debugging tool.

**Let humans edit, not just approve.** The first version had a binary approve/reject flow. Users would reject code, the agent would retry, and everyone was frustrated. The current version shows the generated code in an editor where users can make small edits before running. Most of the time they change one or two lines (a parameter value, a column name) and approve. That's a much better UX than regenerating from scratch.

**Evaluation is the hard part.** Measuring whether generated code is "correct" is genuinely difficult. It's not like code completion where you can compute edit distance against a known answer. For NL2Code, there are many valid solutions. I ended up using a combination of: (1) does it parse, (2) does it run on test data, (3) does a human reviewer rate it as correct. The human review is the bottleneck.

Next up in Part 3: we flip the direction entirely. Instead of generating new code from English descriptions, we'll build a system that reads existing code, spots bugs and anti-patterns, and proposes fixes. Same agent architecture, very different evaluation problem.


---

*Originally published on [AI Terminal](https://ai-terminal.net/llm/ai-engineering/code-gen/2026/03/09/nl2code-agents.medium_tmp/).*

Tags: react, agents, nl2code, codeact, tool-use
