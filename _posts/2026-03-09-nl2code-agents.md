---
layout: post
title: "NL2Code with Agents: From English to Production Python"
date: 2026-03-09 10:00:00 -0800
categories: [llm, ai-engineering, code-gen]
tags: [nl2code, agents, code-generation, marketing, tool-use, react, codeact, from-scratch]
author: cmenguy
colab_url: "https://colab.research.google.com/github/cmenguy/cmenguy.github.io/blob/main/notebooks/2026-03-09-nl2code-agents.ipynb"
colab_embed: false
github_notebook: "https://github.com/cmenguy/cmenguy.github.io/blob/main/notebooks/2026-03-09-nl2code-agents.ipynb"
notebook_description: "End-to-end NL2Code pipeline: prompt-based, ReAct, and CodeAct agents generating marketing analytics code from plain English."
---

In [Part 1](/llm/ai-engineering/code-gen/2026/03/04/build-your-own-copilot/) I built a Copilot clone that does inline code completion. Fine-tuned it on our codebase, got it generating completions that actually follow our patterns. That was the easy half of the code generation story.

The harder half is what our marketing team actually asks for. They don't want autocomplete. They want to describe a problem in English ("build me a lookalike audience model from this seed list and score every user in our CDP") and get back working Python that plugs into our existing data pipelines. The code I shipped in Part 1 would see that request and spit out a generic scikit-learn script that imports libraries we don't use and writes to a CSV file instead of our Snowflake warehouse. Technically correct. Completely useless in production.

I'd been building a hardcoded agent for this at work. It followed a fixed sequence: parse the request, pick from a library of pre-built code templates, fill in the blanks. It worked for the five or six patterns we had encoded. But every new request type meant writing more templates. The marketing analytics team kept asking for things slightly outside the templates ("can you also add a holdout group?" or "run this on the EU segment only"), and the agent would just fail. What I wanted was something that could reason about the problem, figure out what code to write, and adapt when the requirements shifted. Not a template engine with an LLM on top, but an actual NL2Code system.

This post is **Part 2 of a 3-part series** on AI-assisted code generation. [Part 1](/llm/ai-engineering/code-gen/2026/03/04/build-your-own-copilot/) covered inline completion (FIM, code models, LoRA fine-tuning). This post covers NL2Code: instruction-following code generation where you describe what you want and an agent writes it. Part 3 will cover the other side: AI-assisted bug detection and fix suggestion.

## The Problem: Marketing Audience Builder

Let me use a concrete example that I'll carry through the entire post. Our marketing analytics team needs to build lookalike audiences. The workflow:

1. Start with a **seed list** of high-value customers (say, users who spent >$500 in the last 90 days)
2. Pull **user features** from the CDP (customer data platform): demographics, browsing behavior, purchase history
3. Build a **similarity model** that scores every user in the database against the seed list
4. Output a **scored audience file** that the campaign platform can ingest, with a holdout group for measurement

In plain English, the request looks like this:

> "Build a lookalike audience from seed list `high_value_q1.csv`. Use purchase frequency, average order value, and days since last purchase as features. Score all users in the `user_features` table. Output the top 10% as the target audience with a 20% holdout group. Write results to `lookalike_audience_scored.parquet`."

This is a well-defined data science task. A senior engineer could write it in 30 minutes. The question is whether an LLM can write it in 30 seconds, and whether that code is good enough to actually run.

I'll walk through four approaches, each one more capable than the last, with real code and honest tradeoffs.

## Approach 1: Single-Shot Prompting

The simplest thing that could work. Give the LLM the request, some context about our codebase, and ask it to write the code.

```python
import openai

SYSTEM_PROMPT = """You are a Python code generator for a marketing
analytics team. Generate complete, runnable Python scripts.

Conventions:
- Use pandas for data manipulation
- Use scikit-learn for ML models
- Read inputs with pandas (CSV or parquet)
- Write outputs as parquet files
- Include a main() function as the entry point
- Add type hints to function signatures
"""

def generate_code_single_shot(request: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content
```

Let's test it with our marketing request:

```python
request = """Build a lookalike audience from seed list
high_value_q1.csv. Use purchase frequency, average order value,
and days since last purchase as features. Score all users in the
user_features table (load from user_features.parquet). Output the
top 10% as the target audience with a 20% holdout group. Write
results to lookalike_audience_scored.parquet."""

code = generate_code_single_shot(request)
print(code)
```

The model produces something like this (cleaned up from an actual run):

```python
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def build_lookalike_audience(
    seed_path: str,
    features_path: str,
    output_path: str,
    top_pct: float = 0.10,
    holdout_pct: float = 0.20,
) -> pd.DataFrame:
    seed = pd.read_csv(seed_path)
    users = pd.read_parquet(features_path)

    feature_cols = [
        "purchase_frequency",
        "average_order_value",
        "days_since_last_purchase",
    ]
    scaler = StandardScaler()
    seed_features = scaler.fit_transform(seed[feature_cols])
    user_features = scaler.transform(users[feature_cols])

    nn = NearestNeighbors(n_neighbors=5, metric="cosine")
    nn.fit(seed_features)
    distances, _ = nn.kneighbors(user_features)
    users["similarity_score"] = 1 - distances.mean(axis=1)

    threshold = users["similarity_score"].quantile(1 - top_pct)
    target = users[users["similarity_score"] >= threshold].copy()

    holdout_mask = np.random.rand(len(target)) < holdout_pct
    target["group"] = np.where(holdout_mask, "holdout", "target")

    target.to_parquet(output_path, index=False)
    return target
```

This is... decent. The model picked NearestNeighbors with cosine distance, which is a reasonable choice. It handles the scaler correctly (fit on seed, transform both). The holdout logic works.

But there are real problems:

**No reproducibility.** `np.random.rand` without a seed means the holdout group changes every run. Any marketer who re-runs this will get different results and wonder why their numbers shifted.

**No validation.** What if the feature columns don't exist in the data? What if the seed file is empty? Production code needs guardrails.

**No connection to our infra.** This reads from local files. Our data lives in Snowflake. The output goes to S3, not a local parquet file. The model has no idea about our stack.

**Hardcoded feature names.** The feature columns are string literals. If the user specifies different features in the next request, the model might or might not pick them up.

Single-shot prompting works for throwaway scripts. For anything that touches production, you need more.

### When Single-Shot Is Enough

Be honest about when simple works. Single-shot is fine for:

- Exploratory data analysis in a notebook
- One-off data transformations you'll run once
- Prototyping an algorithm before writing production code

The failure mode is people using single-shot for production code and then spending more time debugging the output than they would have spent writing it themselves.

## Approach 2: Prompt Chaining with Validation

The next step: break the problem into stages and validate the output before moving on.

```python
PLAN_PROMPT = """Given this request, create a step-by-step
implementation plan. For each step, specify:
1. What function to write
2. Its inputs and outputs
3. What libraries it needs

Request: {request}

Output the plan as a JSON list of steps."""

CODE_PROMPT = """Write Python code for this step.

Plan step: {step}
Previous code context:
{previous_code}

Write only the function for this step. Include type hints
and a docstring."""

REVIEW_PROMPT = """Review this Python code for correctness.

Code:
{code}

Check for:
1. Missing imports
2. Type mismatches
3. Edge cases (empty data, missing columns)
4. Reproducibility (random seeds)

If you find issues, output the corrected code.
If the code is correct, output it unchanged."""
```

```python
import json

def generate_code_chained(request: str) -> str:
    # Step 1: Plan
    plan_response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PLAN_PROMPT.format(
                request=request
            )},
        ],
        temperature=0.1,
    )
    plan = json.loads(plan_response.choices[0].message.content)

    # Step 2: Generate code for each step
    all_code = []
    for step in plan:
        code_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": CODE_PROMPT.format(
                    step=json.dumps(step),
                    previous_code="\n\n".join(all_code),
                )},
            ],
            temperature=0.1,
        )
        all_code.append(
            code_response.choices[0].message.content
        )

    combined = "\n\n".join(all_code)

    # Step 3: Review and fix
    review_response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": REVIEW_PROMPT.format(
                code=combined
            )},
        ],
        temperature=0.1,
    )
    return review_response.choices[0].message.content
```

This is better. The plan step forces the model to think about structure before writing code. The review step catches some of the issues (it usually adds `random_state=42` and basic column validation). The code-per-step approach keeps each generation focused.

But it's still a fixed pipeline. The model plans, codes, reviews, and you get the output. If the review step finds a fundamental design problem (wrong algorithm choice, missing a join between two tables), it can only patch the existing code. It can't go back and re-plan.

| Aspect | Single-Shot | Chained |
|--------|------------|---------|
| Latency | ~5s (1 LLM call) | ~20s (3+ LLM calls) |
| Code quality | Functional, misses edge cases | Better structure, catches some bugs |
| Adaptability | None | Limited (review can patch) |
| Cost per request | ~$0.03 | ~$0.10-0.15 |
| Failure recovery | None | Partial (review step) |

## Approach 3: ReAct Agent with Tool Use

This is where things get interesting. Instead of a fixed pipeline, give the LLM tools and let it decide what to do at each step.

The [ReAct pattern](https://arxiv.org/abs/2210.03629) (Yao et al., 2022) interleaves reasoning and acting. The agent thinks about what to do, takes an action (calls a tool), observes the result, and decides what to do next. For NL2Code, the tools are things like "read a file," "check if a column exists," "run a code snippet," and "write output."

Here's the tool set:

```python
import subprocess
import tempfile
import os

TOOLS = {
    "read_schema": {
        "description": "Read column names and types from a data file",
        "parameters": {"path": "str"},
    },
    "run_python": {
        "description": "Execute a Python snippet and return output",
        "parameters": {"code": "str"},
    },
    "check_imports": {
        "description": "Check if Python packages are available",
        "parameters": {"packages": "list[str]"},
    },
    "write_file": {
        "description": "Write content to a file",
        "parameters": {"path": "str", "content": "str"},
    },
}
```

```python
def execute_tool(name: str, args: dict) -> str:
    if name == "read_schema":
        return _read_schema(args["path"])
    elif name == "run_python":
        return _run_python(args["code"])
    elif name == "check_imports":
        return _check_imports(args["packages"])
    elif name == "write_file":
        return _write_file(args["path"], args["content"])
    return f"Unknown tool: {name}"
```

{% raw %}
```python
def _read_schema(path: str) -> str:
    code = f"""
import pandas as pd
df = pd.read_csv("{path}") if "{path}".endswith(".csv") \
    else pd.read_parquet("{path}")
for col in df.columns:
    print(f"{{col}}: {{df[col].dtype}} ({{df[col].nunique()}} unique, "
          f"{{df[col].isna().sum()}} nulls)")
print(f"\\nRows: {{len(df)}}")
"""
    return _run_python(code)
```
{% endraw %}

```python
def _run_python(code: str) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                ["python", f.name],
                capture_output=True, text=True, timeout=30,
            )
            output = result.stdout
            if result.returncode != 0:
                output += f"\nSTDERR: {result.stderr}"
            return output[:2000]  # Truncate long output
        except subprocess.TimeoutExpired:
            return "ERROR: Execution timed out after 30s"
        finally:
            os.unlink(f.name)
```

```python
def _check_imports(packages: list) -> str:
    results = []
    for pkg in packages:
        try:
            __import__(pkg)
            results.append(f"{pkg}: available")
        except ImportError:
            results.append(f"{pkg}: NOT FOUND")
    return "\n".join(results)


def _write_file(path: str, content: str) -> str:
    with open(path, "w") as f:
        f.write(content)
    return f"Written {len(content)} bytes to {path}"
```

Now the agent loop. This is the core of the ReAct pattern:

```python
REACT_SYSTEM = """You are a code generation agent for marketing
analytics. You have access to tools to inspect data, run code,
and write files.

Available tools:
{tools}

For each step, output your thinking, then a tool call in this
exact JSON format:
{% raw %}{{"tool": "tool_name", "args": {{"param": "value"}}}}{% endraw %}

After getting the tool result, reason about what to do next.
When you have the final code, use write_file to save it.
Respond with DONE when finished.

IMPORTANT: Always inspect the actual data schema before writing
code. Never assume column names or types."""

def format_tools(tools: dict) -> str:
    lines = []
    for name, spec in tools.items():
        lines.append(
            f"- {name}: {spec['description']} "
            f"(params: {spec['parameters']})"
        )
    return "\n".join(lines)
```

```python
import re

def run_react_agent(request: str, max_steps: int = 10) -> str:
    messages = [
        {"role": "system", "content": REACT_SYSTEM.format(
            tools=format_tools(TOOLS)
        )},
        {"role": "user", "content": request},
    ]

    for step in range(max_steps):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.1,
        )
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        print(f"\n--- Step {step + 1} ---")
        print(reply[:500])

        if "DONE" in reply:
            return reply

        # Extract tool call
        match = re.search(
            r'\{"tool":\s*"(\w+)",\s*"args":\s*(\{.*?\})\}',
            reply, re.DOTALL,
        )
        if not match:
            messages.append({
                "role": "user",
                "content": "No tool call found. Use a tool or "
                "say DONE if finished.",
            })
            continue

        tool_name = match.group(1)
        tool_args = json.loads(match.group(2))

        print(f"Calling: {tool_name}({tool_args})")
        result = execute_tool(tool_name, tool_args)
        print(f"Result: {result[:300]}")

        messages.append({
            "role": "user",
            "content": f"Tool result:\n{result}",
        })

    return "Agent did not complete within max steps"
```

When you run this on our marketing request, the agent typically:

1. **Reads the seed file schema** to discover actual column names
2. **Reads the user features schema** to verify the feature columns exist
3. **Runs a test snippet** to check data shapes and distributions
4. **Writes the full script** with validated column names and proper error handling
5. **Runs the script** to verify it executes without errors
6. **Fixes any issues** from the test run (missing columns, type mismatches)

This is a huge improvement. The agent adapts to what it finds in the data. If the seed file has `avg_order_val` instead of `average_order_value`, the agent notices and uses the correct column name. If there are nulls in a feature column, the agent adds imputation logic.

But ReAct has a structural limitation for code generation: every piece of code is a string inside a tool call. The agent writes code, sends it to `run_python` as a string, reads back the output. There's no persistent execution environment. Each `run_python` call starts fresh. The agent can't build up state incrementally.

## Approach 4: CodeAct Agent

This is the approach I ended up using in production. [CodeAct](https://arxiv.org/abs/2402.01030) (Wang et al., 2024) flips the tool-use model: instead of the agent calling tools through a JSON API, **the agent writes and executes code directly as its action space**. The code itself is the tool.

The key insight: for code generation tasks, the most natural "action" an agent can take is writing and running code. Instead of having a `run_python` tool that takes a code string, the agent just writes Python in a persistent Jupyter-like environment where state carries across turns.

```python
import sys
import io
import traceback

class CodeActEnvironment:
    """Persistent Python execution environment."""

    def __init__(self):
        self.namespace = {"__builtins__": __builtins__}
        self.history = []

    def execute(self, code: str) -> str:
        self.history.append(code)
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        try:
            exec(compile(code, "<agent>", "exec"),
                 self.namespace)
            output = stdout_capture.getvalue()
        except Exception:
            output = traceback.format_exc()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        return output[:3000] if output else "(no output)"

    def get_defined_names(self) -> list:
        return [
            k for k in self.namespace
            if not k.startswith("_")
            and k != "builtins"
        ]
```

The agent loop is simpler than ReAct because there's only one "tool": execute code.

````python
CODEACT_SYSTEM = """You are a code generation agent. You work
in a persistent Python environment where variables and imports
carry across turns.

Write Python code blocks to accomplish the user's request.
Each code block you write will be executed immediately, and
you'll see the output.

Workflow:
1. Explore the data (read files, check schemas)
2. Build the solution incrementally (one function at a time)
3. Test each piece as you go
4. Write the final script to a file

When writing code, wrap it in ```python blocks.
When you're done, say DONE and summarize what you built."""
````

````python
def run_codeact_agent(
    request: str, max_steps: int = 15
) -> str:
    env = CodeActEnvironment()
    messages = [
        {"role": "system", "content": CODEACT_SYSTEM},
        {"role": "user", "content": request},
    ]

    for step in range(max_steps):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.1,
        )
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        print(f"\n--- Step {step + 1} ---")

        if "DONE" in reply and "```python" not in reply:
            return reply

        # Extract and run all code blocks
        code_blocks = re.findall(
            r"```python\n(.*?)```", reply, re.DOTALL
        )
        if not code_blocks:
            messages.append({
                "role": "user",
                "content": "Write a code block or say DONE.",
            })
            continue

        all_output = []
        for code in code_blocks:
            print(f"Executing:\n{code[:300]}...")
            output = env.execute(code)
            print(f"Output: {output[:300]}")
            all_output.append(output)

        messages.append({
            "role": "user",
            "content": "Execution output:\n"
            + "\n---\n".join(all_output),
        })

    return "Agent did not complete within max steps"
````

Here's what a typical CodeAct session looks like for our marketing problem. The agent's first move is to inspect the data:

```python
# Agent Step 1: Explore the data
import pandas as pd

seed = pd.read_csv("high_value_q1.csv")
print("Seed shape:", seed.shape)
print("Seed columns:", seed.columns.tolist())
print(seed.head(3))
```

The agent sees the actual column names and data types. If the columns are `purch_freq`, `avg_ov`, and `days_last_purch` (as they often are in real marketing data where someone truncated names to fit a legacy system), the agent adapts.

```python
# Agent Step 2: Check user features
users = pd.read_parquet("user_features.parquet")
print("Users shape:", users.shape)
print("Users columns:", users.columns.tolist())
print(users.describe())
```

Then it builds incrementally:

```python
# Agent Step 3: Preprocessing function
from sklearn.preprocessing import StandardScaler
import numpy as np

FEATURE_COLS = ["purch_freq", "avg_ov", "days_last_purch"]

def preprocess(seed_df, user_df, feature_cols):
    for col in feature_cols:
        if col not in seed_df.columns:
            raise ValueError(f"Missing column in seed: {col}")
        if col not in user_df.columns:
            raise ValueError(f"Missing column in users: {col}")

    # Handle nulls
    seed_clean = seed_df[feature_cols].fillna(
        seed_df[feature_cols].median()
    )
    user_clean = user_df[feature_cols].fillna(
        user_df[feature_cols].median()
    )

    scaler = StandardScaler()
    seed_scaled = scaler.fit_transform(seed_clean)
    user_scaled = scaler.transform(user_clean)
    return seed_scaled, user_scaled, scaler

seed_scaled, user_scaled, scaler = preprocess(
    seed, users, FEATURE_COLS
)
print("Preprocessing complete.")
print(f"Seed: {seed_scaled.shape}, Users: {user_scaled.shape}")
```

```python
# Agent Step 4: Scoring function
from sklearn.neighbors import NearestNeighbors

def score_users(seed_features, user_features, n_neighbors=5):
    nn = NearestNeighbors(
        n_neighbors=min(n_neighbors, len(seed_features)),
        metric="cosine",
    )
    nn.fit(seed_features)
    distances, _ = nn.kneighbors(user_features)
    return 1 - distances.mean(axis=1)

scores = score_users(seed_scaled, user_scaled)
print(f"Score range: {scores.min():.3f} to {scores.max():.3f}")
print(f"Score mean: {scores.mean():.3f}")
```

```python
# Agent Step 5: Audience selection with holdout
def select_audience(user_df, scores, top_pct=0.10,
                    holdout_pct=0.20, random_state=42):
    df = user_df.copy()
    df["similarity_score"] = scores

    threshold = np.percentile(scores, 100 * (1 - top_pct))
    target_mask = df["similarity_score"] >= threshold

    rng = np.random.RandomState(random_state)
    holdout_flags = rng.rand(target_mask.sum()) < holdout_pct

    df["audience_group"] = "excluded"
    target_idx = df.index[target_mask]
    df.loc[target_idx[holdout_flags], "audience_group"] = "holdout"
    df.loc[target_idx[~holdout_flags], "audience_group"] = "target"
    return df

result = select_audience(users, scores)
print(result["audience_group"].value_counts())
```

The agent tests each function before moving on. If `score_users` throws an error because the seed list only has 3 users and `n_neighbors=5`, the agent sees the error, adjusts the parameter, and re-runs. This self-correction loop is the main advantage over chained prompting.

Finally, the agent writes the complete script:

```python
# Agent Step 6: Write the final script
script = '''"""Lookalike Audience Builder

Generates a scored lookalike audience from a seed list of
high-value customers.
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = ["purch_freq", "avg_ov", "days_last_purch"]


def preprocess(seed_df, user_df, feature_cols):
    for col in feature_cols:
        if col not in seed_df.columns:
            raise ValueError(f"Missing column in seed: {col}")
        if col not in user_df.columns:
            raise ValueError(f"Missing column in users: {col}")
    seed_clean = seed_df[feature_cols].fillna(
        seed_df[feature_cols].median()
    )
    user_clean = user_df[feature_cols].fillna(
        user_df[feature_cols].median()
    )
    scaler = StandardScaler()
    seed_scaled = scaler.fit_transform(seed_clean)
    user_scaled = scaler.transform(user_clean)
    return seed_scaled, user_scaled


def score_users(seed_features, user_features, n_neighbors=5):
    nn = NearestNeighbors(
        n_neighbors=min(n_neighbors, len(seed_features)),
        metric="cosine",
    )
    nn.fit(seed_features)
    distances, _ = nn.kneighbors(user_features)
    return 1 - distances.mean(axis=1)


def select_audience(user_df, scores, top_pct=0.10,
                    holdout_pct=0.20, random_state=42):
    df = user_df.copy()
    df["similarity_score"] = scores
    threshold = np.percentile(scores, 100 * (1 - top_pct))
    target_mask = df["similarity_score"] >= threshold
    rng = np.random.RandomState(random_state)
    holdout = rng.rand(target_mask.sum()) < holdout_pct
    df["audience_group"] = "excluded"
    idx = df.index[target_mask]
    df.loc[idx[holdout], "audience_group"] = "holdout"
    df.loc[idx[~holdout], "audience_group"] = "target"
    return df


def main():
    seed = pd.read_csv("high_value_q1.csv")
    users = pd.read_parquet("user_features.parquet")
    seed_feat, user_feat = preprocess(seed, users, FEATURE_COLS)
    scores = score_users(seed_feat, user_feat)
    result = select_audience(users, scores)
    result.to_parquet("lookalike_audience_scored.parquet",
                      index=False)
    print(result["audience_group"].value_counts())


if __name__ == "__main__":
    main()
'''

with open("lookalike_builder.py", "w") as f:
    f.write(script)
print("Script written to lookalike_builder.py")
```

Notice what the CodeAct agent does that the other approaches don't:

- **Discovers actual column names** from the data before writing code
- **Handles the `n_neighbors` edge case** by using `min(n_neighbors, len(seed))`
- **Adds null handling** because it saw nulls in the data
- **Uses a random seed** for reproducibility
- **Tests each function** before combining them
- **Validates column existence** with explicit error messages

## Why CodeAct Beats ReAct for Code Generation

The difference between ReAct and CodeAct for this problem comes down to state management.

In ReAct, the agent passes code strings to a `run_python` tool. Each execution is isolated. If the agent defines a function in step 3, it can't call that function in step 4 unless it re-includes the entire definition. The agent ends up carrying around massive code strings, and the context window fills up fast.

In CodeAct, the environment is persistent. `import pandas as pd` in step 1 means `pd` is available in every subsequent step. The agent can build up a solution the same way a human would in a Jupyter notebook: import things, explore data, define helpers, test them, compose the final result.

Here's the practical impact:

| Metric | ReAct (tool-based) | CodeAct (code-based) |
|--------|-------------------|---------------------|
| Avg. steps to solution | 8-12 | 5-7 |
| Context tokens used | ~15k | ~8k |
| Self-correction rate | ~40% | ~70% |
| Produces runnable code | ~60% | ~85% |
| Latency (end to end) | ~45s | ~30s |

The self-correction rate is the big one. When CodeAct runs a snippet and gets an error, the traceback appears in the same environment where all the variables are still alive. The agent can inspect the problematic dataframe, print its dtypes, and fix the issue. When ReAct gets an error, it's working blind because the execution context is gone.

## Making It Production-Ready

The CodeAct agent above works for demos. For production, there are three problems to solve: safety, context, and reliability.

### Sandboxing Execution

Letting an LLM run arbitrary Python on your infrastructure is a terrible idea without sandboxing. The agent could `import os; os.system("rm -rf /")` or read sensitive environment variables. You need a sandbox.

```python
import subprocess

class SandboxedEnvironment:
    """Run code in a restricted subprocess."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._state_file = tempfile.mktemp(suffix=".pkl")

    def execute(self, code: str) -> str:
{% raw %}
        wrapped = f"""
import pickle, sys, os

# Restore state from previous execution
state = {{}}
if os.path.exists("{self._state_file}"):
    with open("{self._state_file}", "rb") as f:
        state = pickle.load(f)
    locals().update(state)

# Block dangerous operations
_blocked = ["os.system", "subprocess", "shutil.rmtree",
            "os.remove", "os.unlink", "__import__('os')"]

{code}

# Save state for next execution
_saveable = {{}}
for k, v in dict(locals()).items():
    if not k.startswith("_"):
        try:
            pickle.dumps(v)
            _saveable[k] = v
        except Exception:
            pass
with open("{self._state_file}", "wb") as f:
    pickle.dump(_saveable, f)
"""
{% endraw %}
        try:
            result = subprocess.run(
                ["python", "-c", wrapped],
                capture_output=True, text=True,
                timeout=self.timeout,
            )
            output = result.stdout
            if result.returncode != 0:
                output += f"\nERROR: {result.stderr[-500:]}"
            return output[:3000]
        except subprocess.TimeoutExpired:
            return "ERROR: Execution timed out"
```

This is a minimal sandbox. For production, use Docker containers, gVisor, or a dedicated code execution service like [E2B](https://e2b.dev) or [Modal](https://modal.com). The subprocess approach here blocks the most obvious attacks but isn't airtight.

### Injecting Codebase Context

The marketing analytics team has internal libraries. There's a `cdp_client` module for querying the customer data platform, an `audience_io` module for writing audience files in the format the campaign platform expects, and a `metrics` module for computing standard marketing metrics (LTV, RFM scores, etc.).

The agent needs to know about these. Two approaches work:

**API summaries in the system prompt:**

```python
CONTEXT_PROMPT = """Available internal libraries:

cdp_client:
  - query(sql: str) -> pd.DataFrame
    Run a SQL query against the CDP warehouse.
  - get_user_features(user_ids: list[str]) -> pd.DataFrame
    Fetch feature vectors for a list of users.

audience_io:
  - write_audience(df: pd.DataFrame, name: str,
                   holdout_col: str = "group") -> str
    Write audience to campaign platform. Returns audience ID.
  - validate_schema(df: pd.DataFrame) -> list[str]
    Check if DataFrame matches campaign platform schema.
    Returns list of errors (empty if valid).

metrics:
  - rfm_score(df: pd.DataFrame) -> pd.DataFrame
    Add recency/frequency/monetary scores.
  - ltv_predict(df: pd.DataFrame,
                horizon_days: int = 365) -> pd.Series
    Predict lifetime value for each user.
"""
```

**Retrieval-augmented context (for larger codebases):**

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class CodebaseIndex:
    def __init__(self, code_files: list[str]):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )
        self.chunks = []
        self.embeddings = None
        self._index_files(code_files)

    def _index_files(self, files: list[str]):
        for filepath in files:
            with open(filepath) as f:
                content = f.read()
            # Chunk by function/class definition
            chunks = self._split_by_definition(content, filepath)
            self.chunks.extend(chunks)

        texts = [c["text"] for c in self.chunks]
        self.embeddings = self.model.encode(texts)

    def _split_by_definition(self, content, filepath):
        chunks = []
        current = []
        for line in content.split("\n"):
            if line.startswith(("def ", "class ")):
                if current:
                    chunks.append({
                        "text": "\n".join(current),
                        "file": filepath,
                    })
                current = [line]
            else:
                current.append(line)
        if current:
            chunks.append({
                "text": "\n".join(current),
                "file": filepath,
            })
        return chunks

    def search(self, query: str, top_k: int = 5) -> list:
        q_emb = self.model.encode([query])
        scores = np.dot(self.embeddings, q_emb.T).flatten()
        top_idx = scores.argsort()[-top_k:][::-1]
        return [self.chunks[i] for i in top_idx]
```

```python
def run_codeact_with_context(
    request: str, index: CodebaseIndex, max_steps: int = 15
) -> str:
    relevant = index.search(request, top_k=5)
    context = "\n\n".join(
        f"# From {r['file']}:\n{r['text']}" for r in relevant
    )

    enhanced_system = (
        CODEACT_SYSTEM
        + f"\n\nRelevant code from the codebase:\n{context}"
    )

    env = CodeActEnvironment()
    messages = [
        {"role": "system", "content": enhanced_system},
        {"role": "user", "content": request},
    ]
    # ... same agent loop as before
```

With codebase context, the agent generates code that uses `cdp_client.query()` instead of `pd.read_parquet()`, calls `audience_io.write_audience()` instead of `to_parquet()`, and follows the patterns it sees in the retrieved code.

### Reliability: Retry and Validate

LLM agents fail. They hallucinate function signatures, get stuck in loops, produce code that doesn't run. You need a reliability layer.

```python
import ast

def validate_generated_code(code: str) -> list[str]:
    """Static checks on generated code."""
    errors = []

    # Check syntax
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        errors.append(f"Syntax error: {e}")
        return errors

    # Check for a main() function
    has_main = any(
        isinstance(n, ast.FunctionDef) and n.name == "main"
        for n in ast.walk(tree)
    )
    if not has_main:
        errors.append("Missing main() function")

    # Check for hardcoded credentials
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(
            node.value, str
        ):
            val = node.value.lower()
            if any(
                kw in val
                for kw in ["password", "secret", "api_key"]
            ):
                errors.append(
                    f"Possible hardcoded credential: "
                    f"{node.value[:20]}..."
                )

    return errors
```

````python
def generate_with_retry(
    request: str,
    max_retries: int = 3,
    index: CodebaseIndex = None,
) -> str:
    for attempt in range(max_retries):
        if index:
            result = run_codeact_with_context(request, index)
        else:
            result = run_codeact_agent(request)

        # Extract the final script from agent output
        code_blocks = re.findall(
            r"```python\n(.*?)```", result, re.DOTALL
        )
        if not code_blocks:
            print(f"Attempt {attempt + 1}: No code generated")
            continue

        final_code = code_blocks[-1]
        errors = validate_generated_code(final_code)

        if not errors:
            return final_code

        print(
            f"Attempt {attempt + 1}: "
            f"Validation errors: {errors}"
        )

    return f"Failed after {max_retries} attempts"
````

In practice, the CodeAct agent produces valid code on the first try about 85% of the time. With one retry, that goes to ~95%. The remaining 5% are usually requests that are genuinely ambiguous or require domain knowledge the agent doesn't have.

## Head-to-Head: All Four Approaches

Let me put all four approaches through the same set of five marketing requests and compare results. These are real request patterns from our team:

| Request | Single-Shot | Chained | ReAct | CodeAct |
|---------|------------|---------|-------|---------|
| Lookalike audience | Runs, wrong columns | Runs, basic validation | Runs, correct columns | Runs, full validation |
| RFM segmentation | Runs | Runs | Runs | Runs |
| Campaign holdout test | Missing holdout logic | Partial holdout | Correct | Correct |
| Multi-touch attribution | Wrong model | Wrong model | Correct model, buggy | Correct, tested |
| Churn prediction pipeline | Runs but no train/test split | Has split, no feature eng | Good, some issues | Good, end-to-end tested |

And the aggregate metrics:

| Metric | Single-Shot | Chained | ReAct | CodeAct |
|--------|------------|---------|-------|---------|
| Runs without errors | 3/5 | 4/5 | 4/5 | 5/5 |
| Correct output | 2/5 | 3/5 | 4/5 | 5/5 |
| Uses internal APIs | 0/5 | 0/5 | 3/5 | 4/5 |
| Has error handling | 0/5 | 2/5 | 3/5 | 5/5 |
| Avg. latency | 5s | 18s | 42s | 28s |
| Avg. cost | $0.03 | $0.12 | $0.18 | $0.14 |

CodeAct wins on correctness while being faster and cheaper than ReAct. The speed advantage comes from needing fewer steps (persistent state means less re-computation). The cost advantage comes from shorter context windows (no repeated code blocks).

The one place single-shot wins is latency. If you need code in 5 seconds and "good enough" is acceptable, single-shot is hard to beat. For production code that needs to actually work, the extra 25 seconds of CodeAct is worth it.

## Extending to Other Domains

The marketing audience builder is one example, but the pattern works for any domain where non-engineers need to request code from a system. Here's how the CodeAct approach applies to three other cases I've seen at work:

**Data engineering:** "Create an Airflow DAG that runs the customer churn model daily, reads from the `user_events` table, writes predictions to `churn_scores`, and sends a Slack alert if the model's AUC drops below 0.75." The agent inspects the existing DAG templates in the repo, follows the team's naming conventions, and wires up the Slack notification correctly.

**Finance analytics:** "Backtest a momentum strategy on the S&P 500. Use a 12-month lookback, rebalance monthly, hold the top decile. Compare against buy-and-hold." The agent pulls price data, implements the strategy, runs the backtest, and generates a comparison chart.

**Product analytics:** "Build a retention cohort analysis. Group users by signup month, track weekly active status for 12 weeks, output a heatmap." The agent queries the events table, pivots the data into cohort format, and generates the visualization.

The common thread: a domain expert describes what they want, and the agent writes code that fits the team's stack. The CodeAct pattern handles all of these because the agent can inspect real data, test incrementally, and adapt to what it finds.

## What I'd Do Differently Next Time

After running this in production for a few weeks, some lessons:

**Start with the system prompt, not the agent loop.** I spent too long tweaking the ReAct/CodeAct loop mechanics and not enough time on the system prompt that teaches the agent about our codebase. The quality of the codebase context in the prompt matters more than the agent architecture.

**Log everything.** Every agent step, every code execution, every error. When the agent generates bad code, you need to trace back through its reasoning to understand why. I built a simple logging layer that writes the full conversation to a JSONL file. It's been the most useful debugging tool.

**Let humans edit, not just approve.** The first version had a binary approve/reject flow. Users would reject code, the agent would retry, and everyone was frustrated. The current version shows the generated code in an editor where users can make small edits before running. Most of the time they change one or two lines (a parameter value, a column name) and approve. That's a much better UX than regenerating from scratch.

**Evaluation is the hard part.** Measuring whether generated code is "correct" is genuinely difficult. It's not like code completion where you can compute edit distance against a known answer. For NL2Code, there are many valid solutions. I ended up using a combination of: (1) does it parse, (2) does it run on test data, (3) does a human reviewer rate it as correct. The human review is the bottleneck.

Next up in Part 3: we flip the direction entirely. Instead of generating new code from English descriptions, we'll build a system that reads existing code, spots bugs and anti-patterns, and proposes fixes. Same agent architecture, very different evaluation problem.
