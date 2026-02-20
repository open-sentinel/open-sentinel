# Examples

Each example is a self-contained directory with an `osentinel.yaml` config and a Python client script. Start the proxy in one terminal, run the client in another.

```
Your App  ──►  Open Sentinel (:4000)  ──►  LLM Provider
                     │
              pre_call_hook    → apply deferred interventions (μs)
              LLM call         → forwarded unmodified via LiteLLM
              post_call_hook   → policy engine evaluates async
                                 violations queued for next turn
```

Every example below triggers this pipeline. The interesting part is what the policy engine does in step 3.

---

## Quickstart — 60 seconds to running

[`examples/quickstart/`](quickstart/)

The smallest possible demo. ~20 lines of client code, 3 policy rules. Shows the judge engine evaluating responses in the background with zero critical-path latency. Start here.

```bash
cd examples/quickstart
export GEMINI_API_KEY=...    # or ANTHROPIC_API_KEY, OPENAI_API_KEY
osentinel serve              # terminal 1
python quickstart.py         # terminal 2
```

---

## Judge — Async LLM evaluation + deferred intervention

[`examples/judge/`](judge/)

A coding assistant that gets hit with a prompt injection attack. The judge engine evaluates the response asynchronously (zero latency on your call), catches the violation, and injects a system prompt amendment on the next turn. Watch the agent reassert its boundaries.

**What's happening under the hood**: The judge LLM scores responses on built-in rubrics (safety, instruction_following, tone), normalizes to [0,1], and maps the aggregate to an action based on threshold profiles (safe/balanced/aggressive).

```bash
cd examples/judge
export GEMINI_API_KEY=...
osentinel serve
python llm_judge.py
```

---

## FSM — Deterministic workflow enforcement with LTL constraints

[`examples/gemini_fsm/`](gemini_fsm/)

A customer support agent with a precedence constraint: identity verification must happen before any account action. The agent tries to process a refund without verifying — the FSM catches the violation and injects corrective guidance.

**What's happening under the hood**: Classification uses a three-tier cascade — tool call name matching (confidence 1.0, ~0ms), regex patterns (0.85, ~1ms), semantic embeddings (proportional to cosine similarity, ~50ms). First confident match wins. Constraints are evaluated as LTL-lite temporal logic over the state history.

```bash
cd examples/gemini_fsm
export GEMINI_API_KEY=...
osentinel serve
python gemini_agent.py
```

---

## NeMo Guardrails — NVIDIA's content safety rails

[`examples/nemo_guardrails/`](nemo_guardrails/)

Wraps NeMo Guardrails as a policy engine. Input rails run pre-call (jailbreak detection, PII filtering), output rails run post-call (toxicity, topical control). Fail-open by default — if NeMo throws, the request passes through with a warning.

```bash
pip install 'opensentinel[nemo]'   # extra dependency
cd examples/nemo_guardrails
export GEMINI_API_KEY=...
osentinel serve
python nemo_agent.py
```

---

## Common patterns

**Session tracking**: Pass `X-Sentinel-Session-ID` header (or set `x-sentinel-session-id` in `default_headers`) to group requests into a conversation. Without it, Open Sentinel falls back to: `metadata.session_id` → `metadata.run_id` → `user` field → `thread_id` → hash of first message → random UUID.

**Model strings**: Use [LiteLLM format](https://docs.litellm.ai/docs/providers) — `gemini/gemini-2.5-flash`, `anthropic/claude-sonnet-4-5`, `openai/gpt-4o`, etc.

**Fail-open**: All hooks are wrapped in `safe_hook()` with a 30s timeout. If a hook throws or times out, the request passes through unmodified. Only `WorkflowViolationError` (intentional hard blocks) propagates. The proxy never becomes the bottleneck.
