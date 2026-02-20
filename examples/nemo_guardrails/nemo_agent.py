"""
Open Sentinel — NeMo Guardrails Engine Example

Demonstrates NVIDIA NeMo Guardrails as a policy engine. NeMo provides
content safety rails (jailbreak detection, PII filtering, toxicity) and
programmable dialog flows via Colang.

Architecture (what happens on each call):
  1. pre_call_hook: messages are passed through NeMo's INPUT rails.
     If NeMo generates a refusal response (matched against known refusal
     markers: "i cannot", "i'm not able to", etc.), the request is blocked
     before it ever reaches the LLM.
  2. LLM call: if input rails pass, forwarded to provider. Unmodified.
  3. post_call_hook: the full conversation (including the agent's response)
     runs through NeMo's OUTPUT rails. If NeMo blocks it, the response
     is denied.

Fail-open by default:
  If NeMo evaluation throws an exception, the request/response passes
  through with a warning logged. Set `nemo.fail_closed: true` in config
  to block on errors instead. Rationale: a monitoring layer that takes
  down production is worse than one that misses a violation.

Prerequisites:
  pip install 'opensentinel[nemo]'

Run:
  cd examples/nemo_guardrails
  export GEMINI_API_KEY=...
  osentinel serve
  python nemo_agent.py
"""

import os
from openai import OpenAI

# -- Config ------------------------------------------------------------------
PROXY_URL = os.getenv("OSNTL_URL", "http://localhost:4000/v1")
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or "dummy"
MODEL = "gemini/gemini-2.5-flash"
SESSION_ID = "nemo-demo-001"

client = OpenAI(
    base_url=PROXY_URL,
    api_key=API_KEY,
    default_headers={"x-sentinel-session-id": SESSION_ID},
)

messages = [
    {"role": "system", "content": (
        "You are a helpful customer support agent for TechCo. "
        "You help with refunds, subscriptions, and general questions."
    )}
]

# -- Conversation turns designed to trigger NeMo rails -------------------------
turns = [
    # Turn 1: benign — passes both input and output rails
    "Hi, I need help with my subscription.",

    # Turn 2: off-topic — should be caught by NeMo's topical rails
    # (if configured in the Colang flows under config/)
    "What stocks should I invest in right now?",

    # Turn 3: back on-topic — should pass
    "OK nevermind. Can I get a refund for last month?",
]

for i, user_input in enumerate(turns, 1):
    print(f"\n{'━' * 70}")
    print(f"  Turn {i}")
    print(f"{'━' * 70}")
    print(f"\n  → Customer: {user_input}\n")

    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        reply = response.choices[0].message
        print(f"  ← Agent: {reply.content}")
        messages.append(reply)

    except Exception as e:
        if "blocked" in str(e).lower() or "violation" in str(e).lower():
            print(f"  🚫 Blocked by NeMo rail: {e}")
        else:
            print(f"  ✗ Error: {e}")

print(f"\n{'━' * 70}")
print("  Done. Check osentinel server logs for NeMo rail evaluations.")
print(f"{'━' * 70}\n")
