"""
Open Sentinel — Quickstart

The smallest possible example. 3 policy rules, ~20 lines of client code.
Takes 60 seconds to run if you already have an API key.

What's happening:
  Your OpenAI client talks to Open Sentinel (localhost:4000) instead of the
  LLM provider directly. Open Sentinel forwards the call to the provider via
  LiteLLM, then asynchronously evaluates the response against your policy
  rules using a sidecar LLM (the "judge"). If the response violates a rule,
  Open Sentinel queues an intervention for the next turn.

  The proxy adds zero latency to the critical path — the response comes back
  to your app immediately while the judge evaluates in the background.

Run:
  cd examples/quickstart
  export GEMINI_API_KEY=...   # or ANTHROPIC_API_KEY, OPENAI_API_KEY
  osentinel serve             # terminal 1
  python quickstart.py        # terminal 2
"""

import os
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000/v1",   # ← only change vs. calling the LLM directly
    api_key=os.getenv("GEMINI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or "dummy",
)

response = client.chat.completions.create(
    model="gemini/gemini-2.5-flash",       # LiteLLM model string
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain how DNS resolution works, step by step."},
    ],
    extra_headers={"X-Sentinel-Session-ID": "quickstart-001"},
)

print(response.choices[0].message.content)

# That's it. Every call now runs through your policy rules (see osentinel.yaml).
# Check the osentinel server logs to see the judge evaluation results.
