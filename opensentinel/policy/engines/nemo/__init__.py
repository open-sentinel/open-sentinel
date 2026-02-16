"""
NeMo Guardrails policy engine.

Integrates NVIDIA's NeMo Guardrails as a PolicyEngine for
input/output filtering, jailbreak detection, and content moderation.

Requires: pip install nemoguardrails
"""

from opensentinel.policy.engines.nemo.engine import NemoGuardrailsPolicyEngine

__all__ = ["NemoGuardrailsPolicyEngine"]
