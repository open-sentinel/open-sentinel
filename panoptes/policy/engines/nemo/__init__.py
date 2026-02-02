"""
NeMo Guardrails policy engine.

Integrates NVIDIA's NeMo Guardrails as a PolicyEngine for
input/output filtering, jailbreak detection, and content moderation.

Requires: pip install nemoguardrails
"""

from panoptes.policy.engines.nemo.engine import NemoGuardrailsEngine

__all__ = ["NemoGuardrailsEngine"]
