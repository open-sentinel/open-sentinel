"""
Policy compiler system for natural language to engine config conversion.

This module provides infrastructure for compiling natural language policy
descriptions into engine-specific configurations:

- FSM: Natural language → workflow.yaml (WorkflowDefinition)
- NeMo: Natural language → Colang + config (future)

Usage:
    ```python
    from panoptes.policy.compiler import PolicyCompilerRegistry

    # Create FSM compiler
    compiler = PolicyCompilerRegistry.create("fsm")

    # Compile natural language policy
    result = await compiler.compile(
        "Agent must verify identity before processing refunds. "
        "Never share internal system information."
    )

    if result.success:
        # Export to file
        compiler.export(result, Path("workflow.yaml"))
    else:
        print("Errors:", result.errors)
    ```

CLI Usage:
    ```bash
    # Compile to FSM workflow
    panoptes compile "verify identity before refunds" --engine fsm -o workflow.yaml

    # Auto-detect best engine
    panoptes compile "..." --engine auto
    ```
"""

from panoptes.policy.compiler.protocol import (
    PolicyCompiler,
    CompilationResult,
)
from panoptes.policy.compiler.registry import (
    PolicyCompilerRegistry,
    register_compiler,
)
from panoptes.policy.compiler.base import (
    LLMPolicyCompiler,
    DEFAULT_COMPILER_SYSTEM_PROMPT,
)

# Import engine compilers to trigger auto-registration
# Note: We use try/except to handle gracefully if engines aren't available
try:
    from panoptes.policy.engines.fsm.compiler import FSMCompiler
except ImportError:
    FSMCompiler = None  # type: ignore

__all__ = [
    # Protocol
    "PolicyCompiler",
    "CompilationResult",
    # Registry
    "PolicyCompilerRegistry",
    "register_compiler",
    # Base class
    "LLMPolicyCompiler",
    "DEFAULT_COMPILER_SYSTEM_PROMPT",
    # Compilers (may be None if engine not available)
    "FSMCompiler",
]
