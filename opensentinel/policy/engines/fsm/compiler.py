"""
FSM Policy Compiler - Natural Language to WorkflowDefinition.

Converts natural language policy descriptions into FSM workflow YAML
that can be used with the FSMPolicyEngine.

Example:
    ```python
    compiler = FSMCompiler()
    result = await compiler.compile(
        "Agent must verify identity before processing refunds. "
        "Never share internal system information."
    )

    if result.success:
        compiler.export(result, Path("workflow.yaml"))
    ```
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from opensentinel.policy.compiler.base import LLMPolicyCompiler
from opensentinel.policy.compiler.protocol import CompilationResult
from opensentinel.policy.compiler.registry import register_compiler
from opensentinel.policy.engines.fsm.workflow.schema import (
    ClassificationHint,
    Constraint,
    ConstraintType,
    State,
    Transition,
    WorkflowDefinition,
)

logger = logging.getLogger(__name__)


# Schema description for LLM prompt
FSM_SCHEMA_DESCRIPTION = """
Generate a JSON object with this structure:

{
  "name": "workflow-name",
  "description": "Brief description of the workflow",
  "states": [
    {
      "name": "state_name",  // lowercase with underscores
      "description": "What this state represents",
      "is_initial": true/false,  // First state in workflow
      "is_terminal": true/false,  // End state
      "classification": {
        "tool_calls": ["function_name"],  // Tool/function names that indicate this state
        "patterns": ["regex.*pattern"],   // Regex patterns in response text
        "exemplars": ["example phrase"]   // Example text for semantic matching
      }
    }
  ],
  "transitions": [
    {
      "from_state": "state_a",
      "to_state": "state_b",
      "description": "When/why this transition happens"
    }
  ],
  "constraints": [
    {
      "name": "constraint_name",
      "description": "What this constraint enforces",
      "type": "precedence|never|eventually|response",
      "trigger": "triggering_state",  // Required for precedence, response
      "target": "target_state",       // Required for all except 'always'
      "severity": "warning|error|critical",
      "intervention": "intervention_name"  // Must match key in interventions
    }
  ],
  "interventions": {
    "intervention_name": "Message to inject when constraint is violated. Guide the agent back on track."
  }
}

Constraint types:
- "precedence": target must occur BEFORE trigger (e.g., verify identity before refund)
- "never": target state must never occur (e.g., never share internal info)
- "eventually": target state must eventually be reached
- "response": if trigger occurs, target must eventually follow

Rules:
1. At least one state must have is_initial: true
2. Use snake_case for all names
3. Every constraint intervention must have a matching entry in interventions
4. For "never" constraints, the target can be a conceptual forbidden state not in states list
5. Classification hints help identify when the agent is in each state
"""


@register_compiler("fsm")
class FSMCompiler(LLMPolicyCompiler):
    """
    Compiler that converts natural language to FSM WorkflowDefinition.

    Uses an LLM to parse policy descriptions and generate workflow YAML
    with states, transitions, constraints, and interventions.
    """

    @property
    def engine_type(self) -> str:
        """Engine type this compiler produces config for."""
        return "fsm"

    def _build_compilation_prompt(
        self,
        natural_language: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build the prompt for FSM workflow compilation.

        Args:
            natural_language: User's policy description
            context: Optional hints (domain, tool_names, etc.)

        Returns:
            Complete prompt for LLM
        """
        prompt_parts = [
            "Convert the following natural language policy into an FSM workflow configuration.",
            "",
            FSM_SCHEMA_DESCRIPTION,
            "",
        ]

        # Add context if provided
        if context:
            if context.get("domain"):
                prompt_parts.append(f"Domain: {context['domain']}")

            if context.get("tool_names"):
                tools = ", ".join(context["tool_names"])
                prompt_parts.append(f"Available tools/functions: {tools}")

            if context.get("existing_states"):
                states = ", ".join(context["existing_states"])
                prompt_parts.append(f"Known states to include: {states}")

            prompt_parts.append("")

        prompt_parts.extend([
            "Natural language policy:",
            "---",
            natural_language,
            "---",
            "",
            "Generate the JSON workflow configuration:",
        ])

        return "\n".join(prompt_parts)

    def _parse_compilation_response(
        self,
        response: Dict[str, Any],
        natural_language: str,
    ) -> CompilationResult:
        """
        Parse LLM JSON response into WorkflowDefinition.

        Args:
            response: Parsed JSON from LLM
            natural_language: Original policy for metadata

        Returns:
            CompilationResult with WorkflowDefinition config
        """
        warnings: List[str] = []
        errors: List[str] = []

        try:
            # Parse states
            states = []
            for state_data in response.get("states", []):
                classification_data = state_data.get("classification", {})
                classification = ClassificationHint(
                    tool_calls=classification_data.get("tool_calls"),
                    patterns=classification_data.get("patterns"),
                    exemplars=classification_data.get("exemplars"),
                    min_similarity=classification_data.get("min_similarity", 0.7),
                )

                state = State(
                    name=state_data["name"],
                    description=state_data.get("description"),
                    classification=classification,
                    is_initial=state_data.get("is_initial", False),
                    is_terminal=state_data.get("is_terminal", False),
                    is_error=state_data.get("is_error", False),
                )
                states.append(state)

            if not states:
                errors.append("No states generated")
                return CompilationResult.failure(errors, warnings)

            # Ensure at least one initial state
            if not any(s.is_initial for s in states):
                states[0].is_initial = True
                warnings.append(f"No initial state specified, marked '{states[0].name}' as initial")

            # Parse transitions
            transitions = []
            for trans_data in response.get("transitions", []):
                trans = Transition(
                    from_state=trans_data["from_state"],
                    to_state=trans_data["to_state"],
                    description=trans_data.get("description"),
                )
                transitions.append(trans)

            # Parse interventions
            interventions = response.get("interventions", {})

            # Parse constraints
            constraints = []
            for const_data in response.get("constraints", []):
                try:
                    const_type = ConstraintType(const_data["type"])
                except ValueError:
                    warnings.append(
                        f"Unknown constraint type '{const_data['type']}', skipping"
                    )
                    continue

                # Validate intervention reference
                intervention_name = const_data.get("intervention")
                if intervention_name and intervention_name not in interventions:
                    # Auto-generate a basic intervention
                    interventions[intervention_name] = (
                        f"Policy reminder: {const_data.get('description', const_data['name'])}"
                    )
                    warnings.append(
                        f"Auto-generated missing intervention: {intervention_name}"
                    )

                constraint = Constraint(
                    name=const_data["name"],
                    description=const_data.get("description"),
                    type=const_type,
                    trigger=const_data.get("trigger"),
                    target=const_data.get("target"),
                    severity=const_data.get("severity", "error"),
                    intervention=intervention_name,
                )
                constraints.append(constraint)

            # Build WorkflowDefinition
            workflow = WorkflowDefinition(
                name=response.get("name", "compiled-policy"),
                version=response.get("version", "1.0"),
                description=response.get("description"),
                states=states,
                transitions=transitions,
                constraints=constraints,
                interventions=interventions,
            )

            return CompilationResult(
                success=True,
                config=workflow,
                warnings=warnings,
                metadata={
                    "source": natural_language[:200],  # Truncate for metadata
                    "state_count": len(states),
                    "constraint_count": len(constraints),
                },
            )

        except KeyError as e:
            errors.append(f"Missing required field: {e}")
            return CompilationResult.failure(errors, warnings)
        except Exception as e:
            logger.exception("Failed to parse compilation response")
            errors.append(f"Parse error: {type(e).__name__}: {e}")
            return CompilationResult.failure(errors, warnings)

    def validate_result(self, result: CompilationResult) -> List[str]:
        """
        Validate the compiled WorkflowDefinition.

        Args:
            result: Compilation result to validate

        Returns:
            List of validation errors
        """
        errors = super().validate_result(result)
        if errors:
            return errors

        workflow: WorkflowDefinition = result.config

        # Additional FSM-specific validation
        state_names = {s.name for s in workflow.states}

        # Check transition references
        for trans in workflow.transitions:
            if trans.from_state not in state_names:
                errors.append(f"Transition references unknown state: {trans.from_state}")
            if trans.to_state not in state_names:
                errors.append(f"Transition references unknown state: {trans.to_state}")

        # Check constraint references (except NEVER which can reference conceptual states)
        for const in workflow.constraints:
            if const.trigger and const.trigger not in state_names:
                errors.append(
                    f"Constraint '{const.name}' references unknown trigger: {const.trigger}"
                )
            if (
                const.target
                and const.target not in state_names
                and const.type != ConstraintType.NEVER
            ):
                errors.append(
                    f"Constraint '{const.name}' references unknown target: {const.target}"
                )

        return errors

    def export(self, result: CompilationResult, output_path: Path) -> None:
        """
        Export WorkflowDefinition to YAML file.

        Args:
            result: Successful compilation result
            output_path: Path to write YAML file

        Raises:
            ValueError: If result was not successful
        """
        if not result.success:
            raise ValueError("Cannot export failed compilation result")

        workflow: WorkflowDefinition = result.config

        # Convert to dict for YAML export
        workflow_dict = {
            "name": workflow.name,
            "version": workflow.version,
        }

        if workflow.description:
            workflow_dict["description"] = workflow.description

        # States
        workflow_dict["states"] = []
        for state in workflow.states:
            state_dict: Dict[str, Any] = {"name": state.name}

            if state.description:
                state_dict["description"] = state.description

            if state.is_initial:
                state_dict["is_initial"] = True
            if state.is_terminal:
                state_dict["is_terminal"] = True
            if state.is_error:
                state_dict["is_error"] = True

            # Classification hints
            classification: Dict[str, Any] = {}
            if state.classification.tool_calls:
                classification["tool_calls"] = state.classification.tool_calls
            if state.classification.patterns:
                classification["patterns"] = state.classification.patterns
            if state.classification.exemplars:
                classification["exemplars"] = state.classification.exemplars
            if state.classification.min_similarity != 0.7:
                classification["min_similarity"] = state.classification.min_similarity

            if classification:
                state_dict["classification"] = classification

            workflow_dict["states"].append(state_dict)

        # Transitions
        if workflow.transitions:
            workflow_dict["transitions"] = []
            for trans in workflow.transitions:
                trans_dict: Dict[str, Any] = {
                    "from_state": trans.from_state,
                    "to_state": trans.to_state,
                }
                if trans.description:
                    trans_dict["description"] = trans.description
                workflow_dict["transitions"].append(trans_dict)

        # Constraints
        if workflow.constraints:
            workflow_dict["constraints"] = []
            for const in workflow.constraints:
                const_dict: Dict[str, Any] = {
                    "name": const.name,
                    "type": const.type.value,
                }
                if const.description:
                    const_dict["description"] = const.description
                if const.trigger:
                    const_dict["trigger"] = const.trigger
                if const.target:
                    const_dict["target"] = const.target
                if const.severity != "error":
                    const_dict["severity"] = const.severity
                if const.intervention:
                    const_dict["intervention"] = const.intervention

                workflow_dict["constraints"].append(const_dict)

        # Interventions
        if workflow.interventions:
            workflow_dict["interventions"] = workflow.interventions

        # Write YAML
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(
                workflow_dict,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

        logger.info(f"Exported workflow to {output_path}")
