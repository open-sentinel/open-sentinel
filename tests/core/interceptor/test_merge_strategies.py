"""
Tests for _merge_modifications() handling of all intervention strategy keys.

Covers: user_message_inject, context_reminder, system_prompt_append,
hard_block, combinations of multiple keys, and edge cases.
"""

import pytest

from opensentinel.core.interceptor import Interceptor
from opensentinel.core.intervention.strategies import WorkflowViolationError


class TestUserMessageInject:

    async def test_injects_before_last_user_message(self):
        """user_message_inject inserts guidance before the last user message."""
        interceptor = Interceptor([])
        base = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second question"},
            ]
        }
        mods = {"user_message_inject": "Please verify identity first."}

        result = interceptor._merge_modifications(base, mods)

        # Guidance inserted before the last user message (index 3 → now at 3, user at 4)
        assert len(result["messages"]) == 5
        assert result["messages"][3]["role"] == "user"
        assert "[System Note]:" in result["messages"][3]["content"]
        assert "Please verify identity first." in result["messages"][3]["content"]
        # Original last user message pushed to index 4
        assert result["messages"][4]["content"] == "second question"

    async def test_appends_when_no_user_messages(self):
        """user_message_inject appends when no user messages exist."""
        interceptor = Interceptor([])
        base = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
            ]
        }
        mods = {"user_message_inject": "Verify identity."}

        result = interceptor._merge_modifications(base, mods)

        assert len(result["messages"]) == 2
        assert result["messages"][1]["role"] == "user"
        assert "Verify identity." in result["messages"][1]["content"]


class TestContextReminder:

    async def test_inserts_before_last_message(self):
        """context_reminder inserts assistant reminder before the last message."""
        interceptor = Interceptor([])
        base = {
            "messages": [
                {"role": "user", "content": "do something"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "now do more"},
            ]
        }
        mods = {"context_reminder": "I need to verify identity first."}

        result = interceptor._merge_modifications(base, mods)

        assert len(result["messages"]) == 4
        # Reminder inserted before last message (index 2)
        assert result["messages"][2]["role"] == "assistant"
        assert "[Context reminder:" in result["messages"][2]["content"]
        assert "I need to verify identity first." in result["messages"][2]["content"]
        # Last message still at the end
        assert result["messages"][3]["content"] == "now do more"

    async def test_appends_when_empty(self):
        """context_reminder appends when messages list is empty."""
        interceptor = Interceptor([])
        base = {"messages": []}
        mods = {"context_reminder": "Remember the rules."}

        result = interceptor._merge_modifications(base, mods)

        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "assistant"
        assert "Remember the rules." in result["messages"][0]["content"]


class TestSystemPromptAppend:

    async def test_appends_to_existing_system_msg(self):
        """system_prompt_append appends to existing system message with WORKFLOW GUIDANCE."""
        interceptor = Interceptor([])
        base = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ]
        }
        mods = {"system_prompt_append": "Always be safe."}

        result = interceptor._merge_modifications(base, mods)

        assert "You are helpful." in result["messages"][0]["content"]
        assert "[WORKFLOW GUIDANCE]: Always be safe." in result["messages"][0]["content"]

    async def test_creates_system_msg(self):
        """system_prompt_append creates system message if none exists."""
        interceptor = Interceptor([])
        base = {"messages": [{"role": "user", "content": "hi"}]}
        mods = {"system_prompt_append": "Be safe."}

        result = interceptor._merge_modifications(base, mods)

        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "[WORKFLOW GUIDANCE]: Be safe."
        assert result["messages"][1]["role"] == "user"


class TestHardBlock:

    async def test_hard_block_raises(self):
        """hard_block key raises WorkflowViolationError."""
        interceptor = Interceptor([])
        base = {"messages": [{"role": "user", "content": "hi"}]}
        mods = {"hard_block": "Critical violation detected."}

        with pytest.raises(WorkflowViolationError, match="Critical violation detected"):
            interceptor._merge_modifications(base, mods)


class TestMultipleStrategyKeys:

    async def test_two_keys_both_applied(self):
        """Multiple strategy keys in the same dict are all applied."""
        interceptor = Interceptor([])
        base = {
            "messages": [
                {"role": "system", "content": "Base system."},
                {"role": "user", "content": "hello"},
            ]
        }
        mods = {
            "system_prompt_append": "Extra guidance.",
            "context_reminder": "Remember the workflow.",
        }

        result = interceptor._merge_modifications(base, mods)

        # system_prompt_append applied
        assert "[WORKFLOW GUIDANCE]: Extra guidance." in result["messages"][0]["content"]
        # context_reminder applied (inserted before last message)
        reminder_msgs = [
            m for m in result["messages"]
            if m.get("role") == "assistant" and "Context reminder" in m.get("content", "")
        ]
        assert len(reminder_msgs) == 1
        assert "Remember the workflow." in reminder_msgs[0]["content"]


class TestEdgeCases:

    async def test_no_messages_key_in_base(self):
        """Strategy creates messages key when base has none."""
        interceptor = Interceptor([])
        base = {"model": "gpt-4"}
        mods = {"system_prompt_append": "Be safe."}

        result = interceptor._merge_modifications(base, mods)

        assert "messages" in result
        assert result["messages"][0]["role"] == "system"
        assert "[WORKFLOW GUIDANCE]: Be safe." in result["messages"][0]["content"]

    async def test_multiple_same_strategy_merges(self):
        """Multiple sequential merges with same strategy both apply."""
        interceptor = Interceptor([])
        base = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ]
        }

        result = interceptor._merge_modifications(base, {"system_prompt_append": "Rule 1."})
        result = interceptor._merge_modifications(result, {"system_prompt_append": "Rule 2."})

        content = result["messages"][0]["content"]
        assert "Rule 1." in content
        assert "Rule 2." in content

    async def test_base_not_mutated(self):
        """Original base dict messages are not mutated."""
        interceptor = Interceptor([])
        original_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        base = {"messages": original_messages}
        mods = {"system_prompt_append": "Be safe."}

        interceptor._merge_modifications(base, mods)

        # Original should be unchanged
        assert original_messages[0]["content"] == "You are helpful."
        assert len(original_messages) == 2
