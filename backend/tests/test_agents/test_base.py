"""Tests for the ReAct agent base architecture."""

import pytest
from unittest.mock import patch, MagicMock

from app.agents.base import (
    ReActAgent,
    AgentState,
    AgentStep,
    StepType,
    REACT_SYSTEM_PROMPT,
)
from app.agents.tools import ToolResult


class TestAgentState:
    def test_initial_state(self):
        state = AgentState(task="test task", workflow="test")
        assert state.task == "test task"
        assert state.workflow == "test"
        assert state.status == "running"
        assert state.steps == []
        assert state.final_answer is None
        assert state.run_id  # Should have a UUID

    def test_add_step(self):
        state = AgentState(task="test", workflow="test")
        step = AgentStep(step_type=StepType.REASONING, content="Thinking...")
        state.add_step(step)
        assert len(state.steps) == 1
        assert state.steps[0].content == "Thinking..."

    def test_to_dict(self):
        state = AgentState(task="test task", workflow="test_workflow")
        state.add_step(AgentStep(step_type=StepType.REASONING, content="thinking"))
        d = state.to_dict()

        assert d["task"] == "test task"
        assert d["workflow"] == "test_workflow"
        assert d["status"] == "running"
        assert d["total_steps"] == 1
        assert len(d["steps"]) == 1
        assert d["steps"][0]["step_type"] == "reasoning"


class TestAgentStep:
    def test_step_to_dict(self):
        step = AgentStep(
            step_type=StepType.TOOL_CALL,
            content="Calling search",
            tool_name="search_documents",
            tool_args={"query": "test"},
        )
        d = step.to_dict()
        assert d["step_type"] == "tool_call"
        assert d["tool_name"] == "search_documents"
        assert d["tool_args"] == {"query": "test"}

    def test_step_types(self):
        assert StepType.REASONING.value == "reasoning"
        assert StepType.TOOL_CALL.value == "tool_call"
        assert StepType.TOOL_RESULT.value == "tool_result"
        assert StepType.FINAL_ANSWER.value == "final_answer"
        assert StepType.ERROR.value == "error"


class TestReActAgent:
    @patch("app.agents.base.LLMService")
    def test_agent_final_answer_direct(self, mock_llm_cls):
        """Agent that immediately returns a final answer."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = '{"action": "final_answer", "answer": "The answer is 42."}'
        mock_llm_cls.return_value = mock_llm

        agent = ReActAgent(max_steps=5)
        state = agent.run(task="What is the answer?", workflow="test")

        assert state.status == "completed"
        assert state.final_answer == "The answer is 42."
        assert any(s.step_type == StepType.FINAL_ANSWER for s in state.steps)

    @patch("app.agents.base.LLMService")
    def test_agent_tool_then_answer(self, mock_llm_cls):
        """Agent that uses a tool then gives final answer."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            '{"action": "tool", "tool_name": "search_documents", "tool_args": {"query": "test"}, "reasoning": "I need to search first"}',
            '{"action": "final_answer", "answer": "Found relevant documents."}',
        ]
        mock_llm_cls.return_value = mock_llm

        # Mock the tool
        with patch.dict(
            "app.agents.base.TOOL_REGISTRY",
            {
                "search_documents": {
                    "fn": lambda **kwargs: ToolResult(success=True, data=[{"doc": "test"}]),
                    "description": "Search documents",
                    "parameters": {"query": "str"},
                },
            },
        ):
            agent = ReActAgent(max_steps=5)
            state = agent.run(task="Search for test docs", workflow="test")

        assert state.status == "completed"
        assert state.final_answer == "Found relevant documents."
        step_types = [s.step_type for s in state.steps]
        assert StepType.REASONING in step_types
        assert StepType.TOOL_CALL in step_types
        assert StepType.TOOL_RESULT in step_types
        assert StepType.FINAL_ANSWER in step_types

    @patch("app.agents.base.LLMService")
    def test_agent_max_steps_reached(self, mock_llm_cls):
        """Agent that hits max steps without final answer."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = '{"action": "tool", "tool_name": "search_documents", "tool_args": {"query": "loop"}}'
        mock_llm_cls.return_value = mock_llm

        with patch.dict(
            "app.agents.base.TOOL_REGISTRY",
            {
                "search_documents": {
                    "fn": lambda **kwargs: ToolResult(success=True, data=[]),
                    "description": "Search",
                    "parameters": {"query": "str"},
                },
            },
        ):
            agent = ReActAgent(max_steps=2)
            state = agent.run(task="Keep looping", workflow="test")

        assert state.status == "completed"
        assert "maximum steps" in state.final_answer.lower()

    @patch("app.agents.base.LLMService")
    def test_agent_unknown_tool(self, mock_llm_cls):
        """Agent calls a tool that doesn't exist."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            '{"action": "tool", "tool_name": "nonexistent_tool", "tool_args": {}}',
            '{"action": "final_answer", "answer": "Done after error."}',
        ]
        mock_llm_cls.return_value = mock_llm

        agent = ReActAgent(max_steps=5)
        state = agent.run(task="Test unknown tool", workflow="test")

        assert state.status == "completed"
        # Should have a tool_result step with error
        error_steps = [s for s in state.steps if s.step_type == StepType.TOOL_RESULT and "Unknown tool" in s.content]
        assert len(error_steps) >= 1

    def test_parse_action_json(self):
        agent = ReActAgent()
        result = agent._parse_action('{"action": "final_answer", "answer": "test"}')
        assert result["action"] == "final_answer"
        assert result["answer"] == "test"

    def test_parse_action_json_in_text(self):
        agent = ReActAgent()
        result = agent._parse_action('Some preamble text {"action": "final_answer", "answer": "test"} trailing')
        assert result["action"] == "final_answer"

    def test_parse_action_fallback(self):
        agent = ReActAgent()
        result = agent._parse_action("This is not JSON at all")
        assert result["action"] == "final_answer"
        assert "This is not JSON at all" in result["answer"]

    def test_build_tool_descriptions(self):
        agent = ReActAgent()
        descriptions = agent._build_tool_descriptions()
        assert "search_documents" in descriptions
        assert "get_document_text" in descriptions
