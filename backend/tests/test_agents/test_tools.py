"""Tests for agent tool definitions and ToolResult."""

import pytest

from app.agents.tools import ToolResult, TOOL_REGISTRY


class TestToolResult:
    def test_success_result(self):
        result = ToolResult(success=True, data={"key": "value"})
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None

    def test_error_result(self):
        result = ToolResult(success=False, data="", error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"


class TestToolRegistry:
    def test_registry_has_expected_tools(self):
        expected = [
            "search_documents",
            "get_document_fields",
            "get_document_text",
            "compare_documents",
            "list_documents_by_type",
            "generate_with_llm",
        ]
        for name in expected:
            assert name in TOOL_REGISTRY, f"Missing tool: {name}"

    def test_each_tool_has_required_keys(self):
        for name, info in TOOL_REGISTRY.items():
            assert "fn" in info, f"{name} missing 'fn'"
            assert callable(info["fn"]), f"{name} 'fn' is not callable"
            assert "description" in info, f"{name} missing 'description'"
            assert "parameters" in info, f"{name} missing 'parameters'"
            assert isinstance(info["parameters"], dict), f"{name} parameters should be a dict"

    def test_tool_descriptions_not_empty(self):
        for name, info in TOOL_REGISTRY.items():
            assert len(info["description"]) > 10, f"{name} has too short a description"
