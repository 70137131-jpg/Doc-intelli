"""Tests for the workflow registry and workflow classes."""

import pytest

from app.agents.workflows import (
    WORKFLOWS,
    ContractComparisonAgent,
    InvoiceAnomalyAgent,
    SummarizationAgent,
    ResearchAssistantAgent,
)


class TestWorkflowRegistry:
    def test_all_workflows_registered(self):
        expected = ["contract_comparison", "invoice_anomaly", "summarization", "research_assistant"]
        for name in expected:
            assert name in WORKFLOWS, f"Missing workflow: {name}"

    def test_workflow_has_required_keys(self):
        for name, info in WORKFLOWS.items():
            assert "class" in info, f"{name} missing 'class'"
            assert "description" in info, f"{name} missing 'description'"
            assert "required_params" in info, f"{name} missing 'required_params'"
            assert "optional_params" in info, f"{name} missing 'optional_params'"

    def test_workflow_classes_are_instantiable(self):
        for name, info in WORKFLOWS.items():
            cls = info["class"]
            instance = cls()
            assert hasattr(instance, "run"), f"{name} class missing 'run' method"

    def test_contract_comparison_requires_two_docs(self):
        info = WORKFLOWS["contract_comparison"]
        assert "doc_id_1" in info["required_params"]
        assert "doc_id_2" in info["required_params"]

    def test_summarization_requires_document_id(self):
        info = WORKFLOWS["summarization"]
        assert "document_id" in info["required_params"]

    def test_research_requires_question(self):
        info = WORKFLOWS["research_assistant"]
        assert "research_question" in info["required_params"]

    def test_invoice_anomaly_has_optional_target(self):
        info = WORKFLOWS["invoice_anomaly"]
        assert info["required_params"] == []
        assert "target_doc_id" in info["optional_params"]
