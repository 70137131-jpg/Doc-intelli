"""
Specialized agent workflows for document intelligence tasks.

1. Contract Comparison - Compare two contracts, highlight differences
2. Invoice Anomaly Detection - Detect anomalies across invoices
3. Document Summarization - Multi-level summaries
4. Research Assistant - Cross-document research with citations
"""

from app.agents.base import ReActAgent, AgentState
from app.agents.tools import (
    search_documents,
    get_document_text,
    get_document_fields,
    compare_documents,
    list_documents_by_type,
    generate_with_llm,
    ToolResult,
)
from app.core.logging import get_logger
from app.services.llm_service import LLMService

logger = get_logger(__name__)


class ContractComparisonAgent:
    """Compares two contract documents and generates a structured comparison report."""

    def run(self, doc_id_1: str, doc_id_2: str) -> AgentState:
        state = AgentState(task=f"Compare contracts {doc_id_1} and {doc_id_2}", workflow="contract_comparison")

        try:
            # Step 1: Load both documents
            from app.agents.base import AgentStep, StepType
            state.add_step(AgentStep(step_type=StepType.REASONING, content="Loading both contracts for comparison"))

            comparison = compare_documents(doc_id_1, doc_id_2)
            if not comparison.success:
                state.status = "failed"
                state.final_answer = f"Failed to load documents: {comparison.error}"
                return state

            doc1 = comparison.data["document_1"]
            doc2 = comparison.data["document_2"]

            state.add_step(AgentStep(
                step_type=StepType.TOOL_RESULT,
                content=f"Loaded: {doc1['name']} and {doc2['name']}",
                tool_name="compare_documents",
            ))

            # Step 2: Use LLM to generate comparison report
            state.add_step(AgentStep(step_type=StepType.REASONING, content="Analyzing contract differences"))

            prompt = f"""Compare these two contracts and generate a detailed comparison report.

CONTRACT 1: {doc1['name']}
Type: {doc1['type']}
Fields: {doc1['fields']}
Text (excerpt):
{doc1['text'][:3000]}

---

CONTRACT 2: {doc2['name']}
Type: {doc2['type']}
Fields: {doc2['fields']}
Text (excerpt):
{doc2['text'][:3000]}

---

Generate a structured comparison report with these sections:
1. **Overview**: Brief description of both contracts
2. **Key Differences**: Side-by-side comparison of major terms (parties, dates, amounts, obligations)
3. **Added Clauses**: Clauses present in Contract 2 but not in Contract 1
4. **Removed Clauses**: Clauses present in Contract 1 but not in Contract 2
5. **Modified Terms**: Terms that exist in both but differ
6. **Risk Assessment**: Flag any potentially risky differences or unusual clauses
7. **Recommendation**: Summary of which contract is more favorable and why

Be specific and cite exact language from both documents."""

            llm = LLMService()
            report = llm.generate(prompt)

            state.add_step(AgentStep(
                step_type=StepType.FINAL_ANSWER,
                content=report,
            ))
            state.final_answer = report
            state.status = "completed"

        except Exception as e:
            logger.error(f"Contract comparison failed: {e}")
            state.status = "failed"
            state.final_answer = f"Comparison failed: {e}"

        return state


class InvoiceAnomalyAgent:
    """Analyzes invoices for anomalies by comparing against historical patterns."""

    def run(self, target_doc_id: str | None = None) -> AgentState:
        task = f"Detect invoice anomalies{f' for {target_doc_id}' if target_doc_id else ''}"
        state = AgentState(task=task, workflow="invoice_anomaly")

        try:
            from app.agents.base import AgentStep, StepType

            # Step 1: Get all invoices
            state.add_step(AgentStep(step_type=StepType.REASONING, content="Loading all invoice documents"))

            invoices = list_documents_by_type("Invoice", limit=50)
            if not invoices.success or not invoices.data:
                state.final_answer = "No invoices found in the system."
                state.status = "completed"
                return state

            state.add_step(AgentStep(
                step_type=StepType.TOOL_RESULT,
                content=f"Found {len(invoices.data)} invoices",
                tool_name="list_documents_by_type",
            ))

            # Step 2: Extract fields from all invoices
            state.add_step(AgentStep(step_type=StepType.REASONING, content="Extracting fields from invoices for pattern analysis"))

            invoice_data = []
            for inv in invoices.data[:20]:  # Limit to 20 for performance
                fields = get_document_fields(inv["document_id"])
                if fields.success:
                    invoice_data.append({
                        "document_id": inv["document_id"],
                        "filename": inv["filename"],
                        "fields": fields.data.get("fields", {}),
                    })

            state.add_step(AgentStep(
                step_type=StepType.TOOL_RESULT,
                content=f"Extracted fields from {len(invoice_data)} invoices",
                tool_name="get_document_fields",
            ))

            # Step 3: Analyze with LLM
            state.add_step(AgentStep(step_type=StepType.REASONING, content="Analyzing invoices for anomalies"))

            import json
            invoices_json = json.dumps(invoice_data[:15], indent=2, default=str)

            target_section = ""
            if target_doc_id:
                target_fields = get_document_fields(target_doc_id)
                if target_fields.success:
                    target_section = f"\n\nFocus analysis on this specific invoice:\n{json.dumps(target_fields.data, indent=2, default=str)}"

            prompt = f"""Analyze these invoices for anomalies and suspicious patterns.

Invoice Data:
{invoices_json}
{target_section}

Generate an anomaly detection report with:
1. **Summary Statistics**: Total invoices analyzed, amount range, common vendors
2. **Detected Anomalies**: List each anomaly with:
   - Severity: HIGH / MEDIUM / LOW
   - Type: unusual_amount, duplicate, missing_fields, new_vendor, date_anomaly
   - Description: What was found
   - Affected Invoice: Document ID and filename
3. **Pattern Analysis**: Common patterns in the invoice data
4. **Recommendations**: Actions to take for flagged invoices

Be specific with amounts, dates, and vendor names."""

            llm = LLMService()
            report = llm.generate(prompt)

            state.add_step(AgentStep(step_type=StepType.FINAL_ANSWER, content=report))
            state.final_answer = report
            state.status = "completed"

        except Exception as e:
            logger.error(f"Invoice anomaly detection failed: {e}")
            state.status = "failed"
            state.final_answer = f"Analysis failed: {e}"

        return state


class SummarizationAgent:
    """Generates multi-level summaries of documents."""

    def run(self, document_id: str) -> AgentState:
        state = AgentState(task=f"Summarize document {document_id}", workflow="summarization")

        try:
            from app.agents.base import AgentStep, StepType

            # Step 1: Load document
            state.add_step(AgentStep(step_type=StepType.REASONING, content="Loading document for summarization"))

            doc_result = get_document_text(document_id, max_chunks=20)
            if not doc_result.success:
                state.final_answer = f"Failed to load document: {doc_result.error}"
                state.status = "failed"
                return state

            doc_name = doc_result.data["document_name"]
            doc_text = doc_result.data["text"]

            state.add_step(AgentStep(
                step_type=StepType.TOOL_RESULT,
                content=f"Loaded {doc_name} ({doc_result.data['chunk_count']} chunks)",
                tool_name="get_document_text",
            ))

            # Step 2: Get classification
            fields_result = get_document_fields(document_id)
            doc_type = "Document"
            if fields_result.success:
                doc_type = fields_result.data.get("document_type", "Document")

            # Step 3: Generate hierarchical summary
            state.add_step(AgentStep(step_type=StepType.REASONING, content="Generating multi-level summary"))

            prompt = f"""Create a comprehensive, hierarchical summary of this {doc_type}.

Document: {doc_name}
Content:
{doc_text[:6000]}

Generate THREE levels of summary:

## One-Line Summary
A single sentence capturing the document's essence.

## Executive Summary
A 3-5 sentence paragraph covering the most important points. Written for someone who needs to understand the document in 30 seconds.

## Detailed Summary
A thorough summary organized by sections/topics with:
- Key information and data points
- Important decisions or action items
- Notable dates, amounts, or deadlines
- Any risks or concerns mentioned

## Key Takeaways
- Bullet points of the most actionable items

If this is a meeting document, also extract:
- Attendees
- Action Items (who, what, deadline)
- Decisions Made"""

            llm = LLMService()
            summary = llm.generate(prompt)

            state.add_step(AgentStep(step_type=StepType.FINAL_ANSWER, content=summary))
            state.final_answer = summary
            state.status = "completed"

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            state.status = "failed"
            state.final_answer = f"Summarization failed: {e}"

        return state


class ResearchAssistantAgent:
    """Researches a question across all documents and produces a cited report."""

    def run(self, research_question: str) -> AgentState:
        # Use the general ReAct agent for research since it requires multi-step reasoning
        agent = ReActAgent(max_steps=8)

        task = f"""Research the following question across all available documents and produce a comprehensive answer with citations.

Research Question: {research_question}

Steps to follow:
1. Search for documents relevant to the question
2. Read the most relevant documents in detail
3. Synthesize findings from multiple sources
4. Produce a mini-report with:
   - Answer to the research question
   - Supporting evidence from each document (with citations)
   - Any conflicting information found
   - Areas where more information might be needed

Always cite sources as [Source: document_name, page X]."""

        return agent.run(task=task, workflow="research_assistant")


# Workflow registry
WORKFLOWS = {
    "contract_comparison": {
        "class": ContractComparisonAgent,
        "description": "Compare two contracts and generate a structured difference report with risk assessment.",
        "required_params": ["doc_id_1", "doc_id_2"],
        "optional_params": [],
    },
    "invoice_anomaly": {
        "class": InvoiceAnomalyAgent,
        "description": "Analyze invoices for anomalies: unusual amounts, duplicates, missing fields, new vendors.",
        "required_params": [],
        "optional_params": ["target_doc_id"],
    },
    "summarization": {
        "class": SummarizationAgent,
        "description": "Generate a hierarchical summary of a document: one-liner, executive summary, detailed summary.",
        "required_params": ["document_id"],
        "optional_params": [],
    },
    "research_assistant": {
        "class": ResearchAssistantAgent,
        "description": "Research a question across all documents, synthesize findings, produce a cited report.",
        "required_params": ["research_question"],
        "optional_params": [],
    },
}
