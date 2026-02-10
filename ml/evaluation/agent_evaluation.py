"""
Agent evaluation framework for measuring workflow quality.

Evaluates agents on:
1. Task completion rate
2. Tool usage efficiency (fewer steps = better)
3. Answer quality (faithfulness, relevance, completeness)
4. Error handling (graceful recovery)

Usage:
    python agent_evaluation.py --eval_file test_cases.json --output results.json
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalCase:
    """A single evaluation test case."""

    case_id: str
    workflow: str
    params: dict
    expected_keywords: list[str] = field(default_factory=list)
    expected_tools_used: list[str] = field(default_factory=list)
    max_steps: int = 15
    should_complete: bool = True


@dataclass
class EvalResult:
    case_id: str
    workflow: str
    status: str
    completed: bool
    num_steps: int
    tools_used: list[str]
    duration_ms: float
    keyword_hits: int
    keyword_total: int
    expected_tools_hit: int
    expected_tools_total: int
    error: str | None = None

    @property
    def keyword_score(self) -> float:
        return self.keyword_hits / max(self.keyword_total, 1)

    @property
    def tool_efficiency(self) -> float:
        """Lower is better: ratio of steps used to max steps."""
        return self.num_steps / max(self.num_steps, 1)

    @property
    def tool_coverage(self) -> float:
        return self.expected_tools_hit / max(self.expected_tools_total, 1)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "workflow": self.workflow,
            "status": self.status,
            "completed": self.completed,
            "num_steps": self.num_steps,
            "tools_used": self.tools_used,
            "duration_ms": round(self.duration_ms, 1),
            "keyword_score": round(self.keyword_score, 3),
            "tool_coverage": round(self.tool_coverage, 3),
            "error": self.error,
        }


def evaluate_agent_run(state_dict: dict, case: EvalCase) -> EvalResult:
    """Evaluate a completed agent run against expected outcomes."""
    steps = state_dict.get("steps", [])
    final_answer = (state_dict.get("final_answer") or "").lower()
    status = state_dict.get("status", "unknown")

    # Tools used
    tools_used = [
        s.get("tool_name")
        for s in steps
        if s.get("step_type") == "tool_call" and s.get("tool_name")
    ]

    # Keyword check
    keyword_hits = sum(1 for kw in case.expected_keywords if kw.lower() in final_answer)

    # Expected tools check
    tools_used_set = set(tools_used)
    expected_tools_hit = sum(1 for t in case.expected_tools_used if t in tools_used_set)

    return EvalResult(
        case_id=case.case_id,
        workflow=case.workflow,
        status=status,
        completed=(status == "completed"),
        num_steps=len(steps),
        tools_used=tools_used,
        duration_ms=0.0,
        keyword_hits=keyword_hits,
        keyword_total=len(case.expected_keywords),
        expected_tools_hit=expected_tools_hit,
        expected_tools_total=len(case.expected_tools_used),
    )


def run_evaluation(cases: list[EvalCase], run_fn) -> list[EvalResult]:
    """Run all evaluation cases and collect results.

    Args:
        cases: List of evaluation test cases.
        run_fn: Callable(workflow, params) -> state_dict
    """
    results = []
    for case in cases:
        print(f"Evaluating: {case.case_id} ({case.workflow})")
        start = time.time()
        try:
            state_dict = run_fn(case.workflow, case.params)
            duration = (time.time() - start) * 1000
            result = evaluate_agent_run(state_dict, case)
            result.duration_ms = duration
        except Exception as e:
            duration = (time.time() - start) * 1000
            result = EvalResult(
                case_id=case.case_id,
                workflow=case.workflow,
                status="error",
                completed=False,
                num_steps=0,
                tools_used=[],
                duration_ms=duration,
                keyword_hits=0,
                keyword_total=len(case.expected_keywords),
                expected_tools_hit=0,
                expected_tools_total=len(case.expected_tools_used),
                error=str(e),
            )
        results.append(result)
        print(f"  -> {result.status} | steps={result.num_steps} | keywords={result.keyword_score:.0%}")

    return results


def generate_report(results: list[EvalResult]) -> dict:
    """Generate aggregate evaluation report."""
    total = len(results)
    completed = sum(1 for r in results if r.completed)
    avg_steps = sum(r.num_steps for r in results) / max(total, 1)
    avg_keyword = sum(r.keyword_score for r in results) / max(total, 1)
    avg_tool_cov = sum(r.tool_coverage for r in results) / max(total, 1)
    avg_duration = sum(r.duration_ms for r in results) / max(total, 1)

    return {
        "total_cases": total,
        "completion_rate": round(completed / max(total, 1), 3),
        "avg_steps": round(avg_steps, 1),
        "avg_keyword_score": round(avg_keyword, 3),
        "avg_tool_coverage": round(avg_tool_cov, 3),
        "avg_duration_ms": round(avg_duration, 1),
        "per_workflow": _per_workflow_stats(results),
        "results": [r.to_dict() for r in results],
    }


def _per_workflow_stats(results: list[EvalResult]) -> dict:
    """Break down stats by workflow type."""
    by_wf: dict[str, list[EvalResult]] = {}
    for r in results:
        by_wf.setdefault(r.workflow, []).append(r)

    stats = {}
    for wf, wf_results in by_wf.items():
        n = len(wf_results)
        stats[wf] = {
            "count": n,
            "completion_rate": round(sum(1 for r in wf_results if r.completed) / n, 3),
            "avg_steps": round(sum(r.num_steps for r in wf_results) / n, 1),
            "avg_keyword_score": round(sum(r.keyword_score for r in wf_results) / n, 3),
        }
    return stats


# --- Sample test cases ---
SAMPLE_CASES = [
    EvalCase(
        case_id="summarize_basic",
        workflow="summarization",
        params={"document_id": "test-doc-1"},
        expected_keywords=["summary", "key"],
        expected_tools_used=["get_document_text"],
    ),
    EvalCase(
        case_id="research_basic",
        workflow="research_assistant",
        params={"research_question": "What are the key contract terms?"},
        expected_keywords=["contract", "terms"],
        expected_tools_used=["search_documents"],
    ),
    EvalCase(
        case_id="invoice_anomaly",
        workflow="invoice_anomaly",
        params={},
        expected_keywords=["anomal", "invoice"],
        expected_tools_used=["list_documents_by_type", "get_document_fields"],
    ),
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent Evaluation")
    parser.add_argument("--eval_file", type=str, default=None, help="JSON file with eval cases")
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()

    if args.eval_file:
        with open(args.eval_file) as f:
            raw = json.load(f)
        cases = [EvalCase(**c) for c in raw]
    else:
        cases = SAMPLE_CASES
        print("Using sample evaluation cases (no --eval_file provided)")

    # Import agent runner
    try:
        from app.agents.workflows import WORKFLOWS

        def run_fn(workflow: str, params: dict) -> dict:
            agent_class = WORKFLOWS[workflow]["class"]
            agent = agent_class()
            all_p = WORKFLOWS[workflow]["required_params"] + WORKFLOWS[workflow]["optional_params"]
            filtered = {k: v for k, v in params.items() if k in all_p}
            state = agent.run(**filtered)
            return state.to_dict()

        results = run_evaluation(cases, run_fn)
        report = generate_report(results)

        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nEvaluation Report:")
        print(f"  Completion rate: {report['completion_rate']:.0%}")
        print(f"  Avg steps:       {report['avg_steps']}")
        print(f"  Avg keyword:     {report['avg_keyword_score']:.0%}")
        print(f"  Avg tool cov:    {report['avg_tool_coverage']:.0%}")
        print(f"\nResults saved to {args.output}")

    except ImportError as e:
        print(f"Cannot import agent modules: {e}")
        print("Run from the backend directory with: python -m ml.evaluation.agent_evaluation")
