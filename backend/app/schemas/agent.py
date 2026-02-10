import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    workflow: str = Field(..., description="Workflow name: contract_comparison, invoice_anomaly, summarization, research_assistant")
    params: dict = Field(default_factory=dict, description="Workflow parameters (e.g. doc_id_1, doc_id_2)")


class AgentStepResponse(BaseModel):
    step_type: str
    content: str
    tool_name: str | None = None
    tool_args: dict | None = None
    timestamp: datetime
    duration_ms: float = 0.0


class AgentRunResponse(BaseModel):
    run_id: str
    task: str
    workflow: str
    status: str  # running, completed, failed
    steps: list[AgentStepResponse]
    final_answer: str | None = None
    started_at: datetime
    completed_at: datetime | None = None
    total_steps: int


class AgentRunSummary(BaseModel):
    run_id: str
    workflow: str
    status: str
    final_answer: str | None = None
    total_steps: int
    started_at: datetime
    completed_at: datetime | None = None


class WorkflowInfo(BaseModel):
    name: str
    description: str
    required_params: list[str]
    optional_params: list[str]


class WorkflowListResponse(BaseModel):
    workflows: list[WorkflowInfo]
