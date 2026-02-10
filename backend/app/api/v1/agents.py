import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.agents.workflows import WORKFLOWS
from app.core.logging import get_logger
from app.schemas.agent import (
    AgentRunRequest,
    AgentRunResponse,
    AgentRunSummary,
    AgentStepResponse,
    WorkflowInfo,
    WorkflowListResponse,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/agents", tags=["Agents"])

# In-memory store for completed runs (production would use Redis/DB)
_run_store: dict[str, dict] = {}
# Store agent states for checkpoint resumption
_agent_states = {}

# Thread pool for running sync agent workflows
_executor = ThreadPoolExecutor(max_workers=4)

# --- Rate limiting ---
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 10  # max requests per window per IP
_rate_limit_buckets: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(client_ip: str) -> bool:
    """Simple sliding-window rate limiter. Returns True if allowed."""
    now = time.time()
    bucket = _rate_limit_buckets[client_ip]
    # Prune old entries
    _rate_limit_buckets[client_ip] = [t for t in bucket if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_limit_buckets[client_ip]) >= RATE_LIMIT_MAX:
        return False
    _rate_limit_buckets[client_ip].append(now)
    return True


@router.get("/workflows", response_model=WorkflowListResponse)
async def list_workflows():
    """List all available agent workflows."""
    workflows = [
        WorkflowInfo(
            name=name,
            description=info["description"],
            required_params=info["required_params"],
            optional_params=info["optional_params"],
        )
        for name, info in WORKFLOWS.items()
    ]
    return WorkflowListResponse(workflows=workflows)


@router.post("/run", response_model=AgentRunResponse)
async def run_agent(request: AgentRunRequest, req: Request):
    """Run an agent workflow synchronously and return the full result."""
    client_ip = req.client.host if req.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_MAX} agent runs per {RATE_LIMIT_WINDOW}s.",
        )

    if request.workflow not in WORKFLOWS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown workflow: {request.workflow}. Available: {list(WORKFLOWS.keys())}",
        )

    workflow_info = WORKFLOWS[request.workflow]
    agent_class = workflow_info["class"]

    # Validate required params
    for param in workflow_info["required_params"]:
        if param not in request.params:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required parameter: {param}",
            )

    try:
        agent = agent_class()
        # Filter params to only those accepted by the workflow
        all_params = workflow_info["required_params"] + workflow_info["optional_params"]
        filtered_params = {k: v for k, v in request.params.items() if k in all_params}

        state = agent.run(**filtered_params)

        # Store the result + live state for checkpoint resumption
        state_dict = state.to_dict()
        _run_store[state.run_id] = state_dict
        if state.status == "awaiting_approval":
            _agent_states[state.run_id] = {"state": state, "agent": agent}

        return AgentRunResponse(
            run_id=state.run_id,
            task=state.task,
            workflow=state.workflow,
            status=state.status,
            steps=[
                AgentStepResponse(
                    step_type=s.step_type.value if hasattr(s.step_type, "value") else s.step_type,
                    content=s.content,
                    tool_name=s.tool_name,
                    tool_args=s.tool_args,
                    timestamp=s.timestamp,
                    duration_ms=s.duration_ms,
                )
                for s in state.steps
            ],
            final_answer=state.final_answer,
            started_at=state.started_at,
            completed_at=state.completed_at,
            total_steps=len(state.steps),
        )

    except Exception as e:
        logger.error(f"Agent run failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {e}")


@router.get("/runs/{run_id}", response_model=AgentRunResponse)
async def get_run(run_id: str):
    """Get the result of a previous agent run."""
    if run_id not in _run_store:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    data = _run_store[run_id]
    return AgentRunResponse(**data)


@router.post("/run/stream")
async def run_agent_stream(request: AgentRunRequest):
    """Run an agent workflow with streaming step-by-step output via SSE."""
    if request.workflow not in WORKFLOWS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown workflow: {request.workflow}. Available: {list(WORKFLOWS.keys())}",
        )

    workflow_info = WORKFLOWS[request.workflow]

    for param in workflow_info["required_params"]:
        if param not in request.params:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required parameter: {param}",
            )

    async def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'status', 'content': f'Starting workflow: {request.workflow}'})}\n\n"

            agent_class = workflow_info["class"]
            agent = agent_class()

            all_params = workflow_info["required_params"] + workflow_info["optional_params"]
            filtered_params = {k: v for k, v in request.params.items() if k in all_params}

            # Run the agent (sync) — in production this would use async/background
            import asyncio
            loop = asyncio.get_event_loop()
            state = await loop.run_in_executor(_executor, lambda: agent.run(**filtered_params))

            # Stream each step
            for step in state.steps:
                step_data = step.to_dict()
                yield f"data: {json.dumps({'type': 'step', 'content': step_data})}\n\n"

            # Final result
            yield f"data: {json.dumps({'type': 'done', 'content': {'run_id': state.run_id, 'status': state.status, 'final_answer': state.final_answer}})}\n\n"

            _run_store[state.run_id] = state.to_dict()

        except Exception as e:
            logger.error(f"Streaming agent run failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/runs/{run_id}/approve", response_model=AgentRunResponse)
async def approve_checkpoint(run_id: str, approved: bool = True):
    """Approve or reject a pending human-in-the-loop checkpoint and resume the agent."""
    if run_id not in _agent_states:
        raise HTTPException(
            status_code=404,
            detail=f"No pending checkpoint for run {run_id}. It may have already completed.",
        )

    entry = _agent_states[run_id]
    agent = entry["agent"]
    state = entry["state"]

    if state.status != "awaiting_approval":
        raise HTTPException(status_code=400, detail="Run is not awaiting approval")

    try:
        from app.agents.base import ReActAgent

        if isinstance(agent, ReActAgent):
            state = agent.resume(state, approved=approved)
        else:
            # For non-ReAct workflow agents, just mark as completed
            state.status = "completed"

        state_dict = state.to_dict()
        _run_store[run_id] = state_dict

        if state.status == "awaiting_approval":
            _agent_states[run_id] = {"state": state, "agent": agent}
        else:
            _agent_states.pop(run_id, None)

        return AgentRunResponse(
            run_id=state.run_id,
            task=state.task,
            workflow=state.workflow,
            status=state.status,
            steps=[
                AgentStepResponse(
                    step_type=s.step_type.value if hasattr(s.step_type, "value") else s.step_type,
                    content=s.content,
                    tool_name=s.tool_name,
                    tool_args=s.tool_args,
                    timestamp=s.timestamp,
                    duration_ms=s.duration_ms,
                )
                for s in state.steps
            ],
            final_answer=state.final_answer,
            started_at=state.started_at,
            completed_at=state.completed_at,
            total_steps=len(state.steps),
        )

    except Exception as e:
        logger.error(f"Checkpoint approval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resume agent: {e}")
