"""LangFuse observability for LLM/RAG/Agent tracing."""

import logging
from functools import wraps
from typing import Any, Callable

from app.config import settings

logger = logging.getLogger(__name__)

_langfuse = None


def get_langfuse():
    """Lazy-initialize LangFuse client."""
    global _langfuse
    if _langfuse is not None:
        return _langfuse

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.info("LangFuse not configured — tracing disabled")
        return None

    try:
        from langfuse import Langfuse

        _langfuse = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info("LangFuse initialized successfully")
        return _langfuse
    except Exception as e:
        logger.warning(f"Failed to initialize LangFuse: {e}")
        return None


def trace_llm_call(name: str = "llm_call"):
    """Decorator to trace LLM API calls with LangFuse."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            lf = get_langfuse()
            if lf is None:
                return await func(*args, **kwargs)

            trace = lf.trace(name=name)
            generation = trace.generation(
                name=name,
                model=kwargs.get("model", "gemini-2.5-flash"),
                input=kwargs.get("prompt") or kwargs.get("query") or str(args[:1]),
            )

            try:
                result = await func(*args, **kwargs)
                generation.end(output=str(result)[:2000])
                return result
            except Exception as e:
                generation.end(output=f"ERROR: {e}", level="ERROR")
                raise

        return wrapper

    return decorator


def trace_rag_pipeline(query: str, chunks: list, answer: str, sources: list | None = None):
    """Trace a complete RAG pipeline execution."""
    lf = get_langfuse()
    if lf is None:
        return

    try:
        trace = lf.trace(name="rag_pipeline")

        # Log retrieval step
        trace.span(
            name="retrieval",
            input=query,
            output=f"{len(chunks)} chunks retrieved",
            metadata={"chunk_count": len(chunks)},
        )

        # Log generation step
        trace.generation(
            name="generation",
            model="gemini-2.5-flash",
            input=query,
            output=answer[:2000],
            metadata={
                "sources": len(sources) if sources else 0,
                "chunks_used": len(chunks),
            },
        )
    except Exception as e:
        logger.debug(f"LangFuse tracing error: {e}")


def trace_agent_execution(workflow_id: str, params: dict, steps: list, result: str):
    """Trace an agent workflow execution."""
    lf = get_langfuse()
    if lf is None:
        return

    try:
        trace = lf.trace(
            name=f"agent_{workflow_id}",
            metadata={"workflow_id": workflow_id, "params": params},
        )

        for i, step in enumerate(steps):
            trace.span(
                name=f"step_{i}_{step.get('node', 'unknown')}",
                input=step.get("input", ""),
                output=step.get("output", ""),
            )

        trace.span(
            name="final_result",
            output=result[:2000],
        )
    except Exception as e:
        logger.debug(f"LangFuse agent tracing error: {e}")
