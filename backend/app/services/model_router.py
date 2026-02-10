"""
Model routing and A/B testing framework.

Routes requests between:
  - Gemini API (general queries, fallback)
  - Fine-tuned local model via Ollama/vLLM (specialized document tasks)

Tracks cost, latency, and quality metrics per model.
"""

import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from app.core.logging import get_logger
from app.config import settings

logger = get_logger(__name__)


class ModelProvider(str, Enum):
    GEMINI_API = "gemini_api"
    LOCAL_FINETUNED = "local_finetuned"


@dataclass
class ModelResponse:
    text: str
    model_provider: ModelProvider
    model_name: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)


# Approximate pricing (per 1M tokens)
COST_PER_1M_TOKENS = {
    ModelProvider.GEMINI_API: {"input": 0.075, "output": 0.30},       # Gemini 2.5 Flash
    ModelProvider.LOCAL_FINETUNED: {"input": 0.0, "output": 0.0},     # Self-hosted = free
}


class ABTestConfig:
    """Configuration for A/B testing between models."""

    def __init__(
        self,
        gemini_weight: float = 0.5,
        local_weight: float = 0.5,
        force_model: Optional[ModelProvider] = None,
    ):
        self.gemini_weight = gemini_weight
        self.local_weight = local_weight
        self.force_model = force_model

    def select_model(self) -> ModelProvider:
        if self.force_model:
            return self.force_model
        return random.choices(
            [ModelProvider.GEMINI_API, ModelProvider.LOCAL_FINETUNED],
            weights=[self.gemini_weight, self.local_weight],
        )[0]


class LocalModelClient:
    """Client for locally-served fine-tuned model (Ollama or vLLM)."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self._available = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = self._check_health()
        return self._available

    def _check_health(self) -> bool:
        try:
            import httpx
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, model_name: str = "doc-intelli") -> str:
        import httpx
        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "num_predict": 512,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]

    def generate_stream(self, prompt: str, model_name: str = "doc-intelli"):
        import httpx
        with httpx.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": 0.1, "top_p": 0.95, "num_predict": 512},
            },
            timeout=120,
        ) as response:
            import json
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars per token for English)."""
    return max(1, len(text) // 4)


def calculate_cost(provider: ModelProvider, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD."""
    rates = COST_PER_1M_TOKENS[provider]
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000


class ModelRouter:
    """Routes requests to the appropriate model based on task type and A/B config."""

    def __init__(self, ab_config: ABTestConfig | None = None):
        self.ab_config = ab_config or ABTestConfig(
            gemini_weight=1.0, local_weight=0.0  # Default: Gemini only
        )
        self.local_client = LocalModelClient()
        self._metrics: list[ModelResponse] = []

    def route(
        self,
        prompt: str,
        task_type: str = "general",
        force_model: ModelProvider | None = None,
    ) -> ModelResponse:
        """Route a request to the selected model and return the response with metrics."""

        # Select model
        if force_model:
            provider = force_model
        elif task_type in ("summarization", "extraction", "classification") and self.local_client.available:
            provider = self.ab_config.select_model()
        else:
            provider = ModelProvider.GEMINI_API

        # Fall back to Gemini if local is unavailable
        if provider == ModelProvider.LOCAL_FINETUNED and not self.local_client.available:
            logger.warning("Local model unavailable, falling back to Gemini API")
            provider = ModelProvider.GEMINI_API

        input_tokens = estimate_tokens(prompt)
        start_time = time.time()

        try:
            if provider == ModelProvider.GEMINI_API:
                from app.services.llm_service import LLMService
                llm = LLMService()
                text = llm.generate(prompt)
                model_name = "gemini-2.5-flash"
            else:
                text = self.local_client.generate(prompt)
                model_name = "doc-intelli-qlora"
        except Exception as e:
            # Fallback on error
            if provider == ModelProvider.LOCAL_FINETUNED:
                logger.warning(f"Local model error: {e}, falling back to Gemini")
                from app.services.llm_service import LLMService
                llm = LLMService()
                text = llm.generate(prompt)
                provider = ModelProvider.GEMINI_API
                model_name = "gemini-2.5-flash"
            else:
                raise

        latency_ms = (time.time() - start_time) * 1000
        output_tokens = estimate_tokens(text)
        cost = calculate_cost(provider, input_tokens, output_tokens)

        response = ModelResponse(
            text=text,
            model_provider=provider,
            model_name=model_name,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

        self._metrics.append(response)
        return response

    def get_cost_report(self) -> dict:
        """Generate a cost and performance report across all tracked requests."""
        if not self._metrics:
            return {"total_requests": 0}

        by_provider = {}
        for m in self._metrics:
            key = m.model_provider.value
            if key not in by_provider:
                by_provider[key] = {
                    "requests": 0, "total_cost": 0.0, "total_latency_ms": 0.0,
                    "total_input_tokens": 0, "total_output_tokens": 0,
                }
            by_provider[key]["requests"] += 1
            by_provider[key]["total_cost"] += m.cost_usd
            by_provider[key]["total_latency_ms"] += m.latency_ms
            by_provider[key]["total_input_tokens"] += m.input_tokens
            by_provider[key]["total_output_tokens"] += m.output_tokens

        for key, stats in by_provider.items():
            n = stats["requests"]
            stats["avg_latency_ms"] = round(stats["total_latency_ms"] / n, 2)
            stats["avg_cost_usd"] = round(stats["total_cost"] / n, 6)
            stats["total_cost"] = round(stats["total_cost"], 6)

        return {
            "total_requests": len(self._metrics),
            "total_cost_usd": round(sum(m.cost_usd for m in self._metrics), 6),
            "by_provider": by_provider,
        }
