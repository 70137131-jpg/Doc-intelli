"""Tests for the model router and A/B testing framework."""

import pytest
from unittest.mock import patch, MagicMock

from app.services.model_router import (
    ModelProvider,
    ModelResponse,
    ABTestConfig,
    LocalModelClient,
    ModelRouter,
    estimate_tokens,
    calculate_cost,
    COST_PER_1M_TOKENS,
)


class TestModelProvider:
    def test_enum_values(self):
        assert ModelProvider.GEMINI_API.value == "gemini_api"
        assert ModelProvider.LOCAL_FINETUNED.value == "local_finetuned"


class TestABTestConfig:
    def test_default_weights(self):
        config = ABTestConfig()
        assert config.gemini_weight == 0.5
        assert config.local_weight == 0.5

    def test_force_model(self):
        config = ABTestConfig(force_model=ModelProvider.GEMINI_API)
        for _ in range(20):
            assert config.select_model() == ModelProvider.GEMINI_API

    def test_all_gemini_weight(self):
        config = ABTestConfig(gemini_weight=1.0, local_weight=0.0)
        for _ in range(20):
            assert config.select_model() == ModelProvider.GEMINI_API

    def test_all_local_weight(self):
        config = ABTestConfig(gemini_weight=0.0, local_weight=1.0)
        for _ in range(20):
            assert config.select_model() == ModelProvider.LOCAL_FINETUNED


class TestEstimateTokens:
    def test_basic_estimate(self):
        assert estimate_tokens("hello world") > 0

    def test_empty_string(self):
        assert estimate_tokens("") == 1  # min 1

    def test_longer_text(self):
        text = "a" * 400
        assert estimate_tokens(text) == 100  # ~4 chars per token


class TestCalculateCost:
    def test_gemini_cost(self):
        cost = calculate_cost(ModelProvider.GEMINI_API, 1000, 500)
        assert cost > 0

    def test_local_is_free(self):
        cost = calculate_cost(ModelProvider.LOCAL_FINETUNED, 1000, 500)
        assert cost == 0.0

    def test_cost_scales_with_tokens(self):
        cost_small = calculate_cost(ModelProvider.GEMINI_API, 100, 50)
        cost_large = calculate_cost(ModelProvider.GEMINI_API, 10000, 5000)
        assert cost_large > cost_small


class TestLocalModelClient:
    def test_default_url(self):
        client = LocalModelClient()
        assert client.base_url == "http://localhost:11434"

    def test_unavailable_when_no_server(self):
        client = LocalModelClient(base_url="http://localhost:99999")
        assert client.available is False


class TestModelRouter:
    def test_default_config(self):
        router = ModelRouter()
        assert router.ab_config.gemini_weight == 1.0

    def test_empty_cost_report(self):
        router = ModelRouter()
        report = router.get_cost_report()
        assert report["total_requests"] == 0

    def test_route_to_gemini(self):
        with patch("app.services.llm_service.LLMService") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.generate.return_value = "Test response"
            mock_llm_cls.return_value = mock_llm

            router = ModelRouter()
            response = router.route("Hello", force_model=ModelProvider.GEMINI_API)

            assert response.text == "Test response"
            assert response.model_provider == ModelProvider.GEMINI_API
            assert response.latency_ms >= 0
            assert response.cost_usd >= 0

    def test_cost_report_after_requests(self):
        with patch("app.services.llm_service.LLMService") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.generate.return_value = "Response"
            mock_llm_cls.return_value = mock_llm

            router = ModelRouter()
            router.route("Prompt 1", force_model=ModelProvider.GEMINI_API)
            router.route("Prompt 2", force_model=ModelProvider.GEMINI_API)

            report = router.get_cost_report()
            assert report["total_requests"] == 2
            assert "gemini_api" in report["by_provider"]
            assert report["by_provider"]["gemini_api"]["requests"] == 2
            assert report["total_cost_usd"] >= 0

    def test_local_fallback_to_gemini(self):
        """When local model is unavailable, should fall back to Gemini."""
        router = ModelRouter(
            ab_config=ABTestConfig(gemini_weight=0.0, local_weight=1.0)
        )
        # Local client won't be available
        router.local_client._available = False
        with patch("app.services.llm_service.LLMService") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.generate.return_value = "Fallback response"
            mock_llm_cls.return_value = mock_llm

            response = router.route("Test prompt", task_type="summarization")
            assert response.model_provider == ModelProvider.GEMINI_API
