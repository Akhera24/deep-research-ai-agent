"""
Regression tests for provider request-payload shape.

PLAN.md edge case #2: the July 2026 model migration hit two silent
contract changes (Claude 400s on `temperature`; gpt-5.x 400s on
`max_tokens`). These tests pin the exact payload each client sends so
the next model-ID rotation can't silently reintroduce them.

Also covers edge case #13 (tiktoken encoding drift) and #6 (router
provider-fallback output actually used).

All provider SDKs are mocked — no network, no API spend.

Run with: pytest tests/test_payload_shapes.py -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from unittest.mock import MagicMock

import pytest

from src.models.claude_client import ClaudeClient
from src.models.openai_client import OpenAIClient
from src.models.router import ModelRouter
from src.models.base_client import (
    ModelConfig,
    ModelProvider,
    ModelResponse,
    TaskType,
)


def _claude_sdk_response(text="ok"):
    block = MagicMock()
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


def _openai_sdk_response(text="ok"):
    choice = MagicMock()
    choice.message.content = text
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestClaudePayload:
    """claude-opus-4-8 / claude-sonnet-5 reject sampling params with 400."""

    def test_no_sampling_params_sent(self):
        client = ClaudeClient()
        client.client = MagicMock()
        client.client.messages.create.return_value = _claude_sdk_response()

        out = client._make_api_call("hello", use_cache=False)

        assert out == "ok"
        kwargs = client.client.messages.create.call_args.kwargs
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs
        assert "top_k" not in kwargs
        assert kwargs["model"] == client.config.model_name
        assert kwargs["max_tokens"] == client.config.max_tokens

    def test_temperature_kwarg_override_is_ignored(self):
        """Even an explicit caller override must not reach the API."""
        client = ClaudeClient()
        client.client = MagicMock()
        client.client.messages.create.return_value = _claude_sdk_response()

        client._make_api_call("hello", use_cache=False, temperature=0.9)

        kwargs = client.client.messages.create.call_args.kwargs
        assert "temperature" not in kwargs


class TestOpenAIPayload:
    """gpt-5.x rejects `max_tokens` with 400; `max_completion_tokens` is the param."""

    def test_max_completion_tokens_not_max_tokens(self):
        client = OpenAIClient()
        client.client = MagicMock()
        client.client.chat.completions.create.return_value = _openai_sdk_response()

        out = client._make_api_call("hello")

        assert out == "ok"
        kwargs = client.client.chat.completions.create.call_args.kwargs
        assert "max_completion_tokens" in kwargs
        assert "max_tokens" not in kwargs
        # temperature IS still accepted on the OpenAI path (verified live)
        assert kwargs["temperature"] == client.config.temperature

    def test_json_mode_sets_response_format(self):
        client = OpenAIClient()
        client.client = MagicMock()
        client.client.chat.completions.create.return_value = _openai_sdk_response(
            '{"a": 1}'
        )

        out = client._make_api_call("hello", response_format="json")

        assert out == '{"a": 1}'
        kwargs = client.client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}
        assert "max_completion_tokens" in kwargs


class TestTokenizerSelection:
    """tiktoken.encoding_for_model KeyErrors on 'gpt-5.4-*' IDs; the client
    must pick o200k_base explicitly for gpt-5/gpt-4.1/gpt-4o families."""

    def test_default_model_uses_o200k_base(self):
        client = OpenAIClient()  # settings default: gpt-5.4-mini
        assert client.encoder is not None
        assert client.encoder.name == "o200k_base"

    @pytest.mark.parametrize(
        "model_name,expected",
        [
            ("gpt-5.4-mini", "o200k_base"),
            ("gpt-5.5", "o200k_base"),
            ("gpt-4.1", "o200k_base"),
            ("gpt-4o-mini", "o200k_base"),
            ("gpt-4-turbo-preview", "cl100k_base"),
            ("some-future-model", "cl100k_base"),
        ],
    )
    def test_encoding_by_model_family(self, model_name, expected):
        config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name=model_name,
            api_key="test-key-not-used",
        )
        client = OpenAIClient(config)
        assert client.encoder.name == expected


class TestRouterFallback:
    """Edge case #6: primary provider fails -> fallback fires AND its
    output is what the caller receives."""

    def test_fallback_output_used(self):
        router = ModelRouter()
        primary = router.clients[ModelProvider.ANTHROPIC]
        fallback = router.clients[ModelProvider.OPENAI]

        primary.call = MagicMock(side_effect=Exception("simulated 404"))
        fallback_response = ModelResponse(
            content="fallback-ok",
            provider=ModelProvider.OPENAI,
            model_name="gpt-5.4-mini",
            tokens_used=5,
            cost=0.0001,
            latency_ms=1.0,
        )
        fallback.call = MagicMock(return_value=fallback_response)

        # STRATEGY_PLANNING routes ANTHROPIC primary -> OPENAI first fallback
        response = router.route(prompt="x", task_type=TaskType.STRATEGY_PLANNING)

        assert primary.call.called
        assert fallback.call.called
        assert response.content == "fallback-ok"
        assert response.provider == ModelProvider.OPENAI

    def test_all_providers_failing_raises(self):
        router = ModelRouter()
        for client in router.clients.values():
            client.call = MagicMock(side_effect=Exception("simulated outage"))

        with pytest.raises(Exception, match="All models failed"):
            router.route(prompt="x", task_type=TaskType.STRATEGY_PLANNING)
