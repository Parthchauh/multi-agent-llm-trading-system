"""Provider-neutral structured-output configuration and telemetry tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from tradingagents.agents.structured_output import LangChainStructuredOutputClient
from tradingagents.llm_clients import anthropic_client, google_client, openai_client
from tradingagents.llm_clients.anthropic_client import AnthropicClient
from tradingagents.llm_clients.google_client import GoogleClient
from tradingagents.llm_clients.openai_client import OpenAIClient


class _Response(BaseModel):
    value: int


class _Runnable:
    def __init__(self, response: object) -> None:
        self.response = response

    def invoke(self, messages):
        return self.response


class _LLM:
    def __init__(self, *, temperature: float | None = None, seed: int | None = None) -> None:
        self.temperature = temperature
        self.seed = seed

    def with_structured_output(self, schema):
        return _Runnable({"value": 7})


def test_adapter_records_observed_sampling_and_latency() -> None:
    client = LangChainStructuredOutputClient(
        _LLM(temperature=0.0, seed=17),
        provider="fake",
        model="fake-model",
        temperature=0.0,
        seed=17,
    )
    assert client.invoke(_Response, [("system", "test")]) == _Response(value=7)
    assert client.requested_temperature == 0.0
    assert client.effective_temperature == 0.0
    assert client.requested_seed == 17
    assert client.effective_seed == 17
    assert client.last_invocation is not None
    assert client.last_invocation.latency_ms >= 0.0
    assert client.last_invocation.total_tokens is None


def test_adapter_rejects_sampling_metadata_that_disagrees_with_model() -> None:
    with pytest.raises(ValueError, match="temperature"):
        LangChainStructuredOutputClient(
            _LLM(temperature=0.2),
            provider="fake",
            model="fake-model",
            temperature=0.1,
        )


def test_adapter_extracts_available_provider_usage_metadata() -> None:
    usage = LangChainStructuredOutputClient._usage_metadata(
        SimpleNamespace(
            usage_metadata={"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}
        )
    )
    assert usage == {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}


def test_provider_client_builder_derives_configured_values() -> None:
    class _ProviderClient:
        model = "fake-model"
        kwargs = {"temperature": 0.3, "seed": 9}

        def get_provider_name(self):
            return "fake"

        def get_llm(self):
            return _LLM(temperature=0.3, seed=9)

    client = LangChainStructuredOutputClient.from_provider_client(_ProviderClient())
    assert client.provider == "fake"
    assert client.effective_temperature == 0.3
    assert client.effective_seed == 9


def _disable_model_warning(monkeypatch, client) -> None:
    monkeypatch.setattr(client, "warn_if_unknown_model", lambda: None)


def test_google_forwards_supported_temperature_and_seed(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        google_client,
        "NormalizedChatGoogleGenerativeAI",
        lambda **kwargs: captured.update(kwargs) or kwargs,
    )
    client = GoogleClient("gemini-test", temperature=0.2, seed=8)
    _disable_model_warning(monkeypatch, client)
    client.get_llm()
    assert captured["temperature"] == 0.2
    assert captured["seed"] == 8


def test_anthropic_forwards_temperature_and_rejects_seed(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        anthropic_client,
        "NormalizedChatAnthropic",
        lambda **kwargs: captured.update(kwargs) or kwargs,
    )
    client = AnthropicClient("claude-test", temperature=0.2)
    _disable_model_warning(monkeypatch, client)
    client.get_llm()
    assert captured["temperature"] == 0.2

    seeded = AnthropicClient("claude-test", seed=8)
    _disable_model_warning(monkeypatch, seeded)
    with pytest.raises(ValueError, match="seed"):
        seeded.get_llm()


def test_openai_compatible_chat_forwards_sampling_and_native_seed_fails(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        openai_client,
        "NormalizedChatOpenAI",
        lambda **kwargs: captured.update(kwargs) or kwargs,
    )
    compatible = OpenAIClient(
        "compatible-test", provider="openrouter", temperature=0.2, seed=8
    )
    _disable_model_warning(monkeypatch, compatible)
    compatible.get_llm()
    assert captured["temperature"] == 0.2
    assert captured["seed"] == 8

    native = OpenAIClient("gpt-4.1", provider="openai", seed=8)
    _disable_model_warning(monkeypatch, native)
    with pytest.raises(ValueError, match="seed"):
        native.get_llm()
