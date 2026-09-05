"""Provider-neutral adapter for native/Pydantic structured LLM output."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError

from tradingagents.agents.errors import LLMProviderError, StructuredOutputError

StructuredModel = TypeVar("StructuredModel", bound=BaseModel)
PromptMessages = list[tuple[str, str]]


@dataclass(frozen=True)
class InvocationTelemetry:
    """Best-effort provider telemetry for one structured invocation.

    Latency is always measured locally. Token counts remain ``None`` when the
    provider does not expose them on the structured response.
    """

    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@runtime_checkable
class StructuredOutputClient(Protocol):
    """Minimal interface used by Sprint 4 agents and deterministic fakes."""

    provider: str
    model: str
    temperature: float | None
    seed: int | None
    requested_temperature: float | None
    requested_seed: int | None
    effective_temperature: float | None
    effective_seed: int | None
    last_invocation: InvocationTelemetry | None

    def invoke(
        self,
        schema: type[StructuredModel],
        messages: PromptMessages,
    ) -> StructuredModel: ...


class LangChainStructuredOutputClient:
    """Adapt a configured LangChain chat model without coupling to a provider.

    Sampling controls are configured by the provider client before this adapter
    is constructed. Explicit controls supplied here are verified against the
    wrapped model rather than silently recorded as if they were applied.
    """

    def __init__(
        self,
        llm: Any,
        *,
        provider: str,
        model: str,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> None:
        self._llm = llm
        self.provider = provider
        self.model = model
        observed_temperature = self._observed_float("temperature")
        observed_seed = self._observed_int("seed")
        if (
            temperature is not None
            and observed_temperature is not None
            and temperature != observed_temperature
        ):
            raise ValueError(
                "Requested temperature does not match the configured LangChain model."
            )
        if seed is not None and observed_seed is not None and seed != observed_seed:
            raise ValueError("Requested seed does not match the configured LangChain model.")

        self.requested_temperature = (
            temperature if temperature is not None else observed_temperature
        )
        self.requested_seed = seed if seed is not None else observed_seed
        self.effective_temperature = observed_temperature
        self.effective_seed = observed_seed
        # Compatibility aliases always reflect observed, not merely requested,
        # settings so existing metadata consumers cannot overstate control.
        self.temperature = self.effective_temperature
        self.seed = self.effective_seed
        self.last_invocation: InvocationTelemetry | None = None

    @classmethod
    def from_provider_client(cls, client: Any) -> "LangChainStructuredOutputClient":
        """Build from an existing provider client with capability checks applied."""
        kwargs = getattr(client, "kwargs", {})
        return cls(
            client.get_llm(),
            provider=client.get_provider_name(),
            model=client.model,
            temperature=kwargs.get("temperature"),
            seed=kwargs.get("seed"),
        )

    def _observed_float(self, name: str) -> float | None:
        value = getattr(self._llm, name, None)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    def _observed_int(self, name: str) -> int | None:
        value = getattr(self._llm, name, None)
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value

    def invoke(
        self,
        schema: type[StructuredModel],
        messages: PromptMessages,
    ) -> StructuredModel:
        started = perf_counter()
        try:
            runnable = self._llm.with_structured_output(schema)
            raw = runnable.invoke(messages)
        except (TimeoutError, ConnectionError) as exc:
            raise LLMProviderError(
                f"{self.provider} structured-output request failed: {type(exc).__name__}."
            ) from exc
        except Exception as exc:
            raise LLMProviderError(
                f"{self.provider} structured-output request failed: {type(exc).__name__}."
            ) from exc

        self.last_invocation = InvocationTelemetry(
            latency_ms=round((perf_counter() - started) * 1000.0, 3),
            **self._usage_metadata(raw),
        )

        if isinstance(raw, schema):
            return raw
        try:
            return schema.model_validate(raw)
        except (ValidationError, TypeError, ValueError) as exc:
            raise StructuredOutputError(
                f"Response did not match {schema.__name__}: {type(exc).__name__}."
            ) from exc

    @staticmethod
    def _usage_metadata(raw: Any) -> dict[str, int | None]:
        """Extract provider-neutral token counts when a response exposes them."""
        usage = getattr(raw, "usage_metadata", None)
        if not isinstance(usage, dict):
            response_metadata = getattr(raw, "response_metadata", None)
            usage = (
                response_metadata.get("token_usage")
                if isinstance(response_metadata, dict)
                else None
            )
        if not isinstance(usage, dict):
            return {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            }

        def token_count(*names: str) -> int | None:
            for name in names:
                value = usage.get(name)
                if isinstance(value, int) and not isinstance(value, bool):
                    return value
            return None

        return {
            "input_tokens": token_count("input_tokens", "prompt_tokens"),
            "output_tokens": token_count("output_tokens", "completion_tokens"),
            "total_tokens": token_count("total_tokens"),
        }
