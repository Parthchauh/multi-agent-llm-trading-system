import pytest
from pydantic import BaseModel

from tradingagents.agents.errors import LLMProviderError
from tradingagents.agents.structured_output import (
    InvocationTelemetry,
    LangChainStructuredOutputClient,
)
from tradingagents.agents.telemetry import AuditedStructuredClient, CostBasis


class Result(BaseModel):
    ok: bool


class Client:
    provider = 'recorded'
    model = 'test'
    last_invocation = InvocationTelemetry(1., input_tokens=100, output_tokens=50)
    def invoke(self, schema, messages):
        return Result(ok=True)


def test_estimated_cost_requires_explicit_matching_revision_and_known_counts():
    basis = CostBasis('recorded', 'test', 'USD', 'fixture-rate-v1', 2., 4.)
    audited = AuditedStructuredClient(Client(), experiment_id='test', cost_basis=basis)
    audited.invoke(Result, [('system', 'secret-input-not-logged')])
    assert audited.events[0]['estimated_cost'] == pytest.approx(.0004)
    assert audited.events[0]['cost_basis']['revision'] == 'fixture-rate-v1'
    assert audited.events[0]['transport_attempts'] is None
    assert 'secret-input' not in str(audited.events)
    other = AuditedStructuredClient(Client(), experiment_id='test',
        cost_basis=CostBasis('different', 'test', 'USD', 'v1', 2., 4.))
    other.invoke(Result, [])
    assert other.events[0]['estimated_cost'] is None


def test_failed_invocation_has_no_stale_token_counts_or_exception_text():
    class Failed(Client):
        def invoke(self, *args):
            raise ConnectionError('api_key=not-for-logs')
    audited = AuditedStructuredClient(Failed(), experiment_id='test')
    with pytest.raises(ConnectionError):
        audited.invoke(Result, [])
    assert audited.events[0]['input_tokens'] is None
    assert audited.events[0]['failure_type'] == 'ConnectionError'
    assert 'not-for-logs' not in str(audited.events)


def test_provider_adapter_clears_usage_before_a_failed_request():
    class LLM:
        temperature = 0.
        seed = 1
        def with_structured_output(self, schema):
            raise TimeoutError('secret')
    client = LangChainStructuredOutputClient(LLM(), provider='test', model='test')
    client.last_invocation = InvocationTelemetry(1., input_tokens=100)
    with pytest.raises(LLMProviderError):
        client.invoke(Result, [])
    assert client.last_invocation is None
    assert client._usage_metadata(type('R', (), {'usage_metadata': {'input_tokens': -1}})())['input_tokens'] is None


def test_invalid_schema_still_records_fresh_consumed_usage():
    from tradingagents.agents.errors import StructuredOutputError
    class Invalid(Client):
        def invoke(self, *args):
            self.last_invocation = InvocationTelemetry(2., input_tokens=200, output_tokens=100)
            raise StructuredOutputError('invalid output, not logged')
    audited = AuditedStructuredClient(Invalid(), experiment_id='test',
        cost_basis=CostBasis('recorded', 'test', 'USD', 'v1', 2., 4.))
    with pytest.raises(StructuredOutputError):
        audited.invoke(Result, [])
    assert audited.events[0]['schema_valid'] is False
    assert audited.events[0]['input_tokens'] == 200
    assert audited.events[0]['estimated_cost'] == pytest.approx(.0008)
