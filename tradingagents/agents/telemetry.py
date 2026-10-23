"""Provider-neutral invocation audit without persisting prompts or response text."""

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass
from time import perf_counter


@dataclass(frozen=True)
class CostBasis:
    provider: str
    model: str
    currency: str
    revision: str
    input_per_million: float
    output_per_million: float

    def __post_init__(self):
        if not all((self.provider, self.model, self.currency, self.revision)):
            raise ValueError("Cost basis needs provider, model, currency and revision.")
        for value in (self.input_per_million, self.output_per_million):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("Token rates must be finite and non-negative.")


class AuditedStructuredClient:
    def __init__(self, client, *, experiment_id, prompt_version="research:v1", cost_basis=None):
        self.client = client
        self.experiment_id = experiment_id
        self.prompt_version = prompt_version
        self.events = []
        self.cost_basis = cost_basis

    def __getattr__(self, name):
        return getattr(self.client, name)

    def invoke(self, schema, messages):
        started = perf_counter()
        previous_telemetry = getattr(self.client, 'last_invocation', None)
        event = {
            "experiment_id": self.experiment_id,
            "invocation_id": uuid.uuid4().hex,
            "agent_role": schema.__name__,
            "provider": getattr(self.client, "provider", None),
            "model": getattr(self.client, "model", None),
            "prompt_version": self.prompt_version,
            "prompt_sha256": hashlib.sha256(json.dumps(messages).encode()).hexdigest(),
            "attempt": 1 + sum(e["agent_role"] == schema.__name__ for e in self.events),
            "input_tokens": None,
            "output_tokens": None,
            "estimated_cost": None,
            "cost_basis": None,
            "transport_attempts": None,
        }
        try:
            result = self.client.invoke(schema, messages)
            event["schema_valid"] = True
            event["output_sha256"] = hashlib.sha256(result.model_dump_json().encode()).hexdigest()
            return result
        except Exception as exc:
            event["schema_valid"] = False
            event["failure_type"] = type(exc).__name__
            raise
        finally:
            event["latency_ms"] = (perf_counter() - started) * 1000
            telemetry = getattr(self.client, "last_invocation", None)
            if telemetry is not None and (event["schema_valid"] or telemetry is not previous_telemetry):
                event["input_tokens"] = telemetry.input_tokens
                event["output_tokens"] = telemetry.output_tokens
            basis = self.cost_basis
            if basis and (basis.provider, basis.model) == (event['provider'], event['model']):
                counts = (event['input_tokens'], event['output_tokens'])
                if all(isinstance(count, int) and not isinstance(count, bool) and count >= 0 for count in counts):
                    event['estimated_cost'] = (counts[0]*basis.input_per_million + counts[1]*basis.output_per_million)/1_000_000
                    event['cost_basis'] = asdict(basis)
            self.events.append(event)
