"""ENS runtime data models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid


def _id() -> str:
    return str(uuid.uuid4())


@dataclass
class SignalEnvelope:
    """Normalized ingress signal for ENS processing."""

    type: str
    scope: str
    source: str
    payload: Dict[str, Any]
    assistant_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    priority_hint: Optional[int] = None
    blocking_hint: Optional[bool] = None
    tags: List[str] = field(default_factory=list)
    signal_id: str = field(default_factory=_id)
    trace_id: str = field(default_factory=_id)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    schema_version: str = "ens.v2"


@dataclass
class ENSAction:
    """Action selected by ENS for execution."""

    kind: str
    params: Dict[str, Any]
    execution_class: str = "user_facing"
    action_id: str = field(default_factory=_id)
    idempotency_key: Optional[str] = None


@dataclass
class ENSOutcome:
    """Result of ENS processing used by endpoint adapters."""

    decision_id: str
    trace_id: str
    signal_id: str
    actions: List[ENSAction]
    action_results: List[Dict[str, Any]]
    response_payload: Dict[str, Any] = field(default_factory=dict)
