"""SQLAlchemy models for ENS sessions and observability records."""

from datetime import datetime
import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, JSON, Index, text

from chorus_engine.db.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class ENSSession(Base):
    """Canonical ENS session mapping across surfaces."""

    __tablename__ = "ens_sessions"

    session_id = Column(String(36), primary_key=True, default=_uuid)
    assistant_id = Column(String(50), nullable=False, index=True)
    user_id = Column(String(200), nullable=False, index=True)
    conversation_id = Column(String(36), nullable=True, index=True)
    thread_id = Column(String(36), nullable=True, index=True)
    surface = Column(String(20), nullable=False)
    source = Column(String(20), nullable=False)
    external_session_key = Column(String(200), nullable=True, index=True)
    latency_sensitive = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_signal_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_ens_sessions_surface_source_thread", "surface", "source", "thread_id", unique=True),
    )


class ENSDecision(Base):
    """Decision record persisted for each ENS ingest cycle."""

    __tablename__ = "ens_decisions"

    decision_id = Column(String(36), primary_key=True, default=_uuid)
    trace_id = Column(String(36), nullable=True, index=True)
    signal_id = Column(String(36), nullable=False, index=True)
    session_id = Column(String(36), nullable=True, index=True)
    assistant_id = Column(String(50), nullable=True, index=True)
    user_id = Column(String(200), nullable=True, index=True)
    scope = Column(String(20), nullable=False)
    signal_type = Column(String(100), nullable=False)
    appraisal_json = Column(JSON, nullable=True)
    constraints_json = Column(JSON, nullable=False, default=list)
    intent_proposals_json = Column(JSON, nullable=False, default=list)
    arbitration_json = Column(JSON, nullable=True)
    actions_json = Column(JSON, nullable=False, default=list)
    explanation = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class ENSActionResult(Base):
    """Execution record for each dispatched ENS action."""

    __tablename__ = "ens_action_results"

    action_result_id = Column(String(36), primary_key=True, default=_uuid)
    decision_id = Column(String(36), ForeignKey("ens_decisions.decision_id", ondelete="CASCADE"), nullable=False, index=True)
    action_id = Column(String(36), nullable=False, index=True)
    idempotency_key = Column(String(255), nullable=True, index=True)
    kind = Column(String(100), nullable=False)
    execution_class = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False)
    error_code = Column(String(100), nullable=True)
    error_message = Column(Text, nullable=True)
    metrics_json = Column(JSON, nullable=True)
    output_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class ENSToolCallRequest(Base):
    """Persistent tool call request for ENS tool dispatch and replay safety."""

    __tablename__ = "ens_tool_call_requests"

    tool_call_id = Column(String(100), primary_key=True)
    session_id = Column(String(36), nullable=False, index=True)
    assistant_message_id = Column(String(36), nullable=True, index=True)
    tool_name = Column(String(100), nullable=False, index=True)
    args_json = Column(JSON, nullable=False, default=dict)
    status = Column(String(20), nullable=False, index=True, default="pending")
    idempotency_key = Column(String(255), nullable=False, index=True)
    result_ref = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_ens_tool_call_requests_session_status", "session_id", "status"),
    )


class SurfaceBinding(Base):
    """Canonical surface mapping used by ENS routing resolution."""

    __tablename__ = "surface_bindings"

    id = Column(String(36), primary_key=True, default=_uuid)
    surface_id = Column(String(20), nullable=False, index=True)
    surface_instance_id = Column(String(100), nullable=False, default="", index=True)
    external_thread_id = Column(String(255), nullable=False, index=True)
    relationship_id = Column(String(36), nullable=True, index=True)
    conversation_id = Column(String(36), nullable=False, index=True)
    thread_id = Column(String(36), nullable=False, index=True)
    owner_user_id = Column(String(200), nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_seen_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index(
            "uq_surface_bindings_lookup",
            "surface_id",
            "surface_instance_id",
            "external_thread_id",
            unique=True,
        ),
    )


class SurfaceEgressIntent(Base):
    """Durable outbound surface delivery intent (ENS outbox)."""

    __tablename__ = "surface_egress_intents"

    id = Column(String(36), primary_key=True, default=_uuid)
    surface_id = Column(String(20), nullable=False, index=True)
    surface_instance_id = Column(String(100), nullable=False, default="", index=True)
    external_thread_id = Column(String(255), nullable=False, index=True)
    relationship_id = Column(String(36), nullable=True, index=True)
    conversation_id = Column(String(36), nullable=True, index=True)
    thread_id = Column(String(36), nullable=True, index=True)
    in_reply_to_message_id = Column(String(36), nullable=True, index=True)
    payload_json = Column(JSON, nullable=False, default=dict)
    status = Column(String(20), nullable=False, default="pending", index=True)
    attempt_count = Column(Integer, nullable=False, default=0)
    next_attempt_at = Column(DateTime, nullable=True)
    last_error = Column(Text, nullable=True)
    idempotency_key = Column(String(255), nullable=False, index=True)
    trace_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("uq_surface_egress_intents_idempotency_key", "idempotency_key", unique=True),
        Index("ix_surface_egress_intents_status_surface", "status", "surface_id"),
    )


class ENSSignalQueue(Base):
    """Persistent ENS v3 signal queue entries."""

    __tablename__ = "ens_signal_queue"

    queue_id = Column(String(36), primary_key=True, default=_uuid)
    signal_id = Column(String(36), nullable=False, unique=True, index=True)
    signal_type = Column(String(100), nullable=False, index=True)
    relationship_id = Column(String(36), nullable=True, index=True)
    loop_id = Column(String(36), nullable=True, index=True)
    conversation_id = Column(String(36), nullable=True, index=True)
    surface_id = Column(String(20), nullable=True, index=True)
    priority_tier = Column(String(20), nullable=False, index=True, default="system")
    created_at_us = Column(Integer, nullable=False, index=True)
    idempotency_key = Column(String(255), nullable=True, index=True)
    signal_json = Column(JSON, nullable=False, default=dict)
    status = Column(String(20), nullable=False, default="pending", index=True)
    selected_at = Column(DateTime, nullable=True)
    claimed_at_us = Column(Integer, nullable=True, index=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_ens_signal_queue_status_priority_created", "status", "priority_tier", "created_at_us"),
        Index("ix_ens_signal_queue_surface_rel_status", "surface_id", "relationship_id", "status"),
        Index("uq_ens_signal_queue_idempotency_key", "idempotency_key", unique=True),
        Index(
            "uq_ens_signal_queue_pending_loop_progression_by_loop",
            "loop_id",
            unique=True,
            sqlite_where=text(
                "signal_type = 'loop_progression' AND status = 'pending' AND loop_id IS NOT NULL"
            ),
        ),
    )


class ENSLoopSession(Base):
    """Persistent loop session tracked as a signal producer state machine."""

    __tablename__ = "ens_loop_sessions"

    loop_id = Column(String(36), primary_key=True, default=_uuid)
    loop_kind = Column(String(100), nullable=False, index=True)
    loop_mode = Column(String(20), nullable=False, default="visible", index=True)
    relationship_id = Column(String(36), nullable=False, index=True)
    conversation_id = Column(String(36), nullable=True, index=True)
    surface_id = Column(String(20), nullable=True, index=True)
    step_index = Column(Integer, nullable=False, default=0)
    step_count = Column(Integer, nullable=False, default=0)
    token_budget_used = Column(Integer, nullable=False, default=0)
    tool_budget_used = Column(Integer, nullable=False, default=0)
    state = Column(String(30), nullable=False, default="running", index=True)
    stop_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_ens_loop_sessions_relationship_state", "relationship_id", "state"),
        Index("ix_ens_loop_sessions_surface_state", "surface_id", "state"),
    )


class ENSLoopStepEvent(Base):
    """Auditable per-step loop execution event."""

    __tablename__ = "ens_loop_step_events"

    event_id = Column(String(36), primary_key=True, default=_uuid)
    loop_id = Column(String(36), nullable=False, index=True)
    signal_id = Column(String(36), nullable=True, index=True)
    tick_id = Column(String(36), nullable=True, index=True)
    decision_id = Column(String(36), nullable=True, index=True)
    action_id = Column(String(36), nullable=True, index=True)
    relationship_id = Column(String(36), nullable=True, index=True)
    conversation_id = Column(String(36), nullable=True, index=True)
    surface_id = Column(String(20), nullable=True, index=True)
    step_index_before = Column(Integer, nullable=False, default=0)
    step_index_after = Column(Integer, nullable=False, default=0)
    step_count_after = Column(Integer, nullable=False, default=0)
    state_before = Column(String(30), nullable=True, index=True)
    state_after = Column(String(30), nullable=True, index=True)
    control_action = Column(String(30), nullable=True, index=True)
    tool_requests_count = Column(Integer, nullable=False, default=0)
    provider_finish_reason = Column(String(30), nullable=True)
    output_json = Column(JSON, nullable=True)
    created_at_us = Column(Integer, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_ens_loop_step_events_loop_created", "loop_id", "created_at_us"),
        Index("ix_ens_loop_step_events_signal", "signal_id", "created_at_us"),
        Index(
            "uq_ens_loop_step_events_loop_signal",
            "loop_id",
            "signal_id",
            unique=True,
            sqlite_where=text("signal_id IS NOT NULL"),
        ),
    )


class ENSSchedulerTick(Base):
    """Persistent scheduler selection trace for ENS v3."""

    __tablename__ = "ens_scheduler_ticks"

    tick_id = Column(String(36), primary_key=True, default=_uuid)
    queue_id = Column(String(36), ForeignKey("ens_signal_queue.queue_id", ondelete="SET NULL"), nullable=True, index=True)
    selected_signal_id = Column(String(36), nullable=True, index=True)
    reason_trace_json = Column(JSON, nullable=False, default=dict)
    tie_break_json = Column(JSON, nullable=True)
    created_at_us = Column(Integer, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class ENSFloorControlState(Base):
    """Relationship-level floor control and attention lock state.

    This state is used only to influence scheduling fairness/cooldown weighting.
    It must not hard-gate runnable signals behind surface reopen semantics.
    """

    __tablename__ = "ens_floor_control_state"

    id = Column(String(36), primary_key=True, default=_uuid)
    relationship_id = Column(String(36), nullable=False, index=True)
    active_surface_id = Column(String(20), nullable=True, index=True)
    attention_lock_until_us = Column(Integer, nullable=True, index=True)
    lock_source_signal_id = Column(String(36), nullable=True, index=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("uq_ens_floor_control_state_relationship_id", "relationship_id", unique=True),
        Index("ix_ens_floor_control_state_surface_lock", "active_surface_id", "attention_lock_until_us"),
    )
