"""Loop-kind plugin contracts for ENS loop progression behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, Set

StepPassKind = Literal["primary_generation", "outcome_resolution"]


@dataclass
class StepOutcomePolicy:
    control_tool_name: str = "chorus.control"
    allowed_actions: Set[str] = field(default_factory=lambda: {"CONTINUE", "YIELD", "COMPLETE"})
    default_action: str = "WAIT_FOR_USER"
    use_native_transport: bool = True
    tool_choice: Any = "auto"
    max_tokens: int = 64
    temperature: float = 0.1


@dataclass
class StepPassPlan:
    pass_id: str
    kind: StepPassKind
    emit_to_user: bool = False
    parse_strategy: str = "none"
    native_tool_policy: Optional[Dict[str, Any]] = None
    response_format: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    allow_single_tool_loopback: bool = False
    loop_stage_label: Optional[str] = None


@dataclass
class LoopStepPlan:
    enable_outcome_pass: bool
    primary_pass_label: str
    prompt_addendum: str
    passes: List[StepPassPlan] = field(default_factory=list)
    outcome_policy: Optional[StepOutcomePolicy] = None
    loop_policy: Dict[str, Any] = field(default_factory=dict)
    use_prompt_assembly_context: bool = False
    force_wait_when_missing_control: bool = False
    control_policy_id: str = "default"


@dataclass
class StepOutcomeResolution:
    action: str
    ladder_rung_selected: str
    defaulted_wait: bool
    rung1_native: Dict[str, Any]
    rung2_parse: Dict[str, Any]
    rung3_json_schema: Dict[str, Any]
    source_stage: str = "default_wait"
    source_assistant_result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class PassExecutionResult:
    pass_id: str
    status: str
    output_text: str
    assistant_result_tier: Optional[str]
    finish_reason: Optional[str]
    tool_calls_count: int
    loopback_invoked: bool = False
    error: Optional[str] = None
    timing_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepExecutionAggregate:
    pass_results: List[PassExecutionResult]
    final_visible_text: str
    visible_pass_id: Optional[str]
    final_outcome_action: Optional[str]
    final_control_source: Optional[str]
    defaulted_wait: bool


class LoopKindPlugin(Protocol):
    plugin_id: str

    def kind(self) -> str:
        ...

    def build_step_plan(
        self,
        *,
        split_enabled: bool,
        tool_transport_mode: str,
    ) -> LoopStepPlan:
        ...

    def build_outcome_messages(
        self,
        *,
        beat_text: str,
        user_input_text: str,
        step_index: int,
        loop_kind: str,
        tool_transport_mode: str,
    ) -> List[Dict[str, str]]:
        ...

    def build_outcome_retry_messages(self, *, beat_text: str) -> List[Dict[str, str]]:
        ...

    def outcome_json_schema_response_format(self) -> Dict[str, Any]:
        ...

    def normalize_step_outcome(self, action: Optional[str]) -> Optional[str]:
        ...

    def post_step_hooks(self, *, step_context: Dict[str, Any]) -> None:
        ...
