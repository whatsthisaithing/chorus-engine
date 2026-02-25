"""Interactive narrative v1 loop-kind plugin."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from chorus_engine.ens.loop_plugins.contracts import LoopStepPlan, StepOutcomePolicy, StepPassPlan


_NARRATIVE_V1_WAIT_EQUIVALENTS = {"WAIT_FOR_USER", "YIELD"}


class NarrativeV1LoopPlugin:
    plugin_id = "interactive_narrative_v1"

    def kind(self) -> str:
        return "narrative.v1"

    def build_step_plan(
        self,
        *,
        split_enabled: bool,
        tool_transport_mode: str,
    ) -> LoopStepPlan:
        enable_outcome_pass = bool(split_enabled)
        primary_pass_label = "beat" if enable_outcome_pass else "full"
        outcome_policy = None
        if enable_outcome_pass:
            outcome_policy = StepOutcomePolicy(
                control_tool_name="chorus.control",
                allowed_actions={"CONTINUE", "YIELD", "COMPLETE"},
                default_action="WAIT_FOR_USER",
                use_native_transport=(str(tool_transport_mode or "").strip().lower() == "native"),
                tool_choice="auto",
            )
        passes: List[StepPassPlan] = [
            StepPassPlan(
                pass_id="pass_primary_generation",
                kind="primary_generation",
                emit_to_user=True,
                parse_strategy="none",
                loop_stage_label=primary_pass_label,
            )
        ]
        if enable_outcome_pass and outcome_policy is not None:
            passes.append(
                StepPassPlan(
                    pass_id="pass_outcome_resolution",
                    kind="outcome_resolution",
                    emit_to_user=False,
                    parse_strategy="outcome_ladder",
                    native_tool_policy={
                        "policy_id": f"{self.plugin_id}.outcome_control",
                        "allowed_media_tools": [],
                        "include_control": bool(outcome_policy.use_native_transport),
                        "include_cold_recall": False,
                        "tool_choice": outcome_policy.tool_choice,
                    },
                    temperature=float(outcome_policy.temperature),
                    max_tokens=int(outcome_policy.max_tokens),
                    allow_single_tool_loopback=False,
                    loop_stage_label="control",
                )
            )
        return LoopStepPlan(
            enable_outcome_pass=enable_outcome_pass,
            primary_pass_label=primary_pass_label,
            prompt_addendum=self.loop_step_prompt_addendum(stage=primary_pass_label),
            passes=passes,
            outcome_policy=outcome_policy,
            loop_policy={"max_consecutive_continue": 4},
            use_prompt_assembly_context=True,
            force_wait_when_missing_control=True,
            control_policy_id="narrative.v1.default",
        )

    @staticmethod
    def loop_step_prompt_addendum(*, stage: str = "full") -> str:
        stage_norm = str(stage or "full").strip().lower()
        base = [
            "Loop Step Mode (Mandatory):",
            "- This message is one loop step.",
            "- Write one narrative beat only.",
            "- Do not encode control decisions in prose.",
        ]
        if stage_norm != "beat":
            base.extend(
                [
                    "- You MUST include a control payload in the Chorus sentinel block.",
                    "- Choose exactly one action: CONTINUE, WAIT_FOR_USER, COMPLETE, or YIELD.",
                    "- Do not emit tool calls unless explicitly allowed for this loop kind.",
                ]
            )
        else:
            base.extend(
                [
                    "- Control selection is handled in a separate control-evaluation stage.",
                    "- Do not emit control payloads or control tool calls in this stage.",
                ]
            )
        base.extend(
            [
                "",
                "Interactive Narrative Control Selection Rules:",
                "- Advance immediate consequences or NPC/environment beats without removing user agency.",
                "- Ask questions only when meaningful user choice is required.",
                "- Keep progression natural; do not force cliffhangers every beat.",
            ]
        )
        return "\n".join(base)

    def build_outcome_messages(
        self,
        *,
        beat_text: str,
        user_input_text: str,
        step_index: int,
        loop_kind: str,
        tool_transport_mode: str,
    ) -> List[Dict[str, str]]:
        native_transport = str(tool_transport_mode or "sentinel").strip().lower() == "native"
        system_lines = [
            "You are a loop control evaluator.",
            "Decide exactly one control action for the next loop step.",
            "",
            "Allowed actions:",
            "- CONTINUE: advance autoplay immediately.",
            "- YIELD: pause for user input.",
            "- COMPLETE: end the loop.",
            "",
            "Policy:",
            "- Default to CONTINUE.",
            "- Choose YIELD when the beat asks the user a direct question, requires a choice, or clearly pauses/waits for input.",
            "- Choose COMPLETE when the beat clearly ends the scene/story arc.",
            "- Output no prose.",
        ]
        if native_transport:
            system_lines.extend(
                [
                    "- Emit exactly one native `chorus.control` tool call with {\"action\":\"CONTINUE|YIELD|COMPLETE\"}.",
                    "- Do not include JSON/tool text in visible content.",
                ]
            )
        else:
            system_lines.extend(
                [
                    "- Emit exactly one Chorus sentinel payload containing control.action.",
                    "- No visible prose outside the payload/sentinel requirement.",
                ]
            )
        user_text = (
            f"Loop kind: {loop_kind}\n"
            f"Step index: {step_index}\n"
            f"Last user input: {user_input_text or 'continue'}\n\n"
            "Beat text:\n"
            f"{beat_text or ''}"
        )
        return [
            {"role": "system", "content": "\n".join(system_lines)},
            {"role": "user", "content": user_text},
        ]

    def outcome_json_schema_response_format(self) -> Dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "loop_control_action",
                "schema": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["CONTINUE", "YIELD", "COMPLETE"],
                            "description": "Loop control action",
                        }
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

    def build_outcome_retry_messages(self, *, beat_text: str) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a strict JSON generator. Return only JSON that matches the provided schema. "
                    "No prose and no code fences."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Beat text:\n"
                    f"{beat_text or ''}\n\n"
                    "Choose action:\n"
                    "- CONTINUE if the story can progress without user input.\n"
                    "- YIELD if the beat asks a direct question or requires user choice.\n"
                    "- COMPLETE if the scene has clearly ended.\n\n"
                    "Return JSON with a single field: action."
                ),
            },
        ]

    def normalize_step_outcome(self, action: Optional[str]) -> Optional[str]:
        value = str(action or "").strip().upper()
        if not value:
            return None
        if value in _NARRATIVE_V1_WAIT_EQUIVALENTS:
            return "WAIT_FOR_USER"
        return value

    def post_step_hooks(self, *, step_context: Dict[str, Any]) -> None:
        _ = step_context


class GenericLoopPlugin:
    plugin_id = "generic_loop"

    def kind(self) -> str:
        return "generic"

    def build_step_plan(
        self,
        *,
        split_enabled: bool,
        tool_transport_mode: str,
    ) -> LoopStepPlan:
        _ = (split_enabled, tool_transport_mode)
        return LoopStepPlan(
            enable_outcome_pass=False,
            primary_pass_label="full",
            prompt_addendum=self.loop_step_prompt_addendum(stage="full"),
            passes=[
                StepPassPlan(
                    pass_id="pass_primary_generation",
                    kind="primary_generation",
                    emit_to_user=True,
                    parse_strategy="none",
                    loop_stage_label="full",
                )
            ],
            outcome_policy=None,
            loop_policy={},
            use_prompt_assembly_context=False,
            force_wait_when_missing_control=False,
            control_policy_id="generic.default",
        )

    @staticmethod
    def loop_step_prompt_addendum(*, stage: str = "full") -> str:
        _ = stage
        return (
            "Loop Step Mode (Mandatory):\n"
            "- This message is one loop step.\n"
            "- Write one narrative beat only.\n"
            "- Do not encode control decisions in prose."
        )

    def build_outcome_messages(
        self,
        *,
        beat_text: str,
        user_input_text: str,
        step_index: int,
        loop_kind: str,
        tool_transport_mode: str,
    ) -> List[Dict[str, str]]:
        _ = (beat_text, user_input_text, step_index, loop_kind, tool_transport_mode)
        return []

    def outcome_json_schema_response_format(self) -> Dict[str, Any]:
        return {}

    def build_outcome_retry_messages(self, *, beat_text: str) -> List[Dict[str, str]]:
        _ = beat_text
        return []

    def normalize_step_outcome(self, action: Optional[str]) -> Optional[str]:
        value = str(action or "").strip().upper()
        return value or None

    def post_step_hooks(self, *, step_context: Dict[str, Any]) -> None:
        _ = step_context
