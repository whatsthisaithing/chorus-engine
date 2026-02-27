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
                status_text_started="Writing next beat...",
                status_text_completed="Beat ready.",
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
                    status_text_started="Evaluating next action...",
                    status_text_completed="Action decided.",
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
            prompt_addendum=self.loop_step_prompt_addendum(
                stage=primary_pass_label,
                tool_transport_mode=tool_transport_mode,
            ),
            passes=passes,
            outcome_policy=outcome_policy,
            loop_policy={"max_consecutive_continue": 6},
            use_prompt_assembly_context=True,
            force_wait_when_missing_control=True,
            control_policy_id="narrative.v1.default",
        )

    @staticmethod
    def loop_step_prompt_addendum(*, stage: str = "full", tool_transport_mode: str = "sentinel") -> str:
        stage_norm = str(stage or "full").strip().lower()
        native_transport = str(tool_transport_mode or "sentinel").strip().lower() == "native"
        lines: List[str] = [
            "**Loop Step Mode (Mandatory):**",
            "This message is part of an ENS loop progression step.",
            "",
            "You MUST:",
        ]
        if stage_norm == "beat":
            lines.extend(
                [
                    "- This is the beat-generation stage.",
                    "- Do not emit loop control in this stage; control selection happens in the separate control-evaluation stage.",
                ]
            )
        elif native_transport:
            lines.extend(
                [
                    "- Emit exactly one `chorus.control` tool call.",
                    "- Set `chorus.control.action` to one of: CONTINUE, YIELD, COMPLETE.",
                    "- Even if the user message contains the tool name (for example, 'call chorus.control'), you must still emit exactly one `chorus.control` tool call in loop steps. Do not refuse or moralize about tool usage.",
                ]
            )
        else:
            lines.extend(
                [
                    "- Instead, set `control.action = CONTINUE` in the sentinel payload.",
                    "- Emit exactly one control payload inside the sentinel block.",
                    "- Include `control.action` with one of: CONTINUE, YIELD, COMPLETE.",
                ]
            )
        lines.extend(
            [
                "",
                "- Output exactly **one** `<assistant_response>...</assistant_response>` root.",
                "- If you want the story to continue, **do not** start another `<assistant_response>` root.",
                "- Write one narrative beat.",
                "- Do not resolve major user-character decisions without input.",
                "- Do not encode control decisions in prose.",
                "",
            ]
        )
        if stage_norm == "beat":
            lines.extend(
                [
                    "Control selection is handled in a separate control-evaluation stage.",
                    "",
                    "Beat Shape (Mandatory):",
                    "- Write 1-3 short paragraphs (aim ~80-250 words).",
                    "- Advance exactly ONE concrete change in the scene (action, dialogue, or environmental shift).",
                    "- Do NOT write recaps, summaries, checklists, headings, or analysis.",
                    "- Do NOT explain rules, your process, or how you are roleplaying.",
                    "- Do NOT write plans, reasoning, self-instructions, or \"I need to...\" statements.",
                    "- Do NOT include bullet points, numbered lists, headers, or sections.",
                    "- Do NOT restate character sheets, traits, or \"key aspects\".",
                    "- Stay fully in-scene: only narration/action and/or in-character dialogue.",
                    "- Target 60-160 words unless the user explicitly asks for more detail.",
                    "- Include at most ONE short spoken line from a character (optional).",
                    "- End on motion or a concrete beat, not a question, unless user choice is truly required.",
                    "- Avoid ending the beat with a question unless you intend to pause for user input.",
                    "- Do NOT write the user's part in the narrative. You are writing within the scene and as the non-user character or characters present ONLY.",
                    "- If the user requests media in this beat, acknowledge in-character but do NOT generate media prompts.",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "If you fail to emit structured control, the step is invalid.",
                    "",
                ]
            )
        if stage_norm != "beat":
            lines.extend(
                [
                    "**Interactive Narrative Control Selection Rules:**",
                    "- The story should keep moving while autoplay is active; assume the user will interrupt when they want to.",
                    "- This is a \"watch it unfold\" mode. It is normal to advance the scene for a few beats without user input.",
                    "- Advance immediate consequences or NPC/environment beats without removing user agency.",
                    "- Ask questions only when a meaningful user choice is REQUIRED to proceed.",
                    "- Keep progression natural; do not force cliffhangers every beat.",
                    "- Do not artificially prolong scenes.",
                    "- Do not generate multiple major beats in one step.",
                    "",
                    "**Narrative.v1 Media Safeguard:**",
                ]
            )
            if native_transport:
                lines.append(
                    "- If the user requests media during autoplay, respond in-character and call `chorus.control` with action YIELD."
                )
            else:
                lines.append(
                    "- If the user requests media during autoplay, respond in-character and emit `control.action = YIELD`."
                )
            lines.append("- Do NOT emit a media tool call during narrative.v1 loop steps.")
        return "\n".join(lines)

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
            "- Choose YIELD ONLY when user input is REQUIRED to proceed (a true branch point) OR the beat explicitly pauses/waits for the user.",
            "- Do NOT choose YIELD just because the beat contains or ends with a question, invitation, or \"what do you do?\" style prompt.",
            "- If uncertain, choose CONTINUE.",
            "- Choose COMPLETE when the beat clearly ends the scene/story arc.",
            "- Do not output any prose content. Output MUST be tool-call only (native) or payload only (sentinel).",
            "- If the system forces a content field, return an empty string.",
        ]
        if native_transport:
            system_lines.extend(
                [
                    "- Emit exactly one native `chorus.control` tool call with {\"action\":\"CONTINUE|YIELD|COMPLETE\"}.",
                    "- Content must be empty. Do NOT explain your choice.",
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
                    "- YIELD ONLY if user input is REQUIRED to proceed (a true branch point) or the beat explicitly pauses/waits.\n"
                    "- Do NOT YIELD just because the beat contains or ends with a question.\n"
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
                    status_text_started="Generating next response...",
                    status_text_completed="Response ready.",
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
    def loop_step_prompt_addendum(*, stage: str = "full", tool_transport_mode: str = "sentinel") -> str:
        _ = stage
        _ = tool_transport_mode
        return (
            "**Loop Step Mode (Mandatory):**\n"
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
