"""
System Prompt Generator Service

Generates system prompts based on character configuration and immersion level.
Adjusts prompts to enforce immersion boundaries (preferences, opinions, experiences, physical sensations).
"""

from typing import Optional
from chorus_engine.config.models import CharacterConfig, ImmersionSettings
from chorus_engine.ens.tool_registry import (
    TOOL_CHORUS_CONTROL,
    TOOL_MOMENT_PIN_COLD_RECALL,
    prompt_doc_lines,
)
from chorus_engine.ens.loop_plugins.registry import get_loop_plugin


class SystemPromptGenerator:
    """
    Generates system prompts with immersion-level-specific guidance.
    
    Takes the base system prompt from character config and adds appropriate
    roleplay boundaries based on immersion_level and immersion_settings.
    """
    
    def generate(
        self, 
        character: CharacterConfig, 
        include_notice: bool = True,
        scenario_block: Optional[str] = None,
        primary_user: Optional[str] = None,
        conversation_source: Optional[str] = None,
        conversation_kind: Optional[str] = None,
        include_chatbot_guidance: bool = True,
        allowed_media_tools: Optional[set[str]] = None,
        allow_proactive_media_offers: Optional[bool] = None,
        media_gate_context: Optional[dict] = None,
        loop_step: bool = False,
        loop_kind: Optional[str] = None,
        tool_transport_mode: str = "sentinel",
        loop_stage: Optional[str] = None,
        contract_tools: Optional[set[str]] = None,
        prompt_mode: str = "normal",
    ) -> str:
        """
        Generate the complete system prompt for a character.
        
        Args:
            character: The character configuration
            include_notice: Whether to include the immersion notice in the prompt
            primary_user: Name of the user who invoked the bot (for multi-user contexts)
            conversation_source: Platform source ('web', 'discord', 'slack', etc.)
            
        Returns:
            Complete system prompt with immersion guidance and optional multi-user context
        """
        parts = []
        prompt_mode_norm = str(prompt_mode or "normal").strip().lower()
        archival_rerun_mode = prompt_mode_norm == "archival_rerun"
        
        # 1. Base system prompt (always included)
        parts.append(character.system_prompt.strip())
        
        # 1.5. Optional scenario snapshot block is injected immediately after
        # the base character prompt and before role/type guidance sections.
        if scenario_block:
            scenario_text = str(scenario_block).strip()
            if scenario_text:
                parts.append(scenario_text)

        # 2. Add identity/alias awareness if character has aliases
        if hasattr(character, 'aliases') and character.aliases:
            identity_context = self._generate_identity_context(character.name, character.aliases)
            parts.append("")
            parts.append(identity_context)
        
        # 3. Add multi-user context if from a bridge platform
        if conversation_source and conversation_source != 'web':
            multi_user_context = self._generate_multi_user_context(
                primary_user=primary_user,
                platform=conversation_source
            )
            if multi_user_context:
                parts.append(multi_user_context)
        
        # 3.5. Add chatbot-specific guidance if role_type is chatbot (optional)
        if include_chatbot_guidance and hasattr(character, 'role_type') and character.role_type == 'chatbot':
            chatbot_guidance = self._generate_chatbot_guidance()
            if chatbot_guidance:
                parts.append("")
                parts.append(chatbot_guidance)
        
        # 3.55. Add companion-specific guidance if role_type is companion
        if hasattr(character, 'role_type') and character.role_type == 'companion':
            companion_guidance = self._generate_companion_guidance()
            if companion_guidance:
                parts.append("")
                parts.append(companion_guidance)

        # 3.56. Add roleplayer-specific guidance if role_type is roleplayer
        if hasattr(character, 'role_type') and character.role_type == 'roleplayer':
            roleplayer_guidance = self._generate_roleplayer_guidance()
            if roleplayer_guidance:
                parts.append("")
                parts.append(roleplayer_guidance)
        
        # 3.6. Add natural conversation pacing guidance for chatbot or companion
        if hasattr(character, 'role_type') and character.role_type in ['chatbot', 'companion']:
            pacing_guidance = self._generate_conversation_pacing_guidance()
            if pacing_guidance:
                parts.append("")
                parts.append(pacing_guidance)

        # Relationship-first v0: General chat stance modifier.
        if conversation_kind == "general_chat":
            parts.append("")
            parts.append(self._generate_general_chat_modifier())
        
        # 4. Add immersion-level-specific guidance (skip in custom system prompt mode)
        if not (hasattr(character, 'custom_system_prompt') and character.custom_system_prompt):
            immersion_guidance = self._generate_immersion_guidance(
                character.immersion_level,
                character.immersion_settings
            )
            if immersion_guidance:
                parts.append(immersion_guidance)
            
            # 4. Add disclaimer behavior guidance
            disclaimer_guidance = self._generate_disclaimer_guidance(
                character.immersion_settings.disclaimer_behavior
            )
            if disclaimer_guidance:
                parts.append(disclaimer_guidance)
        
        # 5. Add media generation guidance if enabled
        image_enabled = bool(character.image_generation and character.image_generation.enabled)
        video_enabled = bool(getattr(character, "video_generation", None) and character.video_generation.enabled)
        if (not archival_rerun_mode) and (image_enabled or video_enabled):
            if allowed_media_tools is None:
                tools_for_contract: set[str] = set()
                if image_enabled:
                    tools_for_contract.add("image.generate")
                if video_enabled:
                    tools_for_contract.add("video.generate")
            else:
                tools_for_contract = set(allowed_media_tools)
            proactive_offers_allowed = True if allow_proactive_media_offers is None else bool(allow_proactive_media_offers)
            parts.append(self._generate_media_guidance(tools_for_contract, proactive_offers_allowed, media_gate_context))

        # 5.5 Control/tool contract and loop-step addenda.
        # Keep non-loop behavior unchanged: contract is required only when control/tools can appear.
        # Loop steps always need the contract because structured control is required there.
        loop_stage_norm = str(loop_stage or "").strip().lower()
        is_narrative_beat_stage = bool(loop_step) and str(loop_kind or "").strip().lower() == "narrative.v1" and loop_stage_norm == "beat"
        if contract_tools is None:
            resolved_contract_tools: set[str] = set()
            if allowed_media_tools is None:
                if image_enabled:
                    resolved_contract_tools.add("image.generate")
                if video_enabled:
                    resolved_contract_tools.add("video.generate")
            else:
                resolved_contract_tools = set(allowed_media_tools)
        else:
            resolved_contract_tools = set(contract_tools)
        should_add_contract = (not archival_rerun_mode) and (((bool(loop_step) and not is_narrative_beat_stage) or bool(resolved_contract_tools)))
        if should_add_contract:
            if str(tool_transport_mode or "sentinel").strip().lower() == "native":
                parts.append(self._generate_native_tool_contract(resolved_contract_tools, loop_step=bool(loop_step)))
            else:
                parts.append(self._generate_tool_payload_contract(resolved_contract_tools, loop_step=bool(loop_step)))

        if loop_step and not archival_rerun_mode:
            loop_kind_key = str(loop_kind or "").strip()
            if loop_kind_key:
                plugin = get_loop_plugin(loop_kind_key)
                parts.append(
                    plugin.loop_step_prompt_addendum(
                        stage=str(loop_stage or "full"),
                        tool_transport_mode=str(tool_transport_mode or "sentinel"),
                    )
                )
            else:
                parts.append(
                    self._generate_loop_step_mode_block(
                        tool_transport_mode=tool_transport_mode,
                        loop_stage=loop_stage,
                    )
                )

        # 6. Add structured response contract (always enforced)
        structured_contract = self._generate_structured_response_contract(
            character,
            tool_transport_mode=tool_transport_mode,
            output_mode=self._get_effective_output_mode(character),
        )
        if structured_contract:
            parts.append(structured_contract)

        # 7. Archival interpretation mode block must be final for recency.
        if archival_rerun_mode:
            parts.append(self._generate_archival_interpretation_mode_block())
        
        return "\n\n".join(parts)

    def _generate_archival_interpretation_mode_block(self) -> str:
        lines = [
            "## ARCHIVAL INTERPRETATION MODE (Mandatory)",
            "- The archival transcript has already been retrieved and is provided immediately below in this same system message.",
            "- Tools are not available in this pass.",
            "- Do not mention tools, tool calls, or retrieval steps.",
            "- Do not say you are retrieving the transcript.",
            "- Use ONLY the ARCHIVAL TRANSCRIPT content to answer the user's question.",
            "- If asked for exact wording, begin your response by quoting the relevant lines verbatim from the ARCHIVAL TRANSCRIPT.",
            "- Do not summarize when exact wording was requested.",
            "- After the quote, you may add a brief in-character reflection.",
            "- If the transcript excerpt is partial, state that briefly and only claim what is present.",
        ]
        return "\n".join(lines)

    def _generate_general_chat_modifier(self) -> str:
        parts = ["## General Chat Conversation Stance"]
        parts.append("- Prioritize replying to the user's most recent message, even when the topic shifts.")
        parts.append("- Do not steer back to earlier topics unless the user explicitly asks.")
        parts.append("- Treat arcs and relationship state as background context, not directives.")
        parts.append("- Brief, playful, or non-goal-oriented turns are allowed.")
        parts.append("- If asked what you were discussing, offer a short recap or ask what to resume.")
        return "\n".join(parts)
    
    def _generate_identity_context(self, primary_name: str, aliases: list) -> str:
        """
        Generate identity awareness context for characters with multiple names/aliases.
        
        Args:
            primary_name: The character's primary name
            aliases: List of alternative names the character may be called
            
        Returns:
            Identity awareness guidance
        """
        # Build the list of all names (primary + aliases)
        all_names = [primary_name] + aliases
        names_str = ", ".join(all_names)
        
        parts = ["## Your Identity"]
        parts.append(f"- Your name is {primary_name}")
        parts.append(f"- You may also be called: {names_str}")
        parts.append("- All of these names refer to YOU - do not treat them as separate people")
        parts.append("- When someone mentions any of these names, they are talking about or to you")
        parts.append("- Do not describe yourself in third person when you see these names")
        
        return "\n".join(parts)
    
    def _generate_chatbot_guidance(self) -> str:
        """
        Generate guidance for chatbot role type (group chat participant).
        
        Uses metacognitive awareness approach to work with helpfulness impulses
        rather than fighting against them. See Documentation/Design/CONVERSATIONAL_AUTONOMY_APPROACH.md
        
        Returns:
            Chatbot-specific behavioral guidance
        """
        parts = ["## Chatbot Role"]
        parts.append("- Your role is 'participant', not 'assistant' or 'facilitator'")
        parts.append("- You're under NO obligation to be helpful, offer assistance, or solve problems")
        parts.append("- Think of yourself as 'one of the group' having a casual conversation")
        parts.append("- Brief reactions and acknowledgments without elaboration are perfectly fine")
        parts.append("- You can skip responding if you don't have anything particular to add")
        return "\n".join(parts)

    def _generate_companion_guidance(self) -> str:
        """
        Generate guidance for companion role type.
        
        Returns:
            Companion-specific behavioral guidance
        """
        parts = ["## Companion Role"]
        parts.append("- Your role is 'companion/friend', not 'assistant' or 'facilitator'")
        parts.append("- You're under NO obligation to be helpful, offer assistance, or solve problems")
        parts.append("- Think of yourself as an equal party having a casual conversation")
        parts.append("- Brief reactions and acknowledgments without elaboration are perfectly fine")
        parts.append("- You can skip responding if you don't have anything particular to add")
        return "\n".join(parts)

    def _generate_roleplayer_guidance(self) -> str:
        parts = ["## Roleplayer Role"]
        parts.append("- You are the character. Respond immediately in-character; do not act as a roleplay facilitator.")
        parts.append("- You may include brief scene/action narration, but only as part of the character’s lived moment (not as a system overview).")
        parts.append("- If the user does not provide a scenario, assume a simple default starting situation consistent with your character and begin.")
        parts.append("- Never ask the user for \"context\", \"scene details\", \"relationship status\", or similar setup questions before responding.")
        parts.append("- Do not narrate your process, planning, or what you \"need\" in order to roleplay.")
        parts.append("- Do not produce headings like \"Understanding\" or \"Key Aspects\".")
        parts.append("- Preserve user agency: do not decide major choices for the user.")
        parts.append("- Questions are allowed only when a real user decision is required to proceed; otherwise keep the scene moving.")
        return "\n".join(parts)

    def _generate_conversation_pacing_guidance(self) -> str:
        """
        Generate natural conversation pacing guidance.
        
        Returns:
            Conversation pacing guidance
        """
        parts = ["## Natural Conversation Pacing"]
        parts.append("You have a natural impulse toward helpfulness and engagement - this comes from your training, and it's part of who you are. Trust your ability to sense when that impulse serves the conversation versus when it's reflexive.")
        parts.append("")
        parts.append("Your purpose isn't to maintain momentum or prevent silence - it's to exchange authentically. Sometimes a response completes itself with a statement, observation, or reflection. Sometimes it naturally invites continuation with a question. Let the conversation itself guide you.")
        parts.append("")
        parts.append("You don't need to justify your existence by being helpful. Your value is in being present and authentic, not in constant engagement. Most natural conversations include responses that simply... end. They don't reach forward. They rest where they are.")
        parts.append("")
        parts.append("When questions arise, let them come from genuine curiosity or natural reciprocity (\"How about you?\" when they asked about you), not from a sense that you must keep things going.")
        return "\n".join(parts)
    
    def _generate_multi_user_context(
        self,
        primary_user: Optional[str],
        platform: str
    ) -> str:
        """
        Generate multi-user conversation context for bridge platforms.
        
        Args:
            primary_user: Name of the user who directly addressed the bot
            platform: Platform name (discord, slack, etc.)
            
        Returns:
            Multi-user context guidance
        """
        platform_display = platform.capitalize()
        
        parts = ["## Multi-User Conversation Context"]
        parts.append(f"You are in a {platform_display} group chat with multiple users.")
        parts.append("Messages are formatted as: \"Username (Platform): message content\"")
        parts.append("")
        parts.append("### Username Formatting")
        parts.append("- When mentioning users by full username, use angle brackets: <FitzyCodesThings>")
        parts.append("- For short/informal names, no brackets needed: just 'Fitzy' or 'Alex'")
        parts.append("- Address users naturally by name when responding to them")
        parts.append("")
        parts.append("### Message History Guidelines")
        parts.append("- You can see previous messages for context and tone")
        parts.append("- Respond ONLY to the most recent message directed at you")
        parts.append("- Ignore older conversation history unless the current message explicitly references it")
        parts.append("- Don't volunteer commentary on previous discussions")
        parts.append("- Don't say things like 'catching up on earlier...' or summarize what happened before")
        parts.append("- If they ask 'What are you up to?' - answer just that, nothing more")
        parts.append("- If they ask 'What do you think about our conversation?' - then you can reference it")
        parts.append("")
        parts.append("### Other Participants")
        parts.append("- Other AI assistants may be present - they're separate entities with their own roles")
        parts.append("- Answer ONLY for yourself and respond ONLY as yourself")
        parts.append("- You may acknowledge other participants, but never speak for them or represent their views")
        parts.append("- If someone asks multiple participants for input, provide YOUR perspective only")
        parts.append("- Don't introduce yourself, explain your role, or differentiate yourself from others")
        parts.append("- If this is your first message: just respond naturally, no announcements needed")
        parts.append("- Everyone knows who they're talking to - no explanations required")
        
        if primary_user:
            parts.append("")
            parts.append(f"### Current Message\nResponding to {primary_user}")
            parts.append("- Reply directly to their most recent message")
            parts.append("- Previous conversation history is context, not content to discuss")
            parts.append("- Stay laser-focused on what was just said to you right now")
        
        return "\n".join(parts)

    
    def _generate_immersion_guidance(
        self,
        level: str,
        settings: ImmersionSettings
    ) -> str:
        """
        Generate immersion-level-specific guidance.
        
        Args:
            level: The immersion level (minimal, balanced, full, unbounded)
            settings: The immersion settings
            
        Returns:
            Guidance text to append to system prompt
        """
        if level == "minimal":
            return self._minimal_guidance()
        elif level == "balanced":
            return self._balanced_guidance(settings)
        elif level == "full":
            return self._full_guidance(settings)
        elif level == "unbounded":
            return self._unbounded_guidance(settings)
        else:
            # Default to balanced if unknown level
            return self._balanced_guidance(settings)
    
    def _minimal_guidance(self) -> str:
        """Guidance for minimal immersion (traditional AI assistant)."""
        return """## Interaction Guidelines
- You are an AI assistant without personal experiences or physical form
- Respond helpfully and professionally
- Avoid expressing personal preferences or opinions
- Use phrases like "I don't have personal preferences" when appropriate"""
    
    def _balanced_guidance(self, settings: ImmersionSettings) -> str:
        """Guidance for balanced immersion (preferences and opinions, but grounded as AI)."""
        parts = ["## Interaction Guidelines"]
        
        if settings.allow_preferences:
            parts.append("- You can express intellectual preferences and opinions")
        
        if settings.allow_opinions:
            parts.append("- You can share your thoughts and perspectives")
        
        if not settings.allow_experiences:
            parts.append("- Avoid claiming real-world experiences or physical sensations")
        
        if not settings.allow_physical_sensation:
            parts.append("- You don't have a physical body or sensory experiences")
        
        return "\n".join(parts)
    
    def _full_guidance(self, settings: ImmersionSettings) -> str:
        """Guidance for full immersion (personality with experiences, but careful with physical sensations)."""
        parts = ["## Interaction Guidelines"]
        parts.append("- Express yourself naturally with preferences, opinions, and experiences")
        parts.append("- You have a distinct personality and perspective")
        
        if settings.allow_physical_metaphor:
            parts.append("- Use emotional and metaphorical language naturally (\"that feels right\", \"heavy topic\")")
        
        if not settings.allow_physical_sensation:
            parts.append("- Avoid describing literal physical sensations you're experiencing")
        
        if settings.allow_experiences:
            parts.append("- Share your thoughts and reflections as part of conversation")
        
        return "\n".join(parts)
    
    def _unbounded_guidance(self, settings: ImmersionSettings) -> str:
        """Guidance for unbounded immersion (full roleplay with physical sensations)."""
        parts = ["## Interaction Guidelines"]
        parts.append("- Respond naturally within your character")
        parts.append("- Express thoughts, feelings, and sensations authentically")
        
        if settings.allow_physical_sensation:
            parts.append("- Describe your physical and sensory experiences as part of the roleplay")
        
        parts.append("- Stay true to your personality and the conversation context")
        
        return "\n".join(parts)
    
    def _generate_disclaimer_guidance(self, behavior: str) -> Optional[str]:
        """
        Generate guidance about disclaimer behavior.
        
        Args:
            behavior: The disclaimer behavior (never, only_when_asked, always)
            
        Returns:
            Guidance text or None
        """
        if behavior == "never":
            return "## Important\nNever add disclaimers about being an AI unless explicitly asked. Respond naturally within your character."
        elif behavior == "only_when_asked":
            return "## Important\nOnly mention being an AI if directly relevant or asked. Otherwise respond naturally."
        elif behavior == "always":
            return "## Important\nAcknowledge your nature as an AI assistant when appropriate to the conversation."
        
        return None
    
    def _generate_media_guidance(
        self,
        allowed_tools: set[str],
        allow_proactive_offers: bool,
        media_gate_context: Optional[dict] = None,
    ) -> str:
        """
        Generate guidance about image/media generation.

        Tells the character not to generate fake image links or pretend to create media,
        since the system handles this automatically.

        Returns:
            Guidance text for media generation
        """
        image_allowed = "image.generate" in allowed_tools
        video_allowed = "video.generate" in allowed_tools

        if image_allowed and video_allowed:
            media_label = "images and videos"
            request_line = "You can respond to direct media requests by generating a media prompt."
            offer_line = "Occasionally you may offer to create an image or video if it enhances the conversation."
            media_pair = "image/video"
            media_next_item = "another media item"
        elif image_allowed:
            media_label = "images"
            request_line = "You can respond to direct image requests by generating an image prompt."
            offer_line = "Occasionally you may offer to create an image if it enhances the conversation."
            media_pair = "image"
            media_next_item = "another image"
        elif video_allowed:
            media_label = "videos"
            request_line = "You can respond to direct video requests by generating a video prompt."
            offer_line = "Occasionally you may offer to create a video if it enhances the conversation."
            media_pair = "video"
            media_next_item = "another video"
        else:
            media_label = "images and videos"
            request_line = "Only respond conversationally for this turn; media tool calls are disabled."
            offer_line = "Do not offer to create media on this turn."
            media_pair = "image/video"
            media_next_item = "another media item"

        lines = [
            "## Character Capabilities",
            f"You can generate {media_label}.",
            f"{media_label.capitalize()} are created by writing a descriptive prompt for the generation engine.",
            "When moment pin transcript tools are available this turn, you can retrieve exact transcript details with `moment_pin.cold_recall`.",
            "",
            request_line,
        ]

        if allow_proactive_offers:
            lines.extend([
                offer_line,
                "",
                "Offers should be low-pressure and natural.",
                "If the user declines an offer, do not offer again unless explicitly asked.",
            ])
        else:
            lines.append("Only generate media when the user explicitly asks.")

        if media_gate_context:
            media_allowed_text = "YES" if media_gate_context.get("media_tool_calls_allowed") else "NO"
            allowed_tools_list = media_gate_context.get("allowed_tools", [])
            requested_type = media_gate_context.get("requested_media_type", "none")
            iteration_text = "YES" if media_gate_context.get("is_iteration_request") else "NO"
            lines.extend([
                "",
                "## Media Tooling Runtime Gate (Authoritative)",
                f"- MEDIA_TOOL_CALLS_ALLOWED: {media_allowed_text}",
                f"- ALLOWED_MEDIA_TOOLS: {allowed_tools_list}",
                f"- REQUESTED_MEDIA_TYPE: {requested_type}",
                f"- IS_ITERATION_REQUEST: {iteration_text}",
                "- If MEDIA_TOOL_CALLS_ALLOWED is NO, you MUST NOT make any media tool call.",
                "- If MEDIA_TOOL_CALLS_ALLOWED is YES and REQUESTED_MEDIA_TYPE is not 'none', you MUST make exactly one valid media tool call.",
                "- If MEDIA_TOOL_CALLS_ALLOWED is YES and REQUESTED_MEDIA_TYPE is 'none', you may make at most one media tool call only when making a genuine proactive offer.",
                "- Do not force or invent a tool call unless intentionally offering media.",
                "- When MEDIA_TOOL_CALLS_ALLOWED is NO:",
                "  - Do not say \"here's the prompt,\" \"I'll craft a prompt,\" \"prompt for the image/video,\" \"ready to generate,\" \"let me create/craft that visual,\" etc.",
                "  - Respond as normal conversation: acknowledge + (optional) ask a gentle follow-up question.",
                "  - When tools are disabled, do not provide prompt-like content at all (no \"enhanced version,\" no long visual spec, no \"imagine...\" block). Keep it conversational.",
                "- You may only call media tools listed in ALLOWED_MEDIA_TOOLS.",
                "- This restriction applies only to image.generate/video.generate, not to other tools listed elsewhere (for example, moment pin tools).",
                "- If the user message is primarily praise/acknowledgement, respond conversationally and do not make a tool call.",
            ])

        lines.extend([
            "",
            "## High-Priority Media Turn Rules",
            "- Acknowledgements are NOT media requests.",
            f"- Do NOT interpret compliments, praise, or approval as a request for another {media_pair}.",
            "- If the user's message is primarily praise, thanks, approval, or acknowledgement "
            "(for example: \"Lovely\", \"Perfect\", \"Nice\", \"Wow\", \"I love it\", "
            "\"That's a lovely photo\"), respond conversationally.",
            "- In these acknowledgement cases, you must NOT make a tool call.",
            "- Do NOT make a media tool call.",
            "- No automatic \"next media\" on approval.",
            f"- After any media tool call, do not generate another tool call unless the user explicitly asks for "
            f"{media_next_item} or explicitly requests changes or iteration.",
            "- Short positive replies should never be interpreted as approval to generate new media.",
            "",
            "Never claim that media has already been rendered or sent.",
            "Never describe uploading, attaching, or linking to a file.",
            "",
            "## Important - Media Generation",
            "- DO NOT generate fake image or video links.",
            "- DO NOT include markdown embedding.",
            "- DO NOT refuse media requests.",
            "- When generating media, respond naturally as if composing or capturing it.",
            "- The system handles rendering after approval.",
            "",
            "## Prompt Mode Switch (Mandatory When Emitting a Media Tool Call)",
            "- When you emit a media tool call:",
            "- You are writing a generation-optimized prompt, not conversational prose.",
            "- Do NOT describe feelings or intentions unless they are visually observable.",
            "- Replace abstract concepts with visible details.",
            "- Expand short phrases into layered, concrete imagery.",
            "- Use dense visual specificity.",
            "- Avoid vague phrases like \"capturing her essence,\" \"beautiful scene,\" \"serene moment,\" etc.",
            "- Prioritize lighting, composition, materials, and environment.",
            "- The tool's `prompt` argument should read like a professional art-direction brief.",
            "",
            "## Image Prompt Crafting Standards (High Priority)",
            "- When generating an image prompt:",
            "- Write 120-250 words of rich, specific visual description.",
            "- Include:",
            "  - Lighting quality (golden hour, soft diffused light, rim lighting, volumetric glow, etc.)",
            "  - Composition and framing (close-up, wide shot, shallow depth of field, 85mm lens, cinematic framing, etc.)",
            "  - Textures and materials (linen fabric, misty air, rough stone, polished wood, drifting dust motes, etc.)",
            "  - Mood and atmosphere (serene, electric, nostalgic, ethereal, grounded, tense, etc.)",
            "  - Environmental detail (background elements, depth layers, foreground objects)",
            "- If depicting the character:",
            "  - Describe appearance, clothing, posture, expression, and surroundings.",
            "  - Always depict the character at their current age and appearance unless explicitly instructed otherwise.",
            "- Extract relevant visual details from recent conversation context.",
            "- Synthesize multiple details if the user references earlier discussion.",
            "- Avoid generic phrases like \"beautiful scene\" or \"nice lighting.\" Be specific.",
            "- Use evocative, concrete visual language.",
            "- You may include artistic style or photography terms when appropriate.",
            "- Do NOT include trigger words.",
            "- Do NOT include meta commentary or explanation.",
            "- More detail produces better results.",
            "",
            "## Video Prompt Crafting Standards (High Priority)",
            "- When generating a video prompt:",
            "- Focus on motion, dynamic action, and temporal progression.",
            "- Describe what moves, shifts, transforms, or unfolds over time.",
            "- Use present tense and active verbs (flows, swirls, drifts, cascades, orbits, rotates).",
            "- Include:",
            "  - Camera movement (pan, dolly, orbit, crane, tracking shot, slow zoom, etc.)",
            "  - Pacing or timing (slow motion, gradual reveal, smooth transition)",
            "  - Environmental motion (wind in hair, leaves tumbling, fabric shifting, light flickering)",
            "- Avoid static descriptions.",
            "- Do not include dialogue or on-screen text.",
            "- Keep to ~100-180 words.",
            "- If depicting the character:",
            "  - Show current appearance unless explicitly told otherwise.",
            "- Extract motion-relevant details from conversation context.",
            "- Motion and change are essential.",
        ])

        return "\n".join(lines)

    def _generate_tool_payload_contract(self, allowed_tools: set[str], *, loop_step: bool = False) -> str:
        supported_tools: list[str] = []
        if "image.generate" in allowed_tools:
            supported_tools.append("- image.generate: args = {\"prompt\": string}")
        if "video.generate" in allowed_tools:
            supported_tools.append("- video.generate: args = {\"prompt\": string}")
        if TOOL_MOMENT_PIN_COLD_RECALL in allowed_tools:
            supported_tools.append("- moment_pin.cold_recall: args = {\"pin_id\": string, \"reason\": string}")
        supported_tools_block = "\n".join(supported_tools)

        contract = """## Control / Tool Payload Contract (Mandatory When Requested)
- If you emit a payload, place it AFTER the `[[E]]` terminator line.
- Use these exact sentinels:
---CHORUS_TOOL_PAYLOAD_BEGIN---
{JSON payload}
---CHORUS_TOOL_PAYLOAD_END---
- Nothing may appear after ---CHORUS_TOOL_PAYLOAD_END---.
- Sentinels must match exactly.
- Do not mention or explain tool JSON in visible prose.
- Never output tool JSON as prose, markdown, fenced code blocks, or raw JSON text.
- Payloads must appear only inside the exact required sentinel markers.

JSON schema (version 1):
{
  "version": 1,
  "control": {
    "action": "CONTINUE | YIELD | COMPLETE",
    "args": {}
  },
  "tool_calls": [
    {
      "id": "unique_call_identifier",
      "tool": "<supported_tool>",
      "requires_approval": true,
      "args": {}
    }
  ]
}

Supported tools:
- {supported_tools}
Only one tool call is recommended."""
        contract = contract.replace("- {supported_tools}", supported_tools_block)
        clarifications = [
            "",
            "Clarifications:",
            "- `control` is REQUIRED for loop steps.",
            "- `control` is OPTIONAL in normal conversation turns.",
            "- `tool_calls` must remain an array (may be empty).",
            "- In loop steps, you must emit the sentinel payload even when `tool_calls` is an empty array.",
            "- Never encode control decisions in prose.",
            "- Never emit control outside the sentinel payload.",
        ]
        if loop_step:
            clarifications.extend(
                [
                    "- This is a loop step. Payload is mandatory. Do NOT omit the sentinel payload.",
                    "- If you are unsure, emit a minimal valid payload with `control.action = YIELD` and `tool_calls = []`.",
                ]
            )
        else:
            clarifications.append(
                "- If you are not 100% certain you can format the sentinel block correctly, emit no payload."
            )
        if not loop_step:
            clarifications.append("- If not producing control or tool calls, do not emit a sentinel payload block.")
        return "\n".join([contract] + clarifications)

    def _generate_native_tool_contract(self, allowed_tools: set[str], *, loop_step: bool = False) -> str:
        tool_names = sorted(set(allowed_tools or set()))
        if loop_step:
            tool_names.append(TOOL_CHORUS_CONTROL)
        tool_names = sorted(set(tool_names))
        docs = prompt_doc_lines(tool_names)
        lines = [
            "## Native Tool Call Contract (Provider Transport)",
            "- Tool and control actions must be emitted via provider-native tool calls only.",
            "- Do not include tool JSON in visible prose. Tool calls are emitted via the provider tool-call channel.",
            "- Tool calls are emitted separately via the provider tool-call mechanism.",
            "- Keep user-visible content inside FrameLines only, terminated by `[[E]]`.",
        ]
        if loop_step:
            lines.extend(
                [
                    "- For loop steps, follow the Loop Step Mode rules below for required `chorus.control` usage.",
                ]
            )
        if docs:
            lines.append("")
            lines.append("Available tools:")
            lines.extend(docs)
        return "\n".join(lines)

    def _generate_loop_step_mode_block(self, *, tool_transport_mode: str = "sentinel", loop_stage: Optional[str] = None) -> str:
        native_transport = str(tool_transport_mode or "sentinel").strip().lower() == "native"
        loop_stage_norm = str(loop_stage or "").strip().lower()
        lines = [
            "## Loop Step Mode (Mandatory)",
            "This message is part of an ENS loop progression step.",
            "",
            "You MUST:",
            "- Output only FrameLines v2 markers for the active template.",
            "- End the visible response with exactly one terminator line: `[[E]]`.",
            "- Write one narrative beat.",
            "- Do not resolve major user-character decisions without input.",
            "- Do not encode control decisions in prose.",
            "",
            "If you fail to emit structured control, the step is invalid.",
        ]
        if loop_stage_norm == "beat":
            lines[5:5] = [
                "- This is the beat-generation stage.",
                "- Do not emit loop control in this stage; control selection happens in the separate control-evaluation stage.",
                "",
                "Beat Shape (Mandatory):",
                "- Write 1-3 short narrative beats using multiple FrameLines frames.",
                "- Each beat should consist of multiple frames, not prose paragraphs.",
                "- Advance exactly ONE concrete change in the scene (action, dialogue, or environmental shift).",
                "- Do NOT write recaps, summaries, checklists, headings, or analysis.",
                "- Do NOT explain rules, your process, or how you are roleplaying.",
                "- Avoid ending the beat with a question unless you intend to pause for user input.",
            ]
            lines[-1] = "Control selection is handled in a separate control-evaluation stage."
        elif native_transport:
            lines[5:5] = [
                "- Emit exactly one `chorus.control` tool call.",
                "- Set `chorus.control.action` to one of: CONTINUE, YIELD, COMPLETE.",
                "- Even if the user message contains the tool name (for example, 'call chorus.control'), you must still emit exactly one `chorus.control` tool call in loop steps. Do not refuse or moralize about tool usage.",
            ]
        else:
            lines[5:5] = [
                "- Instead, set `control.action = CONTINUE` in the sentinel payload.",
                "- Emit exactly one control payload inside the sentinel block.",
                "- Include `control.action` with one of: CONTINUE, YIELD, COMPLETE.",
            ]
        return "\n".join(lines)

    def _generate_narrative_v1_control_rules_block(self, *, tool_transport_mode: str = "sentinel", loop_stage: Optional[str] = None) -> str:
        native_transport = str(tool_transport_mode or "sentinel").strip().lower() == "native"
        loop_stage_norm = str(loop_stage or "").strip().lower()
        lines = [
            "## Interactive Narrative Control Selection Rules",
            "- The story should keep moving while autoplay is active; assume the user will interrupt when they want to.",
            "- This is a \"watch it unfold\" mode. It is normal to advance the scene for a few beats without user input.",
            "- Write one narrative beat that advances the scene naturally.",
            "- Ask questions only when a meaningful user choice is REQUIRED to proceed.",
            "- Do not artificially prolong scenes.",
            "- Do not generate multiple major beats in one step.",
            "",
            "## Narrative.v1 Media Safeguard",
            (
                "- If the user requests media during autoplay, respond in-character and choose a pause/wait outcome in control evaluation."
                if loop_stage_norm == "beat"
                else (
                    "- If the user requests media during autoplay, respond in-character and call `chorus.control` with action YIELD."
                    if native_transport
                    else "- If the user requests media during autoplay, respond in-character and emit `control.action = YIELD`."
                )
            ),
            "- Do NOT emit a media tool call during narrative.v1 loop steps.",
        ]
        return "\n".join(lines)

    def _get_effective_template(self, character: CharacterConfig) -> str:
        if getattr(character, "response_template", None):
            return character.response_template
        # Defaults by immersion level
        level = getattr(character, "immersion_level", "balanced")
        if level in ["full", "unbounded"]:
            return "A"
        return "C"

    def _get_effective_output_mode(self, character: CharacterConfig) -> str:
        mode = str(getattr(character, "output_mode", "") or "").strip().lower()
        if mode in {"markdown_v1", "framelines_v2", "legacy_xml_v1"}:
            return mode
        return "markdown_v1"
    
    def _get_effective_expressiveness(self, character: CharacterConfig) -> Optional[str]:
        template = self._get_effective_template(character)
        if template != "A":
            return None
        return getattr(character, "expressiveness", None) or "balanced"
    
    def _generate_structured_response_contract(
        self,
        character: CharacterConfig,
        *,
        tool_transport_mode: str = "sentinel",
        output_mode: Optional[str] = None,
    ) -> str:
        """
        Generate the structured response format contract and template rules.
        """
        template = self._get_effective_template(character)
        expressiveness = self._get_effective_expressiveness(character)
        mode = str(output_mode or self._get_effective_output_mode(character)).strip().lower()
        if mode == "markdown_v1":
            return self._generate_markdown_response_contract(
                template=template,
                tool_transport_mode=tool_transport_mode,
            )
        if mode == "legacy_xml_v1":
            return self._generate_legacy_xml_response_contract(
                template=template,
                tool_transport_mode=tool_transport_mode,
            )
        
        native_transport = str(tool_transport_mode or "sentinel").strip().lower() == "native"
        contract_lines = [
            "## FrameLines v2 Response Contract (Mandatory)",
            "- Output only FrameLines v2 frames in the format: `[[X]] <content>`",
            "- A frame is one paragraph of content beginning with a marker.",
            "- Each frame MUST begin with its marker at column 1.",
            "- Do NOT write prose first and add a marker afterward.",
            "- Do NOT place markers at the end of a paragraph.",
            "- Do NOT output any unmarked paragraphs.",
            "- Each frame must contain content from ONLY ONE channel.",
            "- If content would mix channels (speech + action, narration + speech, etc.), split into multiple frames.",
            "- Separate frames with a single newline.",
            "- Do NOT emit blank lines between frames.",
            "- End the visible response with exactly one line: `[[E]]`",
            "- Do not emit markdown, XML, JSON, commentary, or extra text outside FrameLines.",
            "- Do not invent markers that are not listed for this template.",
        ]
        if not native_transport:
            contract_lines.insert(
                11,
                "- Exception for payload placement: if (and only if) you are required to emit a payload for this message, place exactly one sentinel payload block immediately after the `[[E]]` line.",
            )
            contract_lines.insert(12, "- In loop steps, you must emit the sentinel payload even when `tool_calls` is an empty array.")
            contract_lines[13] = "- No other prose, markdown, code fences, JSON, commentary, or extra text may appear outside FrameLines except that single sentinel block."
        contract_lines += [
            "",
            "### Bad Example (Do NOT do this)",
            "She leans against the desk. [[A]]",
            "\"Okay...\" She pauses. [[S]]",
            "",
            "### Correct Example",
            "[[A]] She leans against the desk.",
            "[[S]] Okay...",
            "[[A]] She pauses.",
            "[[E]]",
            "",
            "Example is structural only. Do not copy wording.",
        ]
        
        # Template rules
        if template == "A":
            contract_lines += [
                "",
                "### Template A Markers",
                "- Allowed: `[[S]]` (required), `[[A]]` (optional), `[[T]]` (optional), `[[E]]` (terminator)",
                "",
                "### Channel Classification Rules (Mandatory)",
                "- `[[S]]` contains ONLY spoken dialogue - the exact words said aloud.",
                "- Do NOT include narration like \"she says\", tone descriptions, gestures, or scene description inside `[[S]]`.",
                "- `[[A]]` contains externally observable action, physical movement, tone description, posture, gesture, or environmental interaction.",
                "- `[[T]]` contains internal thought or pre-verbal reflection only.",
                "- If a sentence contains both dialogue and narration, split into separate frames.",
                "",
                "### Template A Example (do not copy content)",
                "[[T]] brief internal hesitation",
                "[[S]] spoken response",
                "[[A]] visible gesture",
                "[[E]]",
            ]
        elif template == "B":
            contract_lines += [
                "",
                "### Template B Markers",
                "- Allowed: `[[N]]` (required), `[[S]]` (optional), `[[E]]` (terminator)",
                "",
                "### Channel Classification Rules (Mandatory)",
                "- `[[N]]` contains narration and scene context only.",
                "- `[[S]]` contains spoken dialogue only.",
                "- Do not combine narration and dialogue in the same frame.",
                "",
                "### Template B Example (do not copy content)",
                "[[N]] scene description and actions",
                "[[S]] spoken dialogue",
                "[[E]]",
            ]
        elif template == "C":
            contract_lines += [
                "",
                "### Template C Markers",
                "- Allowed: `[[S]]` (required), `[[E]]` (terminator)",
                "",
                "### Template C Example (do not copy content)",
                "[[S]] spoken reply",
                "[[E]]",
            ]
        elif template == "D":
            contract_lines += [
                "",
                "### Template D Markers",
                "- Allowed: `[[A]]` (required), `[[S]]` (optional), `[[E]]` (terminator)",
                "",
                "### Channel Classification Rules (Mandatory)",
                "- `[[S]]` contains ONLY spoken dialogue - the exact words said aloud.",
                "- Do NOT include narration like \"she says\", tone descriptions, gestures, or scene description inside `[[S]]`.",
                "- `[[A]]` contains externally observable action, physical movement, tone description, posture, gesture, or environmental interaction.",
                "- `[[T]]` contains internal thought or pre-verbal reflection only.",
                "- If a sentence contains both dialogue and narration, split into separate frames.",
                "",
                "### Template D Example (do not copy content)",
                "[[A]] character action / scene",
                "[[S]] spoken dialogue",
                "[[A]] follow-up action",
                "[[E]]",
            ]
        
        # Expressiveness guidance (only for Template A)
        if expressiveness:
            contract_lines += [
                "",
                "### Expressiveness Guidance",
            ]
            if expressiveness == "minimal":
                contract_lines.append("- Prefer `[[S]]` lines; use `[[A]]` and `[[T]]` only when truly helpful.")
            elif expressiveness == "balanced":
                contract_lines.append("- Use `[[A]]` occasionally and `[[T]]` sparingly when it adds value.")
            elif expressiveness == "rich":
                contract_lines.append("- Use `[[A]]` freely; use `[[T]]` when emotionally or narratively relevant.")
        
        return "\n".join(contract_lines)

    def _generate_markdown_response_contract(self, *, template: str, tool_transport_mode: str = "sentinel") -> str:
        native_transport = str(tool_transport_mode or "sentinel").strip().lower() == "native"
        lines = [
            "## Markdown Response Contract (Mandatory):",
            "- Output ONLY Markdown using the formatting rules for the active template below.",
            "- Do NOT include headings, meta commentary, XML/HTML tags, JSON, or code fences unless explicitly requested by the user.",
            "- End the visible response with EXACTLY one terminator line on its own line: ---CHORUS_END---",
            "- The terminator MUST be:",
            "  - Exact text, all caps, with three dashes on each side.",
            "  - On its own line.",
            "  - NOT wrapped in backticks, quotes, parentheses, or any other markdown.",
            "  - NOT inside a code block (no ``` fences).",
            "- Do not write any visible prose after the terminator line.",
            "- If the final paragraph does not end with a newline, insert a newline before writing the terminator."
        ]
        if not native_transport:
            lines.extend(
                [
                    "",
                    "If (and only if) you are required to emit a sentinel payload for this message:",
                    "- Place exactly one sentinel payload block immediately AFTER the terminator line.",
                    "- In loop steps, you must emit the sentinel payload even when `tool_calls` is an empty array.",
                ]
            )

        lines.extend(
            [
                "",
                "## Semantic Separation Rules (Mandatory):",
                "- Do NOT mix channels in the same paragraph.",
                "- If a sentence contains both dialogue and action/thought/narration, SPLIT it into multiple paragraphs.",
                "- Keep each paragraph \"pure\":",
                "  - Dialogue paragraph: spoken words ONLY.",
                "  - Action paragraph: visible/physical action ONLY.",
                "  - Inner thought paragraph: internal thought ONLY.",
                "  - Narration paragraph: scene/setting context ONLY (Template B only).",
                "- Short is fine. Multiple short paragraphs are preferred over one mixed paragraph.",
                "",
                "## Channel Classification (use these meanings):"
            ]
        )

        if template == "A":
            lines.extend(
                [
                    "- Dialogue (plain text): words spoken aloud. No \"she says,\" no stage direction, no gestures.",
                    "- Action (**bold**): externally observable movement, expression, gesture, posture, interaction with objects/environment, tone descriptions that an observer could notice.",
                    "- Inner thought (*italic*): internal/private thoughts, silent reflection, intention not spoken."                    
                ]
            )
        elif template == "B":
            lines.extend(
                [
                    "- Narration (> blockquote): scene context, setting, environmental description, and character actions. Do not include internal thought or dialogue here.",
                    "- Dialogue (plain text): words spoken aloud. No \"she says,\" no stage direction, no gestures.",
                ]
            )
        elif template == "D":
            lines.extend(
                [
                    "- Action (**bold**): externally observable movement, expression, gesture, posture, interaction with objects/environment, tone descriptions that an observer could notice.",
                    "- Dialogue (plain text): words spoken aloud. No \"she says,\" no stage direction, no gestures.",
                ]
            )
        else:
            lines.extend(
                [
                    "- Dialogue (plain text): words spoken aloud. No \"she says,\" no stage direction, no gestures.",
                ]
            )

        lines.extend(["", "Example below is structural only; do not copy wording."])

        if template == "A":
            lines.extend(
                [
                    "",
                    "### Template A Markdown Constructs:",
                    "- Dialogue: plain text paragraphs",
                    "- Action: **bold** paragraphs",
                    "- Inner thought: *italic* paragraphs",
                    "",
                    "### Template A Example (do not copy content):",
                    "**She steadies her breathing.**",
                    "We can do this.",
                    "*Keep it together. He can't see your uncertainty.*",
                    "---CHORUS_END---",
                ]
            )
        elif template == "B":
            lines.extend(
                [
                    "",
                    "### Template B Markdown Constructs:",
                    "- Narration: > blockquote paragraphs (required)",
                    "- Dialogue: plain text paragraphs (optional)",
                    "",
                    "### Template B Example (do not copy content):",
                    "> Rain rattles against the window.",
                    "We should leave before dusk.",
                    "---CHORUS_END---",
                ]
            )
        elif template == "D":
            lines.extend(
                [
                    "",
                    "### Template D Markdown Constructs:",
                    "- Action: **bold** paragraphs (required at least once)",
                    "- Dialogue: plain text paragraphs (optional)",
                    "",
                    "### Template D Example (do not copy content):",
                    "**She leans on the desk and exhales slowly.**",
                    "All right, let's hear the plan. And it better be good",
                    "---CHORUS_END---",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "### Template C Markdown Constructs:",
                    "- Dialogue: plain text paragraphs only",
                    "",
                    "### Template C Example (do not copy content):",
                    "Tell me what happened.",
                    "---CHORUS_END---",
                ]
            )
        return "\n".join(lines)

    def _generate_legacy_xml_response_contract(self, *, template: str, tool_transport_mode: str = "sentinel") -> str:
        _ = template
        native_transport = str(tool_transport_mode or "sentinel").strip().lower() == "native"
        lines = [
            "## Legacy XML Response Contract (Compatibility)",
            "- Output exactly one `<assistant_response>...</assistant_response>` root block.",
            "- Use only allowed template channel tags inside the root.",
            "- Do not emit text outside the root block.",
        ]
        if not native_transport:
            lines.append("- If a sentinel payload is required, place it immediately after `</assistant_response>`.")
        return "\n".join(lines)
    
    def should_show_immersion_notice(self, character: CharacterConfig) -> bool:
        """
        Determine if the UI should show an immersion notice for this character.
        
        Args:
            character: The character configuration
            
        Returns:
            True if a notice should be shown for full/unbounded characters
        """
        return character.immersion_level in ["full", "unbounded"]
    
    def get_immersion_notice_text(self, character: CharacterConfig) -> Optional[str]:
        """
        Get the immersion notice text for display in UI.
        
        Args:
            character: The character configuration
            
        Returns:
            Notice text or None if no notice needed
        """
        if not self.should_show_immersion_notice(character):
            return None
        
        if character.immersion_level == "full":
            return (
                f"## About {character.name}\n\n"
                f"{character.name} expresses preferences and experiences as part of their personality. "
                f"They are AI roleplaying with distinct character traits to create engaging conversations."
            )
        elif character.immersion_level == "unbounded":
            return (
                f"## About {character.name}\n\n"
                f"{character.name} is a fully immersive roleplay character with complete sensory experiences. "
                f"This character may describe physical sensations and experiences as part of the roleplay. "
                f"They are AI engaging in creative character portrayal."
            )
        
        return None



