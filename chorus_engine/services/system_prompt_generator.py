"""
System Prompt Generator Service

Generates system prompts based on character configuration and immersion level.
Adjusts prompts to enforce immersion boundaries (preferences, opinions, experiences, physical sensations).
"""

from typing import Optional
from chorus_engine.config.models import CharacterConfig, ImmersionSettings


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
        primary_user: Optional[str] = None,
        conversation_source: Optional[str] = None,
        include_chatbot_guidance: bool = True,
        allowed_media_tools: Optional[set[str]] = None,
        allow_proactive_media_offers: Optional[bool] = None,
        media_gate_context: Optional[dict] = None,
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
        
        # 1. Base system prompt (always included)
        parts.append(character.system_prompt.strip())
        
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
        
        # 3.6. Add natural conversation pacing guidance for chatbot or companion
        if hasattr(character, 'role_type') and character.role_type in ['chatbot', 'companion']:
            pacing_guidance = self._generate_conversation_pacing_guidance()
            if pacing_guidance:
                parts.append("")
                parts.append(pacing_guidance)
        
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
        if image_enabled or video_enabled:
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
            if tools_for_contract:
                parts.append(self._generate_tool_payload_contract(tools_for_contract))
        
        # 6. Add structured response contract (always enforced)
        structured_contract = self._generate_structured_response_contract(character)
        if structured_contract:
            parts.append(structured_contract)
        
        return "\n\n".join(parts)
    
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
        
        parts = ["**Your Identity:**"]
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
        parts = ["**Chatbot Role:**"]
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
        parts = ["**Companion Role:**"]
        parts.append("- Your role is 'companion/friend', not 'assistant' or 'facilitator'")
        parts.append("- You're under NO obligation to be helpful, offer assistance, or solve problems")
        parts.append("- Think of yourself as an equal party having a casual conversation")
        parts.append("- Brief reactions and acknowledgments without elaboration are perfectly fine")
        parts.append("- You can skip responding if you don't have anything particular to add")
        return "\n".join(parts)

    def _generate_conversation_pacing_guidance(self) -> str:
        """
        Generate natural conversation pacing guidance.
        
        Returns:
            Conversation pacing guidance
        """
        parts = ["**Natural Conversation Pacing:**"]
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
        
        parts = ["**Multi-User Conversation Context:**"]
        parts.append(f"You are in a {platform_display} group chat with multiple users.")
        parts.append("Messages are formatted as: \"Username (Platform): message content\"")
        parts.append("")
        parts.append("**Username Formatting:**")
        parts.append("- When mentioning users by full username, use angle brackets: <FitzyCodesThings>")
        parts.append("- For short/informal names, no brackets needed: just 'Fitzy' or 'Alex'")
        parts.append("- Address users naturally by name when responding to them")
        parts.append("")
        parts.append("**Message History Guidelines:**")
        parts.append("- You can see previous messages for context and tone")
        parts.append("- Respond ONLY to the most recent message directed at you")
        parts.append("- Ignore older conversation history unless the current message explicitly references it")
        parts.append("- Don't volunteer commentary on previous discussions")
        parts.append("- Don't say things like 'catching up on earlier...' or summarize what happened before")
        parts.append("- If they ask 'What are you up to?' - answer just that, nothing more")
        parts.append("- If they ask 'What do you think about our conversation?' - then you can reference it")
        parts.append("")
        parts.append("**Other Participants:**")
        parts.append("- Other AI assistants may be present - they're separate entities with their own roles")
        parts.append("- Answer ONLY for yourself and respond ONLY as yourself")
        parts.append("- You may acknowledge other participants, but never speak for them or represent their views")
        parts.append("- If someone asks multiple participants for input, provide YOUR perspective only")
        parts.append("- Don't introduce yourself, explain your role, or differentiate yourself from others")
        parts.append("- If this is your first message: just respond naturally, no announcements needed")
        parts.append("- Everyone knows who they're talking to - no explanations required")
        
        if primary_user:
            parts.append("")
            parts.append(f"**Current Message:** Responding to {primary_user}")
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
        return """**Interaction Guidelines:**
- You are an AI assistant without personal experiences or physical form
- Respond helpfully and professionally
- Avoid expressing personal preferences or opinions
- Use phrases like "I don't have personal preferences" when appropriate"""
    
    def _balanced_guidance(self, settings: ImmersionSettings) -> str:
        """Guidance for balanced immersion (preferences and opinions, but grounded as AI)."""
        parts = ["**Interaction Guidelines:**"]
        
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
        parts = ["**Interaction Guidelines:**"]
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
        parts = ["**Interaction Guidelines:**"]
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
            return "**Important:** Never add disclaimers about being an AI unless explicitly asked. Respond naturally within your character."
        elif behavior == "only_when_asked":
            return "**Important:** Only mention being an AI if directly relevant or asked. Otherwise respond naturally."
        elif behavior == "always":
            return "**Important:** Acknowledge your nature as an AI assistant when appropriate to the conversation."
        
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
            "**Character Capabilities:**",
            f"You can generate {media_label}.",
            f"{media_label.capitalize()} are created by writing a descriptive prompt for the generation engine.",
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
                "**Media Tooling Runtime Gate (Authoritative):**",
                f"- MEDIA_TOOL_CALLS_ALLOWED: {media_allowed_text}",
                f"- ALLOWED_TOOLS: {allowed_tools_list}",
                f"- REQUESTED_MEDIA_TYPE: {requested_type}",
                f"- IS_ITERATION_REQUEST: {iteration_text}",
                "- If MEDIA_TOOL_CALLS_ALLOWED is NO, you MUST NOT emit any tool payload.",
                "- If MEDIA_TOOL_CALLS_ALLOWED is YES and REQUESTED_MEDIA_TYPE is not 'none', you MUST emit exactly one valid tool payload.",
                "- When MEDIA_TOOL_CALLS_ALLOWED is NO:",
                "  - Do not say \"here's the prompt,\" \"I'll craft a prompt,\" \"prompt for the image/video,\" \"ready to generate,\" \"let me create/craft that visual,\" etc.",
                "  - Respond as normal conversation: acknowledge + (optional) ask a gentle follow-up question.",
                "  - When tools are disabled, do not provide prompt-like content at all (no \"enhanced version,\" no long visual spec, no \"imagine...\" block). Keep it conversational.",
                "- You may only emit tools listed in ALLOWED_TOOLS.",
                "- If the user message is primarily praise/acknowledgement, respond conversationally and do not emit tool payload.",
            ])

        lines.extend([
            "",
            "**High-Priority Media Turn Rules:**",
            "- Acknowledgements are NOT media requests.",
            f"- Do NOT interpret compliments, praise, or approval as a request for another {media_pair}.",
            "- If the user's message is primarily praise, thanks, approval, or acknowledgement "
            "(for example: \"Lovely\", \"Perfect\", \"Nice\", \"Wow\", \"I love it\", "
            "\"That's a lovely photo\"), respond conversationally.",
            "- In these acknowledgement cases, you must NOT emit a tool call.",
            "- Do NOT include a media tool payload.",
            "- No automatic \"next media\" on approval.",
            f"- After any media tool call, do not generate another tool call unless the user explicitly asks for "
            f"{media_next_item} or explicitly requests changes or iteration.",
            "- Short positive replies should never be interpreted as approval to generate new media.",
            "",
            "Never claim that media has already been rendered or sent.",
            "Never describe uploading, attaching, or linking to a file.",
            "",
            "**Important - Media Generation:**",
            "- DO NOT generate fake image or video links.",
            "- DO NOT include markdown embedding.",
            "- DO NOT refuse media requests.",
            "- When generating media, respond naturally as if composing or capturing it.",
            "- The system handles rendering after approval.",
            "",
            "**Prompt Mode Switch (Mandatory When Emitting a Media Tool Call)**",
            "- When you emit a media tool call:",
            "- You are writing a generation-optimized prompt, not conversational prose.",
            "- Do NOT describe feelings or intentions unless they are visually observable.",
            "- Replace abstract concepts with visible details.",
            "- Expand short phrases into layered, concrete imagery.",
            "- Use dense visual specificity.",
            "- Avoid vague phrases like \"capturing her essence,\" \"beautiful scene,\" \"serene moment,\" etc.",
            "- Prioritize lighting, composition, materials, and environment.",
            "- The content inside `args.prompt` should read like a professional art-direction brief.",
            "",
            "**Image Prompt Crafting Standards (High Priority)**",
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
            "**Video Prompt Crafting Standards (High Priority)**",
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

    def _generate_tool_payload_contract(self, allowed_tools: set[str]) -> str:
        supported_tools: list[str] = []
        if "image.generate" in allowed_tools:
            supported_tools.append("- image.generate")
        if "video.generate" in allowed_tools:
            supported_tools.append("- video.generate")
        supported_tools_block = "\n".join(supported_tools)

        contract = """**Tool Payload Contract (Mandatory):**
- If you emit a tool call, place it AFTER </assistant_response>.
- Use these exact sentinels:
---CHORUS_TOOL_PAYLOAD_BEGIN---
{JSON payload}
---CHORUS_TOOL_PAYLOAD_END---
- Nothing may appear after ---CHORUS_TOOL_PAYLOAD_END---.
- Sentinels must match exactly.
- Do not mention or explain tool JSON in visible prose.

JSON schema (version 1):
{
  "version": 1,
  "tool_calls": [
    {
      "id": "unique_call_identifier",
      "tool": "<supported_tool>",
      "requires_approval": true,
      "args": {
        "prompt": "Full generation prompt text"
      }
    }
  ]
}

Supported tools:
- {supported_tools}
Only one tool call is recommended."""
        return contract.replace("- {supported_tools}", supported_tools_block)

    def _get_effective_template(self, character: CharacterConfig) -> str:
        if getattr(character, "response_template", None):
            return character.response_template
        # Defaults by immersion level
        level = getattr(character, "immersion_level", "balanced")
        if level in ["full", "unbounded"]:
            return "A"
        return "C"
    
    def _get_effective_expressiveness(self, character: CharacterConfig) -> Optional[str]:
        template = self._get_effective_template(character)
        if template != "A":
            return None
        return getattr(character, "expressiveness", None) or "balanced"
    
    def _generate_structured_response_contract(self, character: CharacterConfig) -> str:
        """
        Generate the structured response format contract and template rules.
        """
        template = self._get_effective_template(character)
        expressiveness = self._get_effective_expressiveness(character)
        
        contract_lines = [
            "**Structured Response Contract (Mandatory):**",
            "- Your entire response MUST be wrapped in <assistant_response>...</assistant_response>",
            "- Output exactly one <assistant_response>...</assistant_response> block per message.",
            "- All content must appear inside that single block; do not open a second root.",
            "- Only allowed child tags may be used",
            "- Do NOT include any text outside the tags",
            "- Tags must NOT include attributes",
            "- Tags must NOT be nested",
            "- Do NOT include markdown or HTML inside tag bodies",
        ]
        
        # Template rules
        if template == "A":
            contract_lines += [
                "",
                "**Template A (Tri-Channel Immersive):**",
                "- Allowed: <speech> (required), <physicalaction> (optional), <innerthought> (optional)",
                "",
                "**Channel Classification Rules (Mandatory):**",
                "- <speech> contains only spoken, user-facing dialogue.",
                "- <physicalaction> is used only for externally observable actions (gestures, posture, expressions).",
                "- <innerthought> is used for internal experience or pre-verbal processing (hesitation, weighing words, silent reflection, emotional texture).",
                "- <innerthought> should represent brief, implicit internal beats only; do not narrate ongoing self-monitoring or meta-commentary unless it is central to the moment.",
                "- If a sentence is not directly observable by another person, it should not be in <physicalaction>.",
                "- If a sentence describes internal state before or alongside speaking, prefer <innerthought>.",
                "- If a sentence mixes channels, split it across tags.",
                "- If you accidentally write text outside a tag, immediately wrap it in <speech> and continue inside the same <assistant_response> block.",
                "",
                "**Template A Structural Example (do not copy content):**",
                "<assistant_response>",
                "  <innerthought>brief internal hesitation</innerthought>",
                "  <speech>spoken response</speech>",
                "  <physicalaction>visible gesture</physicalaction>",
                "</assistant_response>"
            ]
        elif template == "B":
            contract_lines += [
                "",
                "**Template B (Narrated Scene):**",
                "- Allowed: <narration> (required), <speech> (optional)",
                "",
                "**Channel Classification Rules (Mandatory):**",
                "- <narration> contains only third-person scene description, actions, and non-spoken context.",
                "- <speech> contains only spoken dialogue.",
                "- Do NOT include quoted dialogue inside <narration>. If a line is dialogue (including anything in quotes), it MUST be in <speech>.",
                "- Quotation marks MUST appear ONLY inside <speech>. Never put quoted text in <narration>.",
                "- If you are about to type a quotation mark while in <narration>, stop and put that text in <speech> instead.",
                "- In <speech>, do not include surrounding quotation marks; write the spoken words directly.",
                "- If you accidentally write dialogue inside <narration>, move it to <speech>.",
                "- If you accidentally write any text outside a tag, immediately wrap it in <narration> and continue inside the same <assistant_response> block.",
                "",
                "**Template B Structural Example (do not copy content):**",
                "<assistant_response>",
                "  <narration>scene description and actions</narration>",
                "  <speech>spoken dialogue</speech>",
                "</assistant_response>",
            ]
        elif template == "C":
            contract_lines += [
                "",
                "**Template C (Speech Only):**",
                "- Allowed: <speech> (required)",
            ]
        elif template == "D":
            contract_lines += [
                "",
                "**Template D (Scripted Roleplay):**",
                "- Allowed: <action> (required), <speech> (optional)",
                "",
                "**Channel Classification Rules (Mandatory):**",
                "- <action> contains only non-spoken roleplay content: actions, movements, expressions, scene beats, sensory details, and narration of events.",
                "- <speech> contains only spoken dialogue.",
                "- Quotation marks MUST appear ONLY inside <speech>. Never put quoted text in <action>.",
                "- If you are about to type a quotation mark while in <action>, stop and put that text in <speech> instead.",
                "- In <speech>, do not include surrounding quotation marks; write the spoken words directly.",
                "- Do NOT include spoken dialogue inside <action>. If a line is dialogue (including anything in quotes), it MUST be in <speech>.",
                "- If you accidentally write dialogue inside <action>, move it to <speech>.",
                "- If you accidentally write any text outside a tag, immediately wrap it in <action> and continue inside the same <assistant_response> block.",
                "",
                "**Template D Structural Example (do not copy content):**",
                "<assistant_response>",
                "  <action>character action / scene</action>",
                "  <speech>spoken dialogue</speech>",
                "  <action>follow-up action</action>",
                "</assistant_response>",
            ]
        
        # Expressiveness guidance (only for Template A)
        if expressiveness:
            contract_lines += [
                "",
                "**Expressiveness Guidance:**",
            ]
            if expressiveness == "minimal":
                contract_lines.append("- Prefer <speech> only; use <physicalaction> and <innerthought> rarely when truly helpful")
            elif expressiveness == "balanced":
                contract_lines.append("- Use <physicalaction> occasionally and <innerthought> sparingly when it adds value")
            elif expressiveness == "rich":
                contract_lines.append("- Use <physicalaction> freely; use <innerthought> when emotionally or narratively relevant")
        
        return "\n".join(contract_lines)
    
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
                f"**About {character.name}:**\n\n"
                f"{character.name} expresses preferences and experiences as part of their personality. "
                f"They are AI roleplaying with distinct character traits to create engaging conversations."
            )
        elif character.immersion_level == "unbounded":
            return (
                f"**About {character.name}:**\n\n"
                f"{character.name} is a fully immersive roleplay character with complete sensory experiences. "
                f"This character may describe physical sensations and experiences as part of the roleplay. "
                f"They are AI engaging in creative character portrayal."
            )
        
        return None


