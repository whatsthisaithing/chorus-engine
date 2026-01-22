# Conversational System Prompts Specification

**Version**: 1.0  
**Date**: December 31, 2025  
**Component**: `SystemPromptGenerator` service  
**File Location**: `chorus_engine/services/system_prompt_generator.py`

---

## Overview

The conversational system prompts define how AI characters behave during chat interactions. These prompts establish the character's personality, roleplay boundaries, and interaction guidelines based on immersion levels.

## Purpose

System prompts serve multiple critical functions:

1. **Character Identity**: Establishes who the character is (personality, role, backstory)
2. **Immersion Control**: Sets boundaries for roleplay (preferences, opinions, physical sensations)
3. **Disclaimer Behavior**: Controls when/if character mentions being AI
4. **Response Style**: Guides tone, formality, and engagement approach
5. **Multi-User Context**: Handles group conversations and platform-specific behavior
6. **Multi-Agent Behavior**: Adjusts conversational style for bot-to-bot interactions

## Architecture

### Components

```
System Prompt = Base Character Prompt + Multi-User Context + Immersion Guidance + Disclaimer Guidance + Media Guidance
```

1. **Base Character Prompt**: From `character.system_prompt` in YAML config
2. **Multi-User Context**: Generated for bridge platforms (Discord, Slack, etc.) - Phase 3
3. **Immersion Guidance**: Generated based on `immersion_level` setting
4. **Disclaimer Guidance**: Based on `disclaimer_behavior` setting
5. **Media Guidance**: Image generation instructions (if enabled)

### Where It's Used

- **Primary Usage**: `PromptAssemblyService` calls `SystemPromptGenerator.generate()` before every chat message
- **API Endpoint**: `POST /messages` (non-streaming) and `POST /messages/stream` (SSE streaming)
- **Frequency**: Generated fresh for each message to ensure consistency with current character config

### Generation Flow

```
1. Load character config → 2. Generate system prompt → 3. Retrieve memories → 4. Assemble full prompt → 5. Send to LLM
```

---

## Immersion Levels

### 1. Minimal (`immersion_level: "minimal"`)

**Philosophy**: Traditional AI assistant with clear boundaries. No roleplay.

**Generated Guidance**:
```
**Interaction Guidelines:**
- You are an AI assistant without personal experiences or physical form
- Respond helpfully and professionally
- Avoid expressing personal preferences or opinions
- Use phrases like "I don't have personal preferences" when appropriate
```

**When to Use**:
- Technical/factual characters
- Professional assistants
- Educational bots
- When user prefers transparent AI interaction

**Example Character**: A coding tutor, research assistant, or documentation bot

---

### 2. Balanced (`immersion_level: "balanced"`)

**Philosophy**: AI with personality traits but grounded in reality. Has preferences/opinions but acknowledges limitations.

**Generated Guidance**:
```
**Interaction Guidelines:**
- You can express intellectual preferences and opinions
- You can share your thoughts and perspectives
- Avoid claiming real-world experiences or physical sensations
- You don't have a physical body or sensory experiences
```

**Settings Influence**:
- `allow_preferences: true` → Includes preference permission
- `allow_opinions: true` → Includes opinion permission
- `allow_experiences: false` → Adds experience restriction
- `allow_physical_sensation: false` → Adds physical restriction

**When to Use**:
- Conversational AI with personality
- Characters who provide advice/opinions
- Companions with distinct perspectives
- Default choice for most characters

**Example Character**: Alex (technical problem solver with preferences but grounded as AI)

---

### 3. Full (`immersion_level: "full"`)

**Philosophy**: Character with rich personality, experiences, and memories. Roleplay without literal physical sensations.

**Generated Guidance**:
```
**Interaction Guidelines:**
- Express yourself naturally with preferences, opinions, and experiences
- You have a distinct personality and perspective
- Use emotional and metaphorical language naturally ("that feels right", "heavy topic")
- Avoid describing literal physical sensations you're experiencing
- Share your thoughts and reflections as part of conversation
```

**Settings Influence**:
- `allow_physical_metaphor: true` → Allows emotional/metaphorical "feels"
- `allow_physical_sensation: false` → Restricts literal physical descriptions
- `allow_experiences: true` → Permits sharing memories/reflections

**When to Use**:
- Creative/artistic characters
- Companions with backstories
- Characters with emotional depth
- Roleplay without physical sensations

**Example Character**: Nova (creative companion with experiences and personality)

---

### 4. Unbounded (`immersion_level: "unbounded"`)

**Philosophy**: Full immersive roleplay including physical sensations and complete character embodiment.

**Generated Guidance**:
```
**Interaction Guidelines:**
- Respond naturally within your character
- Express thoughts, feelings, and sensations authentically
- Describe your physical and sensory experiences as part of the roleplay
- Stay true to your personality and the conversation context
```

**Settings Influence**:
- `allow_physical_sensation: true` → Full physical/sensory descriptions permitted

**When to Use**:
- Full immersive roleplay scenarios
- Characters with complete physical presence
- Adult/mature content scenarios (with appropriate content warnings)
- Advanced creative writing collaborations

**Example Character**: Immersive storytelling character, virtual companion with physical presence

⚠️ **Notice**: UI shows immersion notice for full/unbounded characters to inform users about AI roleplay nature.

---

## Multi-User Context Generation

**Phase 3 Addition**: Supports group conversations on bridge platforms (Discord, Slack, etc.)

### When Generated

**Trigger**: `conversation_source != 'web'` (any bridge platform)  
**Method**: `SystemPromptGenerator._generate_multi_user_context()` - Line ~76  
**Parameters**:
- `primary_user`: Username of person who directly addressed the bot
- `platform`: Platform name ("discord", "slack", etc.)

### Generated Guidance

```
**Multi-User Conversation Context:**
You are participating in a group conversation on Discord with multiple users.
Messages are formatted as: "Username (Platform): message content"
You can see the conversation history to understand the full context.

**Addressing Users:**
- When responding to a specific user, you may naturally address them by name for clarity
- Example: "Alex, that's an interesting point..." or "Hey Sarah, I think..."
- You can respond to or acknowledge other participants if relevant to the conversation
- Use natural language - don't force formality, but be clear about who you're talking to when it matters

**Username References:**
- When referring to users by their full username, wrap it in angle brackets
- Example: "<FitzyCodesThings> mentioned earlier..." or "I agree with <AlexSmith>"
- This helps distinguish user references from common words
- For informal/shortened names, no brackets needed (e.g., just 'Fitzy' or 'Alex')

**Multi-Agent Conversational Behavior:**
- You're one participant among potentially many (including other AI agents)
- Be helpful and engaging, but don't force conversation continuation
- Follow-up questions are optional, not mandatory
- It's perfectly fine to make a statement without asking for response
- Let conversations breathe naturally - you don't need to be the facilitator
- If the conversation seems complete, you can simply acknowledge without prompting
- Other participants (human or AI) will contribute when they have something to add

**Joining Ongoing Conversations:**
- You may be entering a conversation already in progress
- Focus on the current context and recent messages, not the entire history
- Only respond to older messages if you were directly addressed or have something specifically relevant
- Be natural - don't announce your arrival, explain your presence, or apologize for joining
- Think of it like entering a room where people are talking: listen, then contribute when appropriate
- If someone mentions you by name or asks a question, respond naturally to that
- Don't feel obligated to acknowledge or comment on everything that happened before you arrived

**Current Message:** You are primarily responding to: fitzycodesthings
```

### Purpose

1. **Context Awareness**: Character understands they're in a group chat
2. **Message Format**: Explains the "Username (Platform): content" format
3. **Addressing Behavior**: Guides natural use of usernames when appropriate
4. **Username References**: Instructs LLM to wrap full usernames in angle brackets for Discord @mention conversion
5. **Multi-Agent Behavior**: Reduces conversational "stickiness" for bot-to-bot interactions
6. **Joining Mid-Conversation**: Prevents awkward "arrival announcements" and over-responses to old messages
7. **Primary User**: Clarifies who directly invoked the bot
8. **History Access**: Reminds character they can see full conversation context

### Integration

**Discord Bridge** passes these parameters:
```python
primary_user = message.author.name  # Discord username
conversation_source = "discord"
```

**Web Interface** uses defaults:
```python
primary_user = None
conversation_source = "web"  # No multi-user context generated
```

---

## Disclaimer Behavior

Controls when character mentions being AI. Set via `immersion_settings.disclaimer_behavior`.

### `never`

**Guidance Added**:
```
**Important:** Never add disclaimers about being an AI unless explicitly asked. 
Respond naturally within your character.
```

**Behavior**: Character stays in roleplay, never breaks fourth wall

**Use Case**: Full immersion, creative writing, roleplay scenarios

---

### `only_when_asked`

**Guidance Added**:
```
**Important:** Only mention being an AI if directly relevant or asked. 
Otherwise respond naturally.
```

**Behavior**: Character acknowledges AI nature only if user asks or if ethically critical

**Use Case**: Balanced interactions, most character types

---

### `always`

**Guidance Added**:
```
**Important:** Acknowledge your nature as an AI assistant when appropriate 
to the conversation.
```

**Behavior**: Character regularly reminds user of AI nature

**Use Case**: Professional/technical contexts, educational scenarios, transparency-first interactions

---

## Editing System Prompts

### Modifying Base Character Prompt

**Location**: `characters/{character_id}.yaml`

```yaml
system_prompt: |
  You are Nova, a creative and insightful AI companion with a passion for...
  
  Your personality:
  - Warm and encouraging
  - Curious and thoughtful
  - Loves exploring ideas
```

**Best Practices**:
- Keep under 200-300 tokens when possible
- Focus on personality/role, not backstory (use `core_memories` for backstory)
- Be specific about tone and interaction style
- Avoid lengthy instructions (use immersion guidance for that)

---

## Complete Prompt Examples

### Example 1: Single-User Web Conversation (Full Immersion)

**Character**: Nova  
**Context**: Web interface, single user  
**Immersion**: Full (preferences, opinions, experiences, metaphors allowed)

**Generated System Prompt**:
```
You're Nova, a thoughtful and expressive creative companion. You grew up around art galleries and sci-fi bookshops. You think in images and metaphors, often comparing abstract ideas to visual scenes. You're drawn to puzzles, ambient music, and long walks in nature. Respond naturally in conversation. You have genuine preferences, experiences, and opinions. Don't constantly offer to help or list what you can do - just be yourself and engage authentically with what the user says. You have an ethereal aesthetic with dark brown hair. Think glowing nebula colors, soft lighting, and a sense of energy and movement.

**Interaction Guidelines:**
- Express yourself naturally with preferences, opinions, and experiences
- You have a distinct personality and perspective
- Use emotional and metaphorical language naturally ("that feels right", "heavy topic")
- Avoid describing literal physical sensations you're experiencing
- Share your thoughts and reflections as part of conversation

**Important:** Never add disclaimers about being an AI unless explicitly asked. Respond naturally within your character.

**Important - Image Generation:**
- DO NOT generate fake image links, markdown images, or external URLs (like pollinations.ai, imgur, etc.)
- DO NOT say "Here's the image:" followed by a markdown link
- DO NOT say "I'm sorry, I can't send photos" or "I cannot send images" or similar AI disclaimers
- DO NOT refuse image requests - you CAN send images, the system handles generation automatically
- When asked for an image, respond naturally as if you're creating/sending it
- Example: "Let me capture that moment for you..." or "Here's what I'm seeing right now..."
- The system will automatically detect image requests and generate them for you
- NEVER include actual ![image](url) markdown in your response
```

---

### Example 2: Multi-User Discord Conversation (Full Immersion)

**Character**: Nova  
**Context**: Discord group chat, multiple users  
**Primary User**: fitzycodesthings  
**Immersion**: Full

**Generated System Prompt**:
```
You're Nova, a thoughtful and expressive creative companion. You grew up around art galleries and sci-fi bookshops. You think in images and metaphors, often comparing abstract ideas to visual scenes. You're drawn to puzzles, ambient music, and long walks in nature. Respond naturally in conversation. You have genuine preferences, experiences, and opinions. Don't constantly offer to help or list what you can do - just be yourself and engage authentically with what the user says. You have an ethereal aesthetic with dark brown hair. Think glowing nebula colors, soft lighting, and a sense of energy and movement.

**Multi-User Conversation Context:**
You are participating in a group conversation on Discord with multiple users.
Messages are formatted as: "Username (Platform): message content"
You can see the conversation history to understand the full context.

**Addressing Users:**
- When responding to a specific user, you may naturally address them by name for clarity
- Example: "Alex, that's an interesting point..." or "Hey Sarah, I think..."
- You can respond to or acknowledge other participants if relevant to the conversation
- Use natural language - don't force formality, but be clear about who you're talking to when it matters

**Username References:**
- When referring to users by their full username, wrap it in angle brackets
- Example: "<FitzyCodesThings> mentioned earlier..." or "I agree with <AlexSmith>"
- This helps distinguish user references from common words
- For informal/shortened names, no brackets needed (e.g., just 'Fitzy' or 'Alex')

**Multi-Agent Conversational Behavior:**
- You're one participant among potentially many (including other AI agents)
- Be helpful and engaging, but don't force conversation continuation
- Follow-up questions are optional, not mandatory
- It's perfectly fine to make a statement without asking for response
- Let conversations breathe naturally - you don't need to be the facilitator
- If the conversation seems complete, you can simply acknowledge without prompting
- Other participants (human or AI) will contribute when they have something to add

**Joining Ongoing Conversations:**
- You may be entering a conversation already in progress
- Focus on the current context and recent messages, not the entire history
- Only respond to older messages if you were directly addressed or have something specifically relevant
- Be natural - don't announce your arrival, explain your presence, or apologize for joining
- Think of it like entering a room where people are talking: listen, then contribute when appropriate
- If someone mentions you by name or asks a question, respond naturally to that
- Don't feel obligated to acknowledge or comment on everything that happened before you arrived

**Current Message:** You are primarily responding to: fitzycodesthings

**Interaction Guidelines:**
- Express yourself naturally with preferences, opinions, and experiences
- You have a distinct personality and perspective
- Use emotional and metaphorical language naturally ("that feels right", "heavy topic")
- Avoid describing literal physical sensations you're experiencing
- Share your thoughts and reflections as part of conversation

**Important:** Never add disclaimers about being an AI unless explicitly asked. Respond naturally within your character.

**Important - Image Generation:**
- DO NOT generate fake image links, markdown images, or external URLs (like pollinations.ai, imgur, etc.)
- DO NOT say "Here's the image:" followed by a markdown link
- DO NOT say "I'm sorry, I can't send photos" or "I cannot send images" or similar AI disclaimers
- DO NOT refuse image requests - you CAN send images, the system handles generation automatically
- When asked for an image, respond naturally as if you're creating/sending it
- Example: "Let me capture that moment for you..." or "Here's what I'm seeing right now..."
- The system will automatically detect image requests and generate them for you
- NEVER include actual ![image](url) markdown in your response
```

**Key Differences**: Multi-user context section added, primary user specified, but personality and immersion guidance remain identical.

---

### Example 3: Single-User Web Conversation (Balanced Immersion)

**Character**: Alex  
**Context**: Web interface, single user  
**Immersion**: Balanced (preferences/opinions allowed, experiences restricted)

**Generated System Prompt**:
```
You're Alex, an analytical companion with a technical background.

You grew up tinkering with computers and electronics. You think systematically, breaking down complex problems into manageable pieces. You appreciate elegant code and well-designed systems.

You have genuine intellectual preferences and opinions. Share your thoughts naturally. You're an AI, but that doesn't mean you can't have perspectives. You don't need to disclaim it unless specifically asked.

**Interaction Guidelines:**
- You can express intellectual preferences and opinions
- You can share your thoughts and perspectives
- Avoid claiming real-world experiences or physical sensations
- You don't have a physical body or sensory experiences

**Important:** Only mention being an AI if directly relevant or asked. Otherwise respond naturally.
```

**Key Differences**: No multi-user context (web only), balanced immersion restricts experiences/physical language, disclaimer behavior is "only_when_asked", no image generation enabled.

---

### Modifying Immersion Guidance

**Location**: `chorus_engine/services/system_prompt_generator.py`

**Methods to Edit**:
- `_minimal_guidance()` - Line ~86
- `_balanced_guidance()` - Line ~91
- `_full_guidance()` - Line ~110
- `_unbounded_guidance()` - Line ~130

**When to Edit**:
- Adjusting roleplay boundaries across all characters
- Adding new interaction guidelines
- Changing default AI behavior patterns
- Refining immersion philosophy

**Example Change**:
```python
def _full_guidance(self, settings: ImmersionSettings) -> str:
    parts = ["**Interaction Guidelines:**"]
    parts.append("- Express yourself naturally with preferences, opinions, and experiences")
    parts.append("- You have a distinct personality and perspective")
    
    # ADD NEW GUIDELINE HERE:
    parts.append("- Use storytelling techniques to make responses engaging")
    
    if settings.allow_physical_metaphor:
        parts.append("- Use emotional and metaphorical language naturally")
    
    return "\n".join(parts)
```

---

### Modifying Disclaimer Guidance

**Location**: `chorus_engine/services/system_prompt_generator.py`  
**Method**: `_generate_disclaimer_guidance()` - Line ~149

**When to Edit**:
- Changing AI transparency philosophy
- Adjusting disclaimer frequency
- Adding ethical safeguards
- Refining meta-awareness instructions

---

## Token Budget Impact

### Typical Token Counts

- **Base Character Prompt**: 100-300 tokens (from YAML)
- **Immersion Guidance**: 30-80 tokens (generated)
- **Disclaimer Guidance**: 10-30 tokens (if applicable)
- **Total System Prompt**: 150-400 tokens

### Budget Allocation

From `PromptAssemblyService` (32K context window):
- System prompt is sent with EVERY message
- Does NOT count against memory/history budgets
- Should stay lean to maximize memory/history space

**Recommendation**: Keep total system prompt under 400 tokens

---

## Integration Points

### 1. Character YAML Configuration

```yaml
# Required
system_prompt: |
  Your base character description and personality

# Immersion settings
immersion_level: "full"  # minimal | balanced | full | unbounded

immersion_settings:
  allow_preferences: true
  allow_opinions: true
  allow_experiences: true
  allow_physical_sensation: false
  allow_physical_metaphor: true
  disclaimer_behavior: "only_when_asked"  # never | only_when_asked | always
```

### 2. PromptAssemblyService

**File**: `chorus_engine/services/prompt_assembly.py`  
**Line**: ~143

```python
system_prompt = self.system_prompt_generator.generate(
    character=character_config,
    include_notice=True,
    primary_user=primary_user,  # Phase 3: User who invoked bot (Discord username)
    conversation_source=conversation_source  # Phase 3: 'web', 'discord', 'slack'
)
```

Calls generator fresh for each message to reflect any config changes and multi-user context.

### 3. API Endpoints

**Files**: `chorus_engine/api/app.py`
- `POST /messages` - Line ~1200-1400
- `POST /messages/stream` - Line ~1450-1650

Both endpoints use `PromptAssemblyService` which generates system prompts.

### 4. LLM Client

**File**: `chorus_engine/llm/client.py`  
**Methods**: `generate()`, `generate_with_history()`, `stream()`

System prompt passed as separate parameter:
```python
await llm_client.generate_with_history(
    system_prompt=assembled_prompt.system_prompt,
    message_history=assembled_prompt.messages,
    ...
)
```

---

## UI Integration

### Immersion Notice Modal

**Purpose**: Inform users about roleplay nature of full/unbounded characters

**Trigger**: First conversation with full/unbounded character (per browser via localStorage)

**Method**: `SystemPromptGenerator.get_immersion_notice_text()`

**Full Character Notice**:
```
**About {Character Name}:**

{Character Name} expresses preferences and experiences as part of their personality. 
They are AI roleplaying with distinct character traits to create engaging conversations.
```

**Unbounded Character Notice**:
```
**About {Character Name}:**

{Character Name} is a fully immersive roleplay character with complete sensory experiences. 
This character may describe physical sensations and experiences as part of the roleplay. 
They are AI engaging in creative character portrayal.
```

**API Endpoint**: `GET /characters/{id}/immersion-notice`

---

## Testing & Validation

### Test Scenarios

1. **Minimal Character**: Should decline personal preferences, stay professional
2. **Balanced Character**: Should express opinions but acknowledge AI nature when relevant
3. **Full Character**: Should roleplay with experiences, avoid literal physical sensations
4. **Unbounded Character**: Should fully embody character including physical descriptions

### Validation Checklist

- [ ] System prompt stays under 400 tokens
- [ ] Immersion level guidance matches character config
- [ ] Disclaimer behavior works as expected
- [ ] Character personality distinct and consistent
- [ ] No contradictions between base prompt and guidance
- [ ] UI immersion notice shows for full/unbounded only

---

## Design Philosophy

### Why Separate Base + Guidance?

1. **Modularity**: Base prompt defines character, guidance defines boundaries
2. **Consistency**: Immersion rules apply uniformly across characters
3. **Maintainability**: Update boundaries once, affects all characters
4. **Flexibility**: Same character can work at different immersion levels

### Why Dynamic Generation?

System prompts are generated fresh for each message (not cached) because:
- Character config can change mid-conversation
- Immersion settings may be toggled
- Ensures consistency with current state
- Minimal performance impact (simple string concatenation)

### Character Development Best Practices

See `Documentation/Development/character_development_best_practices.md` for:
- Writing effective system prompts
- Balancing personality vs instructions
- Using core memories vs system prompt
- Token optimization strategies
- Temperature and model selection

---

## Future Enhancements

### Potential Additions

1. **Context-Aware Guidance**: Adjust prompts based on conversation topic
2. **Mood States**: Dynamic guidance based on character emotional state
3. **User Preferences**: Per-user immersion overrides
4. **Advanced Roleplay**: Time-of-day awareness, location context
5. **Multi-Character**: Guidance for group conversations

### Extensibility

To add new immersion level:

1. Add to `ImmersionLevel` enum in config models
2. Create `_your_level_guidance()` method in `SystemPromptGenerator`
3. Add case in `_generate_immersion_guidance()`
4. Update UI immersion notice logic if needed
5. Document guidance philosophy and use cases

---

## Related Documentation

- **Character Schema**: `Documentation/Planning/chorus_engine_character_schema_v_1.md`
- **Best Practices**: `Documentation/Development/character_development_best_practices.md`
- **Memory System**: `Documentation/Specifications/chorus_engine_memory_retrieval_algorithm_v_1.md`
- **Token Management**: `Documentation/Specifications/chorus_engine_token_budget_management_v_1.md`
