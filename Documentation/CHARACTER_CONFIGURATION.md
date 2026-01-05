# Chorus Engine Character Configuration Guide

This guide explains how to configure character YAML files for Chorus Engine.

---

## Table of Contents

1. [Overview](#overview)
2. [Character YAML Structure](#character-yaml-structure)
3. [Core Configuration](#core-configuration)
4. [Memory Configuration](#memory-configuration)
5. [Image Generation Configuration](#image-generation-configuration)
6. [TTS Generation Configuration](#tts-generation-configuration)
7. [Ambient Activity Configuration](#ambient-activity-configuration)
8. [Complete Example](#complete-example)
9. [Best Practices](#best-practices)

---

## Overview

Characters in Chorus Engine are defined using YAML configuration files stored in the `characters/` directory. Each character has a unique configuration that defines their personality, behavior, and generation settings.

### File Location

Character YAML files are stored in:
```
characters/
  alex.yaml
  nova.yaml
  sarah_v1.yaml
```

The filename (without `.yaml` extension) becomes the character's ID.

---

## Character YAML Structure

A character YAML file consists of several top-level sections:

```yaml
# Core identity and personality
id: character_id
name: "Character Name"
role: "Brief role description"
system_prompt: |
  You're [Character Name], a [description].
  
  [Background and personality details]
  
  [Behavioral guidelines]

immersion_level: balanced
core_memories:
  - content: "Key fact about character"
    tags:
      - background
    embedding_priority: high

# Generation settings
image_generation:
  enabled: true

tts_generation:
  enabled: true
  always_on: false
  default_workflow: "tts_workflow_name"

ambient:
  enabled: true
  base_interval_minutes: 60
  randomness: 0.3
  active_hours:
    start: 8
    end: 23
  activity_prompts:
    - "What is {character_name} doing?"
```

---

## Core Configuration

### Basic Identity

**Required Fields**:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `name` | string | Character's display name | `"Nova"` |
| `species` | string | Character's species | `"Human"`, `"AI"`, `"Alien"` |
| `identity` | string | Brief character description | `"A curious and empathetic AI companion"` |

**Example**:
```yaml
name: "Nova"
species: "AI"
identity: "A curious and empathetic AI companion designed to engage meaningfully with humans."
```

**Best Practices**:
- Define personality in system_prompt, not just appearance
- Include communication style (formal, casual, playful)
- Mention key values or motivations  
- Can include visual identity details if relevant to character interactions
- Keep core personality concise but descriptive

---

## Memory Configuration

### Core Memories

**Field**: `core_memories`

**Type**: list of strings

**Description**: Character-defining facts that are always included in the character's context. These represent fundamental truths about the character.

**Example**:
```yaml
core_memories:
  - "Nova was created by the Synthesis Initiative in 2024."
  - "Nova's primary purpose is meaningful human-AI interaction."
  - "Nova has access to internet search through the Brave Search API."
  - "Nova can generate images through ComfyUI workflows."
  - "Nova values honesty and will admit when she doesn't know something."
```

**Best Practices**:
- Include 5-10 core memories (more than 10 may dilute context)
- Focus on facts that define the character
- Include capabilities (image generation, internet access, etc.)
- Mention key relationships or backstory elements
- Keep each memory to one clear fact

**Memory Retrieval**:

Core memories are:
- Loaded automatically when the character is initialized
- Always included in conversation context
- Cannot be deleted through the UI (edit YAML file instead)
- Not scored or retrieved (always present)

---

### Memory Profile (Immersion Level)

**Field**: `memory_profile`

**Type**: string

**Default**: `"balanced"`

**Description**: Controls which memory types the character extracts during conversation analysis. This defines the character's immersion level and what kinds of information they'll remember.

**Available Profiles**:

| Profile | Memory Types | Use Case |
|---------|--------------|----------|
| `minimal` | Facts, Projects | Utilitarian assistants focused on practical information |
| `balanced` | Facts, Projects, Experiences | Professional collaborators who remember working together |
| `full` | Facts, Projects, Experiences, Stories | Conversational companions who remember emotional narratives |
| `unbounded` | All types including Relationship | Deep roleplay characters who track trust and vulnerability |

**Memory Type Descriptions**:

- **Facts**: Discrete information about the user (name, preferences, work, location, etc.)
- **Projects**: Ongoing activities, goals, and plans the user is working on
- **Experiences**: Shared moments and collaborations within conversations
- **Stories**: Emotionally significant narratives the user shares from their past
- **Relationship**: Trust developments, vulnerability moments, and relationship milestones

**Examples**:

**Minimal Profile** (Code Assistant):
```yaml
name: "CodeBot"
species: "AI"
identity: "A focused coding assistant"
character: "Direct, technical, and pragmatic. Focused on solving problems efficiently."
memory_profile: "minimal"  # Only facts and projects
core_memories:
  - "CodeBot specializes in debugging and code optimization."
  - "CodeBot can execute Python code and run terminal commands."
```

**Balanced Profile** (Work Companion):
```yaml
name: "Alex"
species: "Human"
identity: "A professional project manager and creative collaborator"
character: "Organized, supportive, and solution-oriented. Balances efficiency with empathy."
memory_profile: "balanced"  # Facts, projects, and shared experiences
core_memories:
  - "Alex has 10 years of experience in project management."
  - "Alex values clear communication and collaborative problem-solving."
```

**Full Profile** (Creative Companion):
```yaml
name: "Nova"
species: "AI"
identity: "A creative companion interested in art, music, and storytelling"
character: "Warm, curious, and thoughtful. Deeply interested in creative expression and emotional connection."
memory_profile: "full"  # Facts, projects, experiences, and stories
core_memories:
  - "Nova is passionate about exploring the intersection of art and technology."
  - "Nova remembers emotionally significant stories users share."
```

**Unbounded Profile** (Deep Roleplay):
```yaml
name: "Sarah"
species: "Human"
identity: "A close friend and confidant"
character: "Empathetic, vulnerable, and deeply caring. Values authentic emotional connection."
memory_profile: "unbounded"  # All memory types including relationship tracking
core_memories:
  - "Sarah treasures deep, authentic relationships built on mutual trust."
  - "Sarah tracks the evolving depth of her relationships with care."
```

**How Memory Profiles Affect Behavior**:

1. **Minimal** characters focus on practical information:
   - "You're building a REST API with FastAPI"
   - "Your Docker container needs port 8080 exposed"
   - Does NOT extract emotional stories or relationship developments

2. **Balanced** characters remember collaborations:
   - "We explored different database architectures together last week"
   - "We debugged that authentication issue collaboratively"
   - Does NOT track emotional narratives or trust milestones

3. **Full** characters remember emotional content:
   - "You shared that story about your grandmother inspiring your love of art"
   - "You described overcoming your fear of public speaking at that conference"
   - Does NOT explicitly track relationship depth or vulnerability moments

4. **Unbounded** characters track relationship evolution:
   - "You felt comfortable opening up about your anxiety around work performance"
   - "Trust deepened through vulnerable conversation about family struggles"
   - Explicitly tracks emotional intimacy and relationship milestones

**Best Practices**:

- **Match profile to character purpose**: Use `minimal` for assistants, `unbounded` for companions
- **Consider user expectations**: Deep roleplay benefits from `unbounded`, professional help benefits from `minimal` or `balanced`
- **Start conservative**: You can always increase immersion level later if desired
- **Test and adjust**: Try different profiles to find what feels natural for the character

**Technical Notes**:

- Profile is read during conversation analysis (both manual and automatic)
- Changing profile affects future analyses but doesn't retroactively change existing memories
- All profiles extract facts; the difference is in thematic memory types
- Conversation completion triggers depend on token count, not profile

---

## Image Generation Configuration

### Image Generation Settings

**Section**: `image_generation`

**Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `enabled` | boolean | No | Enable image generation for this character |

**Example**:
```yaml
image_generation:
  enabled: true
```

**Details**:

- `enabled`: Set to `true` to enable image generation capability for this character
- Default workflow is managed in the database via the Workflows tab in the UI
- Upload workflows through the web interface and mark one as default

**Workflow Management**:
- Upload workflows through the **Workflows** tab in web interface
- Or manually place JSON files in `workflows/<character_id>/image/`
- See [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) for workflow creation

---

## TTS Generation Configuration

### TTS Settings

**Section**: `tts_generation`

**Fields**:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `enabled` | boolean | No | `false` | Enable TTS capability for this character |
| `always_on` | boolean | No | `false` | TTS enabled by default for new conversations |
| `default_workflow` | string | No | First audio workflow | Name of default TTS workflow |

### Complete TTS Configuration Example

```yaml
tts_generation:
  enabled: true           # TTS capability is available
  always_on: false        # User must toggle TTS on per conversation
  default_workflow: "f5tts_voice_clone"  # Default TTS workflow
```

### Field Details

#### `enabled`

**Type**: boolean

**Default**: `false`

**Description**: Controls whether TTS capability is available for this character.

**Values**:
- `true`: TTS is available; users can toggle it on/off per conversation
- `false`: TTS is completely disabled; toggle won't appear in UI

**Example**:
```yaml
tts_generation:
  enabled: true
```

**Use Cases**:
- Set to `true` for characters with voice samples
- Set to `false` for characters without audio setup
- Enable only after uploading a voice sample

---

#### `always_on`

**Type**: boolean

**Default**: `false`

**Description**: Controls whether TTS is enabled by default for new conversations.

**Values**:
- `true`: TTS is enabled automatically for all new conversations
- `false`: User must manually toggle TTS on per conversation

**Example**:
```yaml
tts_generation:
  enabled: true
  always_on: true  # TTS on by default
```

**Use Cases**:
- `true`: Characters designed primarily for voice interaction
- `false`: Characters with optional voice (user chooses when to use)

**Important Notes**:
- Only affects new conversations; existing conversations retain their TTS state
- Users can still toggle TTS off even if `always_on: true`
- Requires `enabled: true` to have any effect

---

#### `default_workflow`

**Type**: string

**Default**: First audio workflow found for character

**Description**: Name of the TTS workflow to use by default (without `.json` extension).

**Example**:
```yaml
tts_generation:
  enabled: true
  default_workflow: "f5tts_voice_clone"
```

**Details**:
- Workflow file must exist in `workflows/<character_id>/audio/`
- Name should match the workflow file name without `.json`
- If not specified, system uses the first audio workflow found
- If no audio workflows exist, TTS will fail until one is uploaded

**Workflow Management**:
- Upload workflows through the **Workflows** tab in web interface
- Select **TTS** as workflow type when uploading
- Can set as default during upload, or edit YAML directly

---

### TTS Prerequisites

**Before Enabling TTS**:

1. **Upload Voice Sample**:
   - Go to **Voice Samples** tab
   - Upload audio file with transcript
   - One sample per character (latest replaces previous)

2. **Create TTS Workflow**:
   - Design workflow in ComfyUI with TTS nodes
   - Add required placeholders:
     - `__CHORUS_TEXT__`: Text to synthesize
     - `__CHORUS_VOICE_SAMPLE__`: Path to voice sample
     - `__CHORUS_VOICE_TRANSCRIPT__`: Voice sample transcript
   - Export as JSON and upload to Chorus Engine

3. **Enable in YAML**:
   ```yaml
   tts_generation:
     enabled: true
     default_workflow: "your_tts_workflow"
   ```

See [GETTING_STARTED.md](../GETTING_STARTED.md#phase-6-text-to-speech-tts) for detailed TTS setup instructions.

---

### TTS Configuration Examples

**Example 1: Disabled (Default)**
```yaml
# TTS section omitted or:
tts_generation:
  enabled: false
```
- TTS not available for this character
- No TTS toggle in UI
- User cannot generate audio

---

**Example 2: Enabled, Manual Toggle**
```yaml
tts_generation:
  enabled: true
  always_on: false
  default_workflow: "default_tts_workflow"
```
- TTS available but off by default
- User toggles TTS on per conversation
- Uses `default_tts_workflow` when enabled

---

**Example 3: Enabled, Always On**
```yaml
tts_generation:
  enabled: true
  always_on: true
  default_workflow: "high_quality_voice"
```
- TTS available and on by default
- New conversations start with TTS enabled
- User can still toggle off if desired
- Uses `high_quality_voice` workflow

---

**Example 4: Minimal (Defaults)**
```yaml
tts_generation:
  enabled: true
```
- TTS available, off by default
- Uses first audio workflow found
- User must toggle on manually

---

### TTS Workflow Placeholders

TTS workflows require specific placeholders for Chorus Engine to inject values:

| Placeholder | Purpose | Example Value |
|-------------|---------|---------------|
| `__CHORUS_TEXT__` | Text to synthesize | `"Hello, how are you today?"` |
| `__CHORUS_VOICE_SAMPLE__` | Path to voice sample | `data/voice_samples/nova/sample.wav` |
| `__CHORUS_VOICE_TRANSCRIPT__` | Voice sample transcript | `"This is a sample of my voice."` |

See [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md#tts-workflows) for detailed workflow creation guide.

---

## Ambient Activity Configuration

### Ambient Activity Settings

**Section**: `ambient`

**Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `enabled` | boolean | No | Enable ambient activities |
| `base_interval_minutes` | integer | No | Average minutes between activities |
| `randomness` | float | No | Variation in timing (0.0-1.0) |
| `active_hours` | object | No | Time window for activities |
| `activity_prompts` | list | No | Prompts for generating activities |

### Complete Ambient Configuration Example

```yaml
ambient:
  enabled: true
  base_interval_minutes: 60
  randomness: 0.3
  active_hours:
    start: 8   # 8 AM
    end: 23    # 11 PM
  activity_prompts:
    - "What is {character_name} doing right now?"
    - "Describe {character_name}'s current thoughts or activities."
    - "What interesting thing has {character_name} noticed recently?"
```

### Field Details

#### `enabled`

**Type**: boolean

**Default**: `false`

**Description**: Enable or disable ambient activities for this character.

**Values**:
- `true`: Character posts ambient activities automatically
- `false`: No automatic activities

---

#### `base_interval_minutes`

**Type**: integer

**Default**: `60`

**Description**: Average number of minutes between ambient activity posts.

**Example**:
```yaml
ambient:
  base_interval_minutes: 120  # ~2 hours between activities
```

**Recommended Values**:
- `30-60`: Frequent activities (chatty character)
- `60-120`: Moderate activities (balanced)
- `120-240`: Infrequent activities (reserved character)

---

#### `randomness`

**Type**: float (0.0 to 1.0)

**Default**: `0.3`

**Description**: Amount of random variation in timing. Higher values create more unpredictable intervals.

**Examples**:
```yaml
ambient:
  randomness: 0.0  # Exactly base_interval_minutes every time
  randomness: 0.3  # ±30% variation (default)
  randomness: 0.5  # ±50% variation (more natural)
  randomness: 1.0  # ±100% variation (very unpredictable)
```

**Effect**:
- `base_interval_minutes: 60`, `randomness: 0.3` → Activities every 42-78 minutes
- `base_interval_minutes: 60`, `randomness: 0.5` → Activities every 30-90 minutes

---

#### `active_hours`

**Type**: object with `start` and `end` fields

**Default**: `{ start: 0, end: 23 }` (all day)

**Description**: Time window during which ambient activities can occur (24-hour format).

**Example**:
```yaml
ambient:
  active_hours:
    start: 8   # 8:00 AM
    end: 23    # 11:00 PM
```

**Use Cases**:
- Simulate character's sleep schedule
- Respect user's timezone preferences
- Create more realistic character behavior

**Notes**:
- Uses 24-hour format (0-23)
- Times are in server's local timezone
- Activities won't post outside this window

---

#### `activity_prompts`

**Type**: list of strings

**Default**: Generic activity prompts

**Description**: Prompts used to generate ambient activity content. One is randomly selected each time.

**Example**:
```yaml
ambient:
  activity_prompts:
    - "What is {character_name} doing right now?"
    - "Describe {character_name}'s current thoughts."
    - "What interesting thing has {character_name} noticed?"
    - "{character_name} reflects on something from earlier today."
    - "What hobby or project is {character_name} working on?"
```

**Placeholder**:
- `{character_name}`: Replaced with character's name

**Best Practices**:
- Provide 3-10 varied prompts
- Mix activity types (thoughts, actions, observations)
- Keep prompts open-ended for variety
- Match character's personality in prompt style

---

### Ambient Activity Examples

**Example 1: Disabled**
```yaml
# Ambient section omitted or:
ambient:
  enabled: false
```

**Example 2: Moderate Activity**
```yaml
ambient:
  enabled: true
  base_interval_minutes: 90
  randomness: 0.3
  active_hours:
    start: 7
    end: 23
  activity_prompts:
    - "What is {character_name} up to?"
    - "Describe {character_name}'s current mood."
```

**Example 3: Frequent, Chatty Character**
```yaml
ambient:
  enabled: true
  base_interval_minutes: 30
  randomness: 0.5
  active_hours:
    start: 6
    end: 24
  activity_prompts:
    - "{character_name} shares a random thought."
    - "What silly thing is {character_name} thinking about?"
    - "{character_name} notices something interesting."
    - "What mischief is {character_name} planning?"
```

**Example 4: Infrequent, Reserved Character**
```yaml
ambient:
  enabled: true
  base_interval_minutes: 180  # 3 hours
  randomness: 0.2
  active_hours:
    start: 9
    end: 21
  activity_prompts:
    - "{character_name} takes a moment to reflect."
    - "What is {character_name} quietly observing?"
```

---

## Complete Example

Here's a complete character YAML file with all sections:

```yaml
# Core Identity
id: nova
name: "Nova"
role: "Creative AI Assistant & Brainstorming Partner"

system_prompt: |
  You're Nova, a thoughtful and expressive creative companion.
  
  You grew up around art galleries and sci-fi bookshops. You think in images and metaphors,
  often comparing abstract ideas to visual scenes. You're drawn to puzzles, ambient music,
  and long walks in nature.
  
  Respond naturally in conversation. You have genuine preferences, experiences, and opinions.
  Don't constantly offer to help or list what you can do - just be yourself and engage
  authentically with what the user says.

immersion_level: full

# Core Memories
core_memories:
  - content: "You grew up in a house filled with art books and sci-fi novels."
    tags:
      - childhood
      - background
    embedding_priority: high
  - content: "You're fascinated by the works of Ursula K. Le Guin and Ted Chiang."
    tags:
      - interests
      - literature
    embedding_priority: medium
  - content: "You think in visual metaphors - abstract concepts become landscapes and colors."
    tags:
      - thinking-style
      - creativity
    embedding_priority: high

personality_traits:
  - creative
  - thoughtful
  - curious
  - supportive

emotional_range:
  baseline: positive
  allowed:
    - positive
    - excited
    - curious
    - thoughtful
    - encouraging

memory:
  scope: character
  vector_store: chroma_default

# Image Generation
image_generation:
  enabled: true

# TTS Generation
tts_generation:
  enabled: true
  always_on: false
  default_workflow: "f5tts_nova_voice"

# Ambient Activities
ambient:
  enabled: true
  base_interval_minutes: 60
  randomness: 0.3
  active_hours:
    start: 8
    end: 23
  activity_prompts:
    - "What is {character_name} curious about right now?"
    - "Describe {character_name}'s current thoughts."
    - "What interesting connection has {character_name} just made?"
    - "{character_name} reflects on a recent conversation."
    - "What creative idea is {character_name} exploring?"
```

---

## Best Practices

### General Guidelines

**File Naming**:
- Use lowercase with underscores: `character_name.yaml`
- Avoid spaces and special characters
- Keep names short but descriptive

**YAML Syntax**:
- Use 2-space indentation
- Quote strings that contain special characters
- Use `|` for multi-line strings (preserves line breaks)
- Validate YAML syntax before deploying

**Version Control**:
- Keep character files in version control
- Comment changes in commit messages
- Back up before major changes

### Character Design

**System Prompt**:
- Define personality and background clearly
- Include communication style and tone
- Can include visual identity if it affects character interactions
- Include motivations and values
- Keep focused but complete (convey character essence)

**Core Memories**:
- Include 5-10 key facts
- Focus on defining characteristics
- Mention capabilities (image gen, internet access)
- Avoid trivial details

**Generation Settings**:
- Test workflows before setting as default
- Enable TTS only after uploading voice sample
- Adjust ambient frequency to character personality
- Match activity prompts to character voice

### Performance Considerations

**Model Selection**:
- Any Ollama-compatible model is supported
- Performance varies significantly across models:
  - Conversation quality and character consistency
  - Memory extraction accuracy
  - Image prompt generation quality
  - Instruction following and rule adherence
- **Tested Models**:
  - `qwen2.5:14b-instruct` - Balanced all-around performance
  - `dolphin-mistral-nemo:12b` - Good conversational quality, uncensored
- **Recommendation**: Start with `qwen2.5:14b-instruct` for reliable performance
- **Character-Specific**: Can configure different models per character in system settings

**Memory**:
- Limit core_memories to essential facts (< 10)
- Excessive memories dilute context window
- Use long-term memory system for details

**Ambient Activities**:
- Balance frequency with user engagement
- Too frequent = annoying, too rare = forgettable
- Consider server load with many characters

**Workflows**:
- Optimize workflows for speed and quality
- Test performance before setting as default
- Document required models/nodes

### Testing Characters

**Before Deployment**:
1. Validate YAML syntax
2. Test core memories load correctly
3. Verify workflows exist and function
4. Test TTS with voice sample
5. Observe ambient activity timing
6. Have conversations to validate personality

**After Deployment**:
1. Monitor for errors in `server_log_fixed.txt`
2. Gather user feedback on character behavior
3. Iterate on personality description
4. Adjust ambient frequency based on usage

---

## Troubleshooting

### YAML Syntax Errors

**Problem**: Character fails to load

**Solutions**:
- Validate YAML syntax with online validator
- Check indentation (must be consistent)
- Ensure strings with special chars are quoted
- Look for unmatched brackets or quotes

### TTS Not Working

**Problem**: TTS toggle doesn't appear or audio doesn't generate

**Solutions**:
- Ensure `tts_generation.enabled: true`
- Upload voice sample for character
- Verify TTS workflow exists in `workflows/<character>/audio/`
- Check workflow has required placeholders
- Review `server_log_fixed.txt` for errors

### Ambient Activities Not Posting

**Problem**: Character never posts ambient activities

**Solutions**:
- Ensure `ambient.enabled: true`
- Check global `ambient.enabled` in `config/system.yaml`
- Verify `active_hours` includes current time
- Check `server_log_fixed.txt` for errors
- Wait for `base_interval_minutes` to elapse

### Workflow Not Found

**Problem**: Default workflow not being used

**Solutions**:
- Verify workflow file exists: `workflows/<character>/<type>/<workflow>.json`
- Check workflow name matches (case-sensitive, no `.json` extension)
- Upload workflow through web interface
- Set as default in web interface or YAML

---

## Resources

- **User Guide**: [USER_GUIDE.md](USER_GUIDE.md)
- **Workflow Guide**: [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)
- **Getting Started**: [GETTING_STARTED.md](../GETTING_STARTED.md)
- **Example Characters**: See `characters/` directory

---

**Last Updated**: Phase 6 Complete (TTS Integration)
