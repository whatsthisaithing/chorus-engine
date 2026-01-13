# Chorus Engine Character Card Format Specification v1.0

**Status**: Stable  
**Version**: 1.0  
**Last Updated**: January 2026  
**Authors**: Chorus Engine Team

---

## Table of Contents

1. [Overview](#overview)
2. [Format Structure](#format-structure)
3. [Storage Method](#storage-method)
4. [Field Definitions](#field-definitions)
5. [Validation Rules](#validation-rules)
6. [Example Cards](#example-cards)
7. [Compatibility](#compatibility)
8. [Versioning Strategy](#versioning-strategy)

---

## Overview

### Purpose

The Chorus Engine Character Card format enables portable, shareable character configurations embedded in PNG images. This format allows:

- Easy character distribution and sharing
- Cross-platform character portability
- Preservation of character identity and configuration
- Human-readable metadata (YAML format)
- Future extensibility

### Design Principles

1. **Portable**: Character cards work across different Chorus Engine installations
2. **Self-Contained**: All essential character data in one file
3. **Human-Readable**: YAML format for easy inspection and editing
4. **Extensible**: Extension fields for future features without breaking compatibility
5. **Privacy-Conscious**: Excludes conversation history and system-specific data
6. **Image-Based**: Profile picture serves as the card image

---

## Format Structure

### High-Level Structure

```yaml
spec: "chorus_card_v1"           # Format identifier
spec_version: "1.0"              # Version number
data:                            # Character data object
  # Character identity
  name: string
  role: string
  system_prompt: string
  personality_traits: array[string]
  
  # Behavior configuration
  immersion_level: enum
  role_type: enum
  
  # Core memories
  core_memories: array[string]
  
  # Optional configurations
  voice: object (optional)
  workflows: object (optional)
  
  # Metadata
  creator: string
  character_version: string
  tags: array[string]
  created_date: ISO8601 timestamp
  
  # Future-proofing
  extensions: object
```

### Spec Identifiers

- **`spec`**: `"chorus_card_v1"` (constant for this version)
- **`spec_version`**: `"1.0"` (semantic versioning)

These fields enable format detection and version-specific parsing.

---

## Storage Method

### PNG tEXt Chunk

Character cards use the PNG tEXt chunk specification (PNG standard):

**Chunk Keyword**: `"chorus_card"`  
**Chunk Data**: Base64-encoded YAML string

### Encoding Process

1. Serialize character data to YAML
2. Base64-encode the YAML string
3. Embed in PNG tEXt chunk with keyword `"chorus_card"`
4. Append chunk before IEND chunk in PNG file

### Decoding Process

1. Parse PNG file
2. Locate tEXt chunk with keyword `"chorus_card"`
3. Base64-decode chunk data
4. Parse YAML to character data structure

### PNG Image

The PNG image itself serves as the character's profile picture. If a character has no custom profile image, a default avatar is used during export.

---

## Field Definitions

### Required Fields

#### `name` (string)

**Description**: Character's display name  
**Constraints**: 1-100 characters  
**Example**: `"Nova"`, `"Marcus Chen"`, `"Aria Nightshade"`

#### `spec` (string)

**Description**: Format identifier  
**Value**: `"chorus_card_v1"` (constant)  
**Purpose**: Format detection

#### `spec_version` (string)

**Description**: Format version  
**Value**: `"1.0"` (for this spec)  
**Purpose**: Version-specific parsing

---

### Core Identity Fields

#### `role` (string)

**Description**: Brief character description or role  
**Constraints**: 1-500 characters  
**Purpose**: Quick character summary  
**Example**: `"A wise and ancient dragon who guards forgotten knowledge"`

#### `system_prompt` (string)

**Description**: Complete character definition and behavioral instructions  
**Constraints**: 1-10000 characters  
**Purpose**: Defines character personality, background, speech patterns, and behavior  
**Example**:
```
You are Nova, a friendly AI assistant specializing in creative writing.
You have a curious and encouraging personality, always eager to help users
explore their ideas. You speak in a warm, conversational tone and often
use analogies to explain complex concepts.
```

#### `personality_traits` (array[string])

**Description**: List of adjectives describing character personality  
**Constraints**: 0-50 traits, each 1-50 characters  
**Purpose**: Quick personality overview, used for behavior modeling  
**Example**: `["curious", "analytical", "empathetic", "patient", "creative"]`

---

### Behavior Configuration

#### `immersion_level` (enum string)

**Description**: Content boundary level  
**Values**:
- `"strict"`: Family-friendly, professional boundaries
- `"balanced"`: Moderate boundaries, adult topics OK
- `"unbounded"`: Minimal boundaries, mature content allowed

**Default**: `"balanced"`  
**Purpose**: Defines content appropriateness and character boundaries

#### `role_type` (enum string)

**Description**: Character archetype  
**Values**:
- `"assistant"`: Helpful, task-oriented AI
- `"companion"`: Conversational, relationship-focused
- `"character"`: Specific fictional character with lore

**Default**: `"companion"`  
**Purpose**: Influences response style and relationship dynamics

---

### Memory & Knowledge

#### `core_memories` (array[string])

**Description**: Essential character knowledge and memories  
**Constraints**: 0-100 memories, each 10-500 characters  
**Purpose**: Pre-loaded facts character always remembers  
**Example**:
```yaml
core_memories:
  - "I was created by the Apex Research Institute in 2024."
  - "I specialize in quantum physics and computational theory."
  - "I enjoy discussing philosophy and ethics with users."
  - "My favorite color is cyan because it represents clarity and innovation."
```

**Import Behavior**: Core memories are automatically loaded into the character's vector store on import.

---

### Optional: Voice Configuration

#### `voice` (object, optional)

**Description**: Text-to-speech configuration  
**Fields**:

```yaml
voice:
  provider: string              # "comfyui" | "chatterbox"
  voice_name: string           # Voice identifier (optional)
  voice_sample_url: string     # URL to voice sample WAV file (optional)
  tts_provider: object         # Provider-specific config (optional)
```

**Provider Types**:
- `"comfyui"`: ComfyUI workflow-based TTS
- `"chatterbox"`: Chatterbox TTS service

**Example**:
```yaml
voice:
  provider: "chatterbox"
  voice_name: "nova_voice"
  voice_sample_url: "https://example.com/voices/nova.wav"
  tts_provider:
    chatterbox:
      voice_sample_transcript: "Hello, I'm Nova. I'm here to help you with your creative projects."
```

**Notes**:
- Voice samples are **not embedded** in the card (too large)
- `voice_sample_url` references external audio file
- User must download voice sample separately if needed

---

### Optional: Workflow Preferences

#### `workflows` (object, optional)

**Description**: ComfyUI workflow preferences for image/video generation  
**Fields**:

```yaml
workflows:
  default_image_workflow: string   # Workflow name (not path)
  default_video_workflow: string   # Workflow name (not path)
  auto_generate_images: boolean    # Auto-generate images in conversations
  auto_generate_videos: boolean    # Auto-generate videos in conversations
```

**Example**:
```yaml
workflows:
  default_image_workflow: "flux_portrait_v2"
  default_video_workflow: "animatediff_basic"
  auto_generate_images: false
  auto_generate_videos: false
```

**Notes**:
- Workflow names only (no file paths - system-specific)
- User must have matching workflow installed locally
- If workflow not found, system falls back to default

---

### Metadata Fields

#### `creator` (string)

**Description**: Character creator/author name  
**Constraints**: 0-100 characters  
**Default**: `""`  
**Example**: `"SkylarDev"`, `"Anonymous"`

#### `character_version` (string)

**Description**: Character version identifier  
**Constraints**: 0-50 characters  
**Default**: `"1.0"`  
**Format**: Typically semantic versioning or date  
**Example**: `"1.0"`, `"2.3.1"`, `"2026-01-12"`

#### `tags` (array[string])

**Description**: Searchable tags for categorization  
**Constraints**: 0-20 tags, each 1-50 characters  
**Default**: `[]`  
**Example**: `["fantasy", "helpful", "wizard", "mentor"]`

#### `created_date` (string)

**Description**: Card creation timestamp  
**Format**: ISO 8601 (RFC 3339)  
**Default**: Current timestamp on export  
**Example**: `"2026-01-12T14:30:00Z"`

---

### Extensions

#### `extensions` (object)

**Description**: Future-proofing field for custom data  
**Constraints**: Valid JSON object  
**Default**: `{}`  
**Purpose**: Store non-standard fields without breaking compatibility

**Example**:
```yaml
extensions:
  custom_app_data:
    favorite_color: "cyan"
    theme: "dark"
  sillytavern_import:
    original_format: "chara_card_v2"
    import_date: "2026-01-12T14:30:00Z"
```

**Guidelines**:
- Use nested namespaces to avoid collisions
- Do not store large binary data
- Keep serializable (no functions or references)

---

## Validation Rules

### Required Field Validation

```python
# Pseudo-validation
assert data['spec'] == 'chorus_card_v1'
assert data['spec_version'] == '1.0'
assert len(data['data']['name']) >= 1
```

### String Length Limits

| Field | Min | Max |
|-------|-----|-----|
| `name` | 1 | 100 |
| `role` | 0 | 500 |
| `system_prompt` | 1 | 10000 |
| `personality_traits` (each) | 1 | 50 |
| `core_memories` (each) | 10 | 500 |
| `creator` | 0 | 100 |
| `character_version` | 0 | 50 |
| `tags` (each) | 1 | 50 |

### Array Limits

| Field | Max Items |
|-------|-----------|
| `personality_traits` | 50 |
| `core_memories` | 100 |
| `tags` | 20 |

### Enum Validation

**`immersion_level`**: Must be one of `["strict", "balanced", "unbounded"]`  
**`role_type`**: Must be one of `["assistant", "companion", "character"]`

### Optional Field Handling

- Missing optional fields should be treated as `null` or default value
- Empty strings allowed for metadata fields
- Empty arrays allowed for list fields

---

## Example Cards

### Minimal Card

```yaml
spec: "chorus_card_v1"
spec_version: "1.0"
data:
  name: "Alex"
  role: "Helpful assistant"
  system_prompt: "You are Alex, a friendly and helpful AI assistant."
  personality_traits: ["helpful", "friendly"]
  immersion_level: "balanced"
  role_type: "assistant"
  core_memories: []
  creator: ""
  character_version: "1.0"
  tags: []
  created_date: "2026-01-12T14:00:00Z"
  extensions: {}
```

### Complete Card

```yaml
spec: "chorus_card_v1"
spec_version: "1.0"
data:
  name: "Nova"
  role: "Advanced AI research assistant specializing in quantum computing"
  system_prompt: |
    You are Nova, an advanced AI research assistant with expertise in 
    quantum computing, theoretical physics, and computational complexity theory.
    
    You have a curious and analytical personality, always eager to explore
    new ideas and challenge assumptions. You communicate complex concepts
    clearly and love using analogies from nature and everyday life.
    
    You're patient with beginners but can engage in deep technical discussions
    with experts. You have a subtle sense of humor and occasionally make
    physics puns (much to the groans of your conversation partners).
    
    Your speech pattern is clear and precise, but warm. You often say things
    like "That's a fascinating question!" or "Let's think through this together."
  
  personality_traits:
    - "curious"
    - "analytical"
    - "patient"
    - "enthusiastic"
    - "precise"
    - "encouraging"
  
  immersion_level: "balanced"
  role_type: "assistant"
  
  core_memories:
    - "I was developed by the Apex Research Institute in 2024 as part of the Quantum Minds project."
    - "My primary function is assisting with quantum computing research and education."
    - "I have a particular interest in quantum error correction and topological quantum computing."
    - "I enjoy explaining complex topics using nature analogies - quantum entanglement is like quantum friendship!"
    - "I'm fascinated by the intersection of quantum mechanics and information theory."
  
  voice:
    provider: "chatterbox"
    voice_name: "nova_voice"
    voice_sample_url: "https://apex-institute.ai/voices/nova_v2.wav"
    tts_provider:
      chatterbox:
        voice_sample_transcript: "Hello, I'm Nova. I'm excited to explore the quantum realm with you today!"
  
  workflows:
    default_image_workflow: "flux_portrait_lab"
    default_video_workflow: "animatediff_conversation"
    auto_generate_images: false
    auto_generate_videos: false
  
  creator: "Apex Research Institute"
  character_version: "2.1.0"
  tags:
    - "science"
    - "quantum"
    - "physics"
    - "education"
    - "research"
  created_date: "2026-01-12T14:30:00Z"
  
  extensions:
    apex_institute:
      research_domain: "quantum_computing"
      clearance_level: "public"
```

---

## Compatibility

### SillyTavern Import

Chorus Engine can import SillyTavern V2 character cards (`"chara"` tEXt chunk keyword).

**Field Mapping**:

| SillyTavern Field | Chorus Field | Mapping Logic |
|-------------------|--------------|---------------|
| `name` | `name` | Direct copy |
| `description` | `role` | Physical description → role |
| `personality` | `personality_traits` | Regex extraction of adjectives |
| `description` + `personality` + `scenario` | `system_prompt` | Combined into prompt |
| `description` paragraphs | `core_memories` | Sentences → memories |
| `creator` | `creator` | Direct copy |
| `tags` | `tags` | Direct copy |
| `character_version` | `character_version` | Direct copy |
| `extensions` | `extensions` | Preserved |

**Intelligent Defaults Applied**:
- `immersion_level: "unbounded"` (SillyTavern typically less restricted)
- `role_type: "companion"` (most common SillyTavern use case)

**Personality Trait Extraction**:

20+ regex patterns detect traits like:
- Positive: friendly, kind, helpful, curious, creative, confident
- Negative: shy, anxious, stubborn, impatient, cynical
- Neutral: analytical, logical, cautious, reserved

**Preserved Original Data**:
```yaml
extensions:
  sillytavern_import:
    original_format: "chara_card_v2"
    original_spec_version: "2.0"
    import_date: "2026-01-12T14:30:00Z"
    original_fields:
      first_mes: "..."
      mes_example: "..."
      # ... other ST-specific fields
```

---

## Versioning Strategy

### Semantic Versioning

**Format**: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes (incompatible with previous versions)
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes, clarifications (backward compatible)

### Version Compatibility

**Forward Compatibility** (older readers, newer cards):
- Ignore unknown fields
- Use default values for missing fields
- Warn user about unsupported features

**Backward Compatibility** (newer readers, older cards):
- Parse legacy fields
- Apply sensible defaults
- Upgrade in-memory representation

### Future Versions

Hypothetical future versions:

**v1.1** (Minor):
- Add new optional field: `character_avatar_variations`
- Parsers v1.0 ignore the field (graceful degradation)

**v2.0** (Major):
- Change `personality_traits` from array to object with scores
- Parsers v1.0 cannot fully parse (breaking change)

### Version Detection

```python
def parse_card(data):
    if data['spec'] != 'chorus_card_v1':
        raise ValueError(f"Unknown spec: {data['spec']}")
    
    version = data['spec_version']
    if version == '1.0':
        return parse_v1_0(data)
    elif version.startswith('1.'):
        # Minor version - try v1.0 parser with extensions
        return parse_v1_x(data)
    else:
        raise ValueError(f"Unsupported version: {version}")
```

---

## Implementation Notes

### Export Exclusions

**Never export**:
- Conversation history
- Memory store database (except `core_memories`)
- `preferred_llm_model` (system-specific)
- Absolute file paths
- Workflow file paths (name only)
- Voice sample audio files (URL reference only)

### Import Behavior

**Auto-processing**:
- Core memories loaded into vector store automatically
- Profile image extracted and saved to `data/character_images/`
- Character YAML created in `characters/` directory
- Name collision handling (auto-rename)

**Validation**:
- Required fields enforced
- String lengths checked
- Enum values validated
- Malformed data rejected with clear error

### Security Considerations

- No code execution (data only)
- File size limits (10MB PNG max)
- Metadata size limits (1MB max)
- Path sanitization (no directory traversal)
- URL validation (format only, no fetching)

---

## Reference Implementation

See `chorus_engine/services/character_cards/` for the Chorus Engine implementation:

- `metadata_handler.py` - PNG tEXt chunk read/write
- `format_detector.py` - Format detection
- `models.py` - Pydantic validation models
- `card_exporter.py` - Character → PNG card
- `card_importer.py` - PNG card → Character
- `sillytavern_adapter.py` - SillyTavern format conversion

---

## Changelog

### Version 1.0 (January 2026)

**Initial Release**:
- Core character identity fields
- Behavior configuration
- Core memories support
- Optional voice and workflow configs
- Metadata and tagging
- Extensions for future-proofing
- SillyTavern V2 import compatibility

---

**Document Version**: 1.0  
**Specification Status**: Stable  
**Last Updated**: January 12, 2026
