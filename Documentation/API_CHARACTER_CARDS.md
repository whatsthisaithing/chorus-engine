# Character Cards API Documentation

This document describes the API endpoints for character card import/export functionality.

---

## Table of Contents

1. [Overview](#overview)
2. [Export Character Card](#export-character-card)
3. [Import Card Preview](#import-card-preview)
4. [Import Card Confirm](#import-card-confirm)
5. [Upload Profile Image](#upload-profile-image)
6. [Error Codes](#error-codes)
7. [Data Models](#data-models)

---

## Overview

Character cards are PNG images with embedded metadata (base64-encoded YAML/JSON in PNG tEXt chunks). The API supports:

- **Chorus Engine Format**: Native format with `"chorus_card"` tEXt chunk keyword
- **SillyTavern V2 Format**: Import compatibility with `"chara"` tEXt chunk keyword

**Base URL**: `http://localhost:8000`

---

## Export Character Card

Export a character as a PNG card with embedded metadata.

### Endpoint

```
POST /api/characters/export-card
```

### Request Body

```json
{
  "character_id": "nova",
  "character_name": "Nova",
  "include_voice": true,
  "include_workflows": true,
  "voice_sample_url": "https://example.com/voice.wav"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `character_id` | string | Yes | Character ID (filename without .yaml) |
| `character_name` | string | Yes | Display name of character |
| `include_voice` | boolean | No | Include voice configuration in export (default: true) |
| `include_workflows` | boolean | No | Include workflow preferences (default: true) |
| `voice_sample_url` | string | No | URL to voice sample audio file |

### Response

**Success (200)**:
- **Content-Type**: `image/png`
- **Content-Disposition**: `attachment; filename="Nova.card.png"`
- **Body**: PNG file data with embedded metadata

**Error (400)**:
```json
{
  "detail": "Character not found: invalid_character"
}
```

**Error (404)**:
```json
{
  "detail": "Character file not found: nova.yaml"
}
```

### Example (curl)

```bash
curl -X POST http://localhost:8000/api/characters/export-card \
  -H "Content-Type: application/json" \
  -d '{
    "character_id": "nova",
    "character_name": "Nova",
    "include_voice": true,
    "include_workflows": true
  }' \
  --output Nova.card.png
```

### Example (JavaScript)

```javascript
const response = await fetch('/api/characters/export-card', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    character_id: 'nova',
    character_name: 'Nova',
    include_voice: true,
    include_workflows: true
  })
});

const blob = await response.blob();
const url = window.URL.createObjectURL(blob);
const a = document.createElement('a');
a.href = url;
a.download = 'Nova.card.png';
a.click();
```

---

## Import Card Preview

Preview a character card before importing (does not save).

### Endpoint

```
POST /api/characters/import-card/preview
```

### Request

**Content-Type**: `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | PNG file containing character card |

### Response

**Success (200)**:
```json
{
  "format": "chorus_card_v1",
  "character_data": {
    "name": "Nova",
    "role": "AI Assistant",
    "system_prompt": "You are Nova, a friendly AI assistant...",
    "personality_traits": ["helpful", "curious", "analytical"],
    "immersion_level": "balanced",
    "role_type": "assistant",
    "core_memories": [
      "I was created to help users with various tasks.",
      "I value clear communication and accuracy."
    ]
  },
  "profile_image": "data:image/png;base64,iVBORw0KG...",
  "warnings": [],
  "sillytavern_import": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `format` | string | Detected format (`chorus_card_v1`, `chara_card_v2`, etc.) |
| `character_data` | object | Parsed character configuration |
| `profile_image` | string | Base64-encoded profile image (data URL) |
| `warnings` | array | Any issues encountered during parsing |
| `sillytavern_import` | boolean | True if imported from SillyTavern format |

**Error (400)**:
```json
{
  "detail": "No character card data found in image"
}
```

```json
{
  "detail": "Invalid PNG file"
}
```

```json
{
  "detail": "File must be an image"
}
```

### Example (curl)

```bash
curl -X POST http://localhost:8000/api/characters/import-card/preview \
  -F "file=@Nova.card.png"
```

### Example (JavaScript)

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('/api/characters/import-card/preview', {
  method: 'POST',
  body: formData
});

const preview = await response.json();
console.log('Character:', preview.character_data.name);
console.log('Format:', preview.format);
```

---

## Import Card Confirm

Confirm and save an imported character card.

### Endpoint

```
POST /api/characters/import-card/confirm
```

### Request Body

```json
{
  "preview_data": {
    "format": "chorus_card_v1",
    "character_data": { /* character data from preview */ },
    "profile_image": "data:image/png;base64,..."
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `preview_data` | object | Yes | Complete preview data from preview endpoint |

### Response

**Success (200)**:
```json
{
  "success": true,
  "character_filename": "nova.yaml",
  "profile_image": "nova.png",
  "core_memories_loaded": 5
}
```

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Import success status |
| `character_filename` | string | Saved character YAML filename |
| `profile_image` | string | Saved profile image filename |
| `core_memories_loaded` | integer | Number of core memories loaded into vector store |

**Error (400)**:
```json
{
  "detail": "Character name already exists: nova"
}
```

```json
{
  "detail": "Invalid character data: missing required field 'name'"
}
```

### Example (JavaScript)

```javascript
const response = await fetch('/api/characters/import-card/confirm', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    preview_data: previewData  // From preview endpoint
  })
});

const result = await response.json();
if (result.success) {
  console.log('Character imported:', result.character_filename);
  console.log('Core memories loaded:', result.core_memories_loaded);
}
```

---

## Upload Profile Image

Upload a custom profile image for a character.

### Endpoint

```
POST /api/characters/{character_id}/upload-profile-image
```

### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `character_id` | string | Yes | Character ID (filename without .yaml) |

### Request

**Content-Type**: `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Image file (PNG, JPG, JPEG, WEBP) |

### Response

**Success (200)**:
```json
{
  "success": true,
  "filename": "nova.png",
  "image_path": "data/character_images/nova.png"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Upload success status |
| `filename` | string | Saved image filename |
| `image_path` | string | Full path to saved image |

**Error (400)**:
```json
{
  "detail": "File must be an image"
}
```

**Error (404)**:
```json
{
  "detail": "Character not found: invalid_character"
}
```

**Error (500)**:
```json
{
  "detail": "Failed to upload profile image: <error details>"
}
```

### Example (curl)

```bash
curl -X POST http://localhost:8000/api/characters/nova/upload-profile-image \
  -F "file=@profile.png"
```

### Example (JavaScript)

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch(`/api/characters/${characterId}/upload-profile-image`, {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log('Profile image saved:', result.filename);
```

---

## Error Codes

### HTTP Status Codes

| Code | Description | Common Causes |
|------|-------------|---------------|
| 200 | Success | Request completed successfully |
| 400 | Bad Request | Invalid input, missing required fields, invalid file format |
| 404 | Not Found | Character not found, file not found |
| 500 | Internal Server Error | Server-side error during processing |

### Common Error Messages

#### Export Errors

- **"Character not found: {id}"**: Character does not exist in memory
- **"Character file not found: {filename}"**: Character YAML file missing
- **"No profile image found for character"**: Character has no profile image (will use default)

#### Import Errors

- **"No character card data found in image"**: PNG has no metadata or wrong chunk keyword
- **"Invalid PNG file"**: File is not a valid PNG format
- **"File must be an image"**: Uploaded file is not an image type
- **"Character name already exists"**: Name collision (will auto-rename)
- **"Failed to parse metadata"**: Corrupted or invalid metadata
- **"Missing required field: {field}"**: Character data incomplete

#### Upload Errors

- **"File must be an image"**: Wrong file type
- **"Character not found"**: Invalid character ID
- **"Failed to upload profile image"**: File write error or image processing error

---

## Data Models

### Chorus Character Card Format

```yaml
spec: "chorus_card_v1"
spec_version: "1.0"
data:
  # Character Identity
  name: string
  role: string
  system_prompt: string
  personality_traits: array[string]
  
  # Behavior Configuration
  immersion_level: string  # "strict" | "balanced" | "unbounded"
  role_type: string        # "assistant" | "companion" | "character"
  
  # Core Memories
  core_memories: array[string]
  
  # Optional: Voice Configuration
  voice:
    provider: string       # "comfyui" | "chatterbox"
    voice_name: string
    voice_sample_url: string
  
  # Optional: Workflow Preferences
  workflows:
    default_image_workflow: string
    default_video_workflow: string
  
  # Metadata
  creator: string
  character_version: string
  tags: array[string]
  created_date: ISO8601 timestamp
  
  # Extensions (future-proofing)
  extensions: object
```

### SillyTavern V2 Format (Import Only)

```json
{
  "spec": "chara_card_v2",
  "spec_version": "2.0",
  "data": {
    "name": "string",
    "description": "string",
    "personality": "string",
    "scenario": "string",
    "first_mes": "string",
    "mes_example": "string",
    "creator": "string",
    "character_version": "string",
    "tags": ["string"],
    "creator_notes": "string",
    "system_prompt": "string",
    "post_history_instructions": "string",
    "alternate_greetings": ["string"],
    "character_book": { /* lorebook data */ },
    "extensions": {}
  }
}
```

**Field Mapping (SillyTavern → Chorus)**:

| SillyTavern | Chorus Engine | Notes |
|-------------|---------------|-------|
| `name` | `name` | Direct mapping |
| `description` | `role` | Physical description → role field |
| `personality` | `personality_traits` | Extracted via regex |
| `description` + `personality` | `system_prompt` | Combined |
| `scenario` | `system_prompt` | Appended to system prompt |
| `creator` | Metadata | Preserved |
| `tags` | Metadata | Preserved |
| `character_version` | Metadata | Preserved |
| `extensions` | `extensions` | Preserved for future use |

**Intelligent Defaults Applied**:
- `immersion_level: "unbounded"` (SillyTavern characters typically less restricted)
- `role_type: "companion"` (most common use case)
- Description paragraphs → `core_memories` (extractable knowledge)

### Profile Image Focus Point

```yaml
profile_image_focus:
  x: 52.5  # Horizontal percentage (0-100)
  y: 25.0  # Vertical percentage (0-100)
```

Used for CSS `object-position` to center important parts of non-square images in circular avatars.

---

## Integration Examples

### Complete Import Flow

```javascript
// 1. Preview card
async function previewCard(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('/api/characters/import-card/preview', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
}

// 2. Show preview to user (UI code)

// 3. Confirm import
async function confirmImport(previewData) {
  const response = await fetch('/api/characters/import-card/confirm', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ preview_data: previewData })
  });
  
  return await response.json();
}

// Usage
const preview = await previewCard(fileInput.files[0]);
// ... show preview UI ...
if (userConfirms) {
  const result = await confirmImport(preview);
  console.log('Imported:', result.character_filename);
  // Reload character list
}
```

### Complete Export Flow

```javascript
async function exportCharacter(characterId, characterName, options = {}) {
  const response = await fetch('/api/characters/export-card', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      character_id: characterId,
      character_name: characterName,
      include_voice: options.includeVoice ?? true,
      include_workflows: options.includeWorkflows ?? true,
      voice_sample_url: options.voiceSampleUrl
    })
  });
  
  // Download file
  const blob = await response.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${characterName}.card.png`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(url);
}

// Usage
await exportCharacter('nova', 'Nova', {
  includeVoice: true,
  includeWorkflows: true
});
```

---

## Security Considerations

### File Upload Limits

- **Max file size**: 10MB
- **Allowed formats**: PNG, JPG, JPEG, WEBP (for profile images)
- **Card format**: PNG only (for import)

### Metadata Validation

- All imported data validated against Pydantic schemas
- Required fields enforced
- Character names sanitized (alphanumeric, underscores, hyphens only)
- File paths validated (no directory traversal)

### Privacy

- Exported cards exclude:
  - Conversation history
  - Memory store (except core memories)
  - System-specific file paths
  - LLM model preferences (may be system-specific)

### URL Validation

- `voice_sample_url` validated as proper URL format
- No URL fetching/execution during import (user responsibility)

---

## Testing

### Test with curl

```bash
# Export
curl -X POST http://localhost:8000/api/characters/export-card \
  -H "Content-Type: application/json" \
  -d '{"character_id":"nova","character_name":"Nova"}' \
  --output test.png

# Preview
curl -X POST http://localhost:8000/api/characters/import-card/preview \
  -F "file=@test.png"

# Upload profile image
curl -X POST http://localhost:8000/api/characters/nova/upload-profile-image \
  -F "file=@profile.png"
```

### Verify PNG Metadata (Python)

```python
from PIL import Image
import base64
import yaml

# Read PNG
img = Image.open('Nova.card.png')

# Extract tEXt chunk
if 'chorus_card' in img.info:
    metadata = img.info['chorus_card']
    decoded = base64.b64decode(metadata)
    data = yaml.safe_load(decoded)
    print(data)
```

---

**Last Updated**: January 2026 (V1 Character Card System Complete)
