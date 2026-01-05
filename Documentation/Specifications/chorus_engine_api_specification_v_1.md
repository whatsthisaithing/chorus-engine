# Chorus Engine – API Specification (v1)

This document defines the complete HTTP and WebSocket API for Chorus Engine v1.

The API is designed for:
- Local-first operation
- Real-time streaming where possible
- Explicit state management
- Single-user access patterns
- Seamless mode switching (text ↔ voice)

---

## Base Architecture

**Protocol**: HTTP/1.1 + WebSocket
**Base URL**: `http://localhost:8080` (configurable)
**Content-Type**: `application/json` (unless otherwise specified)
**ID Format**: UUID v4 for all resources

---

## Authentication & Security

### v1 Approach
- No authentication required by default
- Single-user, local-only access
- Optional "lock" feature via encrypted preference

### Lock Feature (Optional)
```http
POST /api/system/lock
{
  "code": "user-defined-pin"
}

POST /api/system/unlock
{
  "code": "user-defined-pin"
}

GET /api/system/lock-status
Response: { "locked": boolean }
```

Lock state stored in encrypted user preferences.

---

## Core Endpoints Overview

### Characters
- `GET /api/characters` - List all characters
- `GET /api/characters/{id}` - Get character details
- `POST /api/characters` - Create character
- `PUT /api/characters/{id}` - Update character
- `DELETE /api/characters/{id}` - Delete character
- `POST /api/characters/{id}/validate` - Validate character config

### Conversations
- `GET /api/conversations` - List conversations (all or by character)
- `GET /api/conversations/{id}` - Get conversation details
- `POST /api/conversations` - Create conversation
- `PUT /api/conversations/{id}` - Update conversation metadata
- `DELETE /api/conversations/{id}` - Delete conversation
- `POST /api/conversations/{id}/archive` - Archive conversation

### Threads
- `GET /api/conversations/{conv_id}/threads` - List threads
- `GET /api/threads/{id}` - Get thread details
- `POST /api/conversations/{conv_id}/threads` - Create side thread
- `DELETE /api/threads/{id}` - Delete thread
- `POST /api/threads/{id}/clear` - Clear thread messages

### Chat (Text)
- `POST /api/chat/text` - Send text message (blocking, returns complete response)
- `POST /threads/{thread_id}/messages/stream` - Send with streaming response (SSE)
- `WebSocket /api/chat/text/stream` - Real-time text streaming (planned for Phase 3)

### Chat (Voice)
- `POST /api/chat/voice` - Send voice message (returns stream URL)
- `WebSocket /api/chat/voice/stream` - Real-time voice streaming

### Memory
- `GET /api/memory` - Query memory entries
- `GET /api/memory/{id}` - Get specific memory entry
- `POST /api/memory` - Create explicit memory entry
- `PUT /api/memory/{id}` - Update memory entry
- `DELETE /api/memory/{id}` - Delete memory entry
- `POST /api/memory/search` - Semantic search memory

### Visual Generation
- `POST /api/visual/generate` - Request image generation
- `GET /api/visual/jobs/{id}` - Get job status
- `GET /api/visual/jobs/{id}/result` - Get generated image
- `DELETE /api/visual/jobs/{id}` - Cancel job

### Ambient Activity
- `GET /api/conversations/{id}/activity` - Get current activity
- `POST /api/conversations/{id}/activity` - Set activity
- `DELETE /api/conversations/{id}/activity` - Clear activity
- `POST /api/conversations/{id}/activity/propose` - Character-propose activity

### System
- `GET /api/system/health` - Health check
- `GET /api/system/status` - System status (models loaded, ComfyUI connected, etc.)
- `GET /api/system/config` - Get system configuration
- `PUT /api/system/config` - Update system configuration

---

## Data Models

### Character

```json
{
  "id": "uuid",
  "name": "Nova",
  "role": "creative_companion",
  "system_prompt": "string",
  "personality_traits": ["expressive", "imaginative"],
  "preferred_llm": {
    "provider": "local",
    "model": "mixtral-8x7b-instruct"
  },
  "voice": {
    "tts_engine": "xtts_v2",
    "voice_sample": "samples/nova.wav",
    "speaking_style": {
      "pace": "medium",
      "tone": "warm",
      "expressiveness": "high"
    }
  },
  "emotional_range": {
    "baseline": "positive",
    "allowed": ["positive", "thoughtful", "calm"]
  },
  "memory": {
    "scope": "character",
    "vector_store": "chroma_default"
  },
  "ambient_activity": {
    "enabled": true,
    "default_behavior": "propose_if_idle",
    "style_guidelines": "string"
  },
  "visual_identity": {
    "default_workflow": "flux_dev_portrait",
    "reference_images": ["path/to/image.png"],
    "loras": [
      {
        "name": "nova_style",
        "path": "loras/nova_style.safetensors",
        "weight": 0.8,
        "trigger_words": ["nova-style", "soft painterly"]
      }
    ]
  },
  "created_at": "2025-12-27T10:00:00Z",
  "updated_at": "2025-12-27T10:00:00Z"
}
```

### Conversation

```json
{
  "id": "uuid",
  "character_id": "uuid",
  "title": "Main Conversation",
  "type": "main",
  "created_at": "2025-12-27T10:00:00Z",
  "updated_at": "2025-12-27T10:00:00Z",
  "last_message_at": "2025-12-27T12:30:00Z",
  "message_count": 42,
  "is_archived": false,
  "metadata": {
    "tags": ["creative", "project-x"],
    "custom_fields": {}
  }
}
```

### Thread

```json
{
  "id": "uuid",
  "conversation_id": "uuid",
  "type": "main" | "side",
  "title": "Side Thread: Vacation Planning",
  "created_at": "2025-12-27T10:00:00Z",
  "updated_at": "2025-12-27T10:00:00Z",
  "message_count": 15,
  "metadata": {}
}
```

### Message

```json
{
  "id": "uuid",
  "thread_id": "uuid",
  "role": "user" | "assistant" | "system",
  "content": "string",
  "content_type": "text" | "voice",
  "timestamp": "2025-12-27T12:30:00Z",
  "metadata": {
    "voice_input_transcript": "string (if voice)",
    "audio_file": "path/to/audio.wav (if voice)",
    "generation_params": {},
    "attached_images": ["uuid"]
  }
}
```

### Memory Entry

```json
{
  "id": "uuid",
  "content": "User's anniversary is October 15th",
  "type": "explicit" | "implicit" | "ephemeral",
  "scope": "global" | "character" | "thread",
  "character_id": "uuid (if character scope)",
  "thread_id": "uuid (if thread scope)",
  "confidence": 1.0,
  "source": "user_explicit" | "inferred" | "system",
  "created_at": "2025-12-27T10:00:00Z",
  "updated_at": "2025-12-27T10:00:00Z",
  "embedding": [0.1, 0.2, ...],
  "metadata": {
    "tags": ["personal", "important"],
    "user_confirmed": true
  }
}
```

### Ambient Activity

```json
{
  "id": "uuid",
  "conversation_id": "uuid",
  "description": "Painting a portrait in a quiet studio",
  "scope": "thread" | "character",
  "mood": "calm",
  "started_at": "2025-12-27T10:00:00Z",
  "user_confirmed": false,
  "proposed_by": "character" | "user",
  "metadata": {}
}
```

### Visual Generation Job

```json
{
  "id": "uuid",
  "conversation_id": "uuid",
  "thread_id": "uuid",
  "message_id": "uuid (optional - associates with message)",
  "status": "queued" | "processing" | "completed" | "failed" | "cancelled",
  "workflow": "flux_dev_portrait",
  "prompt": "A serene portrait in soft painterly style",
  "parameters": {
    "steps": 30,
    "guidance_scale": 4.5,
    "seed": 42
  },
  "created_at": "2025-12-27T10:00:00Z",
  "started_at": "2025-12-27T10:00:30Z",
  "completed_at": "2025-12-27T10:03:45Z",
  "result": {
    "image_path": "/absolute/path/to/output.png",
    "chorus_path": "data/conversations/{conv_id}/images/{job_id}.png",
    "metadata": {
      "comfy_prompt_id": "string",
      "generation_time_seconds": 195
    }
  },
  "error": {
    "message": "ComfyUI connection failed",
    "code": "COMFY_UNAVAILABLE",
    "details": {}
  }
}
```

---

## Character Endpoints

### List Characters

```http
GET /api/characters
```

**Response**
```json
{
  "characters": [
    {
      "id": "uuid",
      "name": "Nova",
      "role": "creative_companion",
      "created_at": "2025-12-27T10:00:00Z"
    }
  ],
  "total": 2
}
```

### Get Character

```http
GET /api/characters/{id}
```

**Response**: Full Character object (see Data Models)

### Create Character

```http
POST /api/characters
Content-Type: application/json

{
  "name": "Aurora",
  "role": "technical_assistant",
  "system_prompt": "You are Aurora...",
  "personality_traits": ["analytical", "precise"],
  ...
}
```

**Response**
```json
{
  "character": { /* Full Character object */ },
  "validation_warnings": [
    "Voice sample file not found, will use default"
  ]
}
```

### Update Character

```http
PUT /api/characters/{id}
Content-Type: application/json

{
  "system_prompt": "Updated prompt...",
  "voice": { ... }
}
```

Partial updates supported. Only provided fields are updated.

**Response**: Full updated Character object

### Delete Character

```http
DELETE /api/characters/{id}
```

**Query Parameters**
- `delete_conversations`: boolean (default: false) - Also delete all conversations
- `delete_memory`: boolean (default: false) - Also delete character-scoped memory

**Response**
```json
{
  "success": true,
  "deleted_conversations": 3,
  "deleted_memory_entries": 42
}
```

### Validate Character Config

```http
POST /api/characters/{id}/validate
```

Validates character configuration without saving.

**Response**
```json
{
  "valid": true,
  "errors": [],
  "warnings": [
    "LoRA file not found: loras/custom.safetensors",
    "Workflow 'custom_workflow' not found in workflows/"
  ]
}
```

---

## Conversation Endpoints

### List Conversations

```http
GET /api/conversations
```

**Query Parameters**
- `character_id`: uuid (filter by character)
- `archived`: boolean (default: false)
- `limit`: int (default: 50)
- `offset`: int (default: 0)

**Response**
```json
{
  "conversations": [ /* Conversation objects */ ],
  "total": 10,
  "limit": 50,
  "offset": 0
}
```

### Get Conversation

```http
GET /api/conversations/{id}
```

**Response**
```json
{
  "conversation": { /* Full Conversation object */ },
  "threads": [
    {
      "id": "uuid",
      "type": "main",
      "message_count": 42
    }
  ],
  "current_activity": { /* Activity object or null */ }
}
```

### Create Conversation

```http
POST /api/conversations
Content-Type: application/json

{
  "character_id": "uuid",
  "title": "New Discussion",
  "metadata": {}
}
```

**Response**: Full Conversation object with auto-created main thread

### Update Conversation

```http
PUT /api/conversations/{id}
Content-Type: application/json

{
  "title": "Updated Title",
  "metadata": {
    "tags": ["updated", "important"]
  }
}
```

**Response**: Updated Conversation object

### Delete Conversation

```http
DELETE /api/conversations/{id}
```

**Query Parameters**
- `delete_memory`: boolean (default: false) - Delete thread-scoped memory

**Response**
```json
{
  "success": true,
  "deleted_threads": 3,
  "deleted_messages": 127,
  "deleted_memory_entries": 5
}
```

### Archive Conversation

```http
POST /api/conversations/{id}/archive
```

**Response**: Updated Conversation object with `is_archived: true`

---

## Thread Endpoints

### List Threads

```http
GET /api/conversations/{conv_id}/threads
```

**Response**
```json
{
  "threads": [ /* Thread objects */ ],
  "main_thread": { /* Main thread object */ }
}
```

### Get Thread

```http
GET /api/threads/{id}
```

**Query Parameters**
- `include_messages`: boolean (default: true)
- `message_limit`: int (default: 50)
- `message_offset`: int (default: 0)

**Response**
```json
{
  "thread": { /* Thread object */ },
  "messages": [ /* Message objects */ ]
}
```

### Create Side Thread

```http
POST /api/conversations/{conv_id}/threads
Content-Type: application/json

{
  "title": "Vacation Planning",
  "metadata": {}
}
```

**Response**: New Thread object (type: "side")

### Delete Thread

```http
DELETE /api/threads/{id}
```

**Error**: Cannot delete main thread (400)

**Response**
```json
{
  "success": true,
  "deleted_messages": 15
}
```

### Clear Thread

```http
POST /api/threads/{id}/clear
```

Removes all messages but keeps thread.

**Response**: Empty Thread object

---

## Chat Endpoints (Text)

### Send Text Message

```http
POST /api/chat/text
Content-Type: application/json

{
  "thread_id": "uuid",
  "content": "Hello, how are you today?",
  "stream": true,
  "metadata": {
    "client_timestamp": "2025-12-27T12:30:00Z"
  }
}
```

**Response** (when stream=false)
```json
{
  "message_id": "uuid",
  "response_message_id": "uuid",
  "content": "I'm doing well, thank you for asking!",
  "metadata": {
    "generation_time_ms": 1250,
    "model_used": "mixtral-8x7b-instruct",
    "tokens_generated": 12
  }
}
```

**Response** (when stream=true)
```json
{
  "message_id": "uuid",
  "stream_url": "ws://localhost:8080/api/chat/text/stream/{session_id}"
}
```

### Text Streaming WebSocket

```
WebSocket ws://localhost:8080/api/chat/text/stream/{session_id}
```

**Client → Server** (Handshake)
```json
{
  "type": "init",
  "thread_id": "uuid",
  "message_id": "uuid"
}
```

**Server → Client** (Token Stream)
```json
{
  "type": "token",
  "content": "I'm",
  "is_final": false
}

{
  "type": "token",
  "content": " doing",
  "is_final": false
}

...

{
  "type": "complete",
  "message_id": "uuid",
  "full_content": "I'm doing well, thank you for asking!",
  "metadata": {
    "generation_time_ms": 1250,
    "tokens_generated": 12
  }
}
```

**Server → Client** (Error)
```json
{
  "type": "error",
  "error": {
    "code": "LLM_UNAVAILABLE",
    "message": "Local LLM connection failed",
    "recoverable": false
  }
}
```

---

## Chat Endpoints (Voice)

### Send Voice Message

```http
POST /api/chat/voice
Content-Type: multipart/form-data

thread_id: uuid
audio: (binary audio file)
format: wav | mp3 | ogg (optional, default: wav)
stream: true
return_transcript: true
return_audio: true
```

**Response** (when stream=false)
```json
{
  "message_id": "uuid",
  "input_transcript": "Hello how are you today",
  "response_message_id": "uuid",
  "response_transcript": "I'm doing well, thank you for asking!",
  "response_audio_path": "data/conversations/{conv_id}/audio/{msg_id}.wav",
  "metadata": {
    "stt_time_ms": 450,
    "generation_time_ms": 1250,
    "tts_time_ms": 800,
    "total_time_ms": 2500
  }
}
```

**Response** (when stream=true)
```json
{
  "message_id": "uuid",
  "input_transcript": "Hello how are you today",
  "stream_url": "ws://localhost:8080/api/chat/voice/stream/{session_id}"
}
```

### Voice Streaming WebSocket

```
WebSocket ws://localhost:8080/api/chat/voice/stream/{session_id}
```

**Server → Client** (Text Token Stream)
```json
{
  "type": "transcript_token",
  "content": "I'm",
  "is_final": false
}
```

**Server → Client** (Complete Text)
```json
{
  "type": "transcript_complete",
  "full_transcript": "I'm doing well, thank you for asking!",
  "message_id": "uuid"
}
```

**Server → Client** (Audio Chunks) - Future Enhancement
```json
{
  "type": "audio_chunk",
  "data": "base64_encoded_audio",
  "chunk_index": 0,
  "is_final": false
}
```

**Server → Client** (Complete)
```json
{
  "type": "complete",
  "response_audio_path": "data/conversations/{conv_id}/audio/{msg_id}.wav",
  "metadata": {
    "generation_time_ms": 1250,
    "tts_time_ms": 800
  }
}
```

**Notes**
- Voice responses always generate and store both transcript and audio
- Transcript is immediately available via streaming
- Audio file path provided when TTS completes
- Future enhancement: stream audio chunks for faster playback

---

## Memory Endpoints

### Query Memory

```http
GET /api/memory
```

**Query Parameters**
- `scope`: global | character | thread
- `character_id`: uuid (required if scope=character)
- `thread_id`: uuid (required if scope=thread)
- `type`: explicit | implicit | ephemeral
- `limit`: int (default: 50)
- `offset`: int (default: 0)

**Response**
```json
{
  "memories": [ /* Memory Entry objects */ ],
  "total": 42,
  "limit": 50,
  "offset": 0
}
```

### Get Memory Entry

```http
GET /api/memory/{id}
```

**Response**: Full Memory Entry object

### Create Memory Entry

```http
POST /api/memory
Content-Type: application/json

{
  "content": "User's anniversary is October 15th",
  "type": "explicit",
  "scope": "global",
  "metadata": {
    "tags": ["personal", "dates"],
    "user_confirmed": true
  }
}
```

**Response**: Created Memory Entry object with generated embedding

### Update Memory Entry

```http
PUT /api/memory/{id}
Content-Type: application/json

{
  "content": "Updated content...",
  "metadata": {
    "tags": ["updated"]
  }
}
```

**Response**: Updated Memory Entry object with re-generated embedding

### Delete Memory Entry

```http
DELETE /api/memory/{id}
```

**Response**
```json
{
  "success": true
}
```

### Search Memory

```http
POST /api/memory/search
Content-Type: application/json

{
  "query": "when is the user's anniversary?",
  "scope": "global",
  "character_id": "uuid (optional)",
  "thread_id": "uuid (optional)",
  "limit": 10,
  "min_similarity": 0.7
}
```

**Response**
```json
{
  "results": [
    {
      "memory": { /* Memory Entry object */ },
      "similarity": 0.95,
      "rank": 1
    }
  ],
  "query_embedding": [0.1, 0.2, ...],
  "total_searched": 100
}
```

---

## Visual Generation Endpoints

### Generate Image

```http
POST /api/visual/generate
Content-Type: application/json

{
  "conversation_id": "uuid",
  "thread_id": "uuid (optional)",
  "message_id": "uuid (optional)",
  "workflow": "flux_dev_portrait (optional, uses character default)",
  "prompt": "A serene portrait in soft painterly style",
  "parameters": {
    "steps": 30,
    "guidance_scale": 4.5,
    "seed": 42
  },
  "character_context": true,
  "use_loras": true
}
```

**character_context**: Include character visual identity in prompt assembly
**use_loras**: Apply character LoRAs from visual_identity config

**Response**
```json
{
  "job": { /* Visual Generation Job object */ },
  "estimated_time_seconds": 180,
  "poll_url": "/api/visual/jobs/{job_id}"
}
```

### Get Job Status

```http
GET /api/visual/jobs/{id}
```

**Response**: Full Visual Generation Job object

### Get Job Result

```http
GET /api/visual/jobs/{id}/result
```

Returns image file directly.

**Response**: Binary image data (PNG/JPEG)
**Headers**: `Content-Type: image/png`

**Error (404)**: Job not complete or failed

### Cancel Job

```http
DELETE /api/visual/jobs/{id}
```

**Response**
```json
{
  "success": true,
  "job": { /* Updated job with status: "cancelled" */ }
}
```

---

## Ambient Activity Endpoints

### Get Current Activity

```http
GET /api/conversations/{id}/activity
```

**Response**
```json
{
  "activity": { /* Activity object or null */ }
}
```

### Set Activity

```http
POST /api/conversations/{id}/activity
Content-Type: application/json

{
  "description": "Painting a portrait in a quiet studio",
  "scope": "character",
  "mood": "calm",
  "user_confirmed": true
}
```

**Response**: Created Activity object

### Clear Activity

```http
DELETE /api/conversations/{id}/activity
```

**Response**
```json
{
  "success": true
}
```

### Propose Activity

```http
POST /api/conversations/{id}/activity/propose
Content-Type: application/json

{
  "context": "User asked 'what are you up to?'"
}
```

Triggers character to propose an activity via LLM.

**Response**
```json
{
  "activity": {
    "description": "Sketching ideas for a new creative project",
    "mood": "thoughtful",
    "user_confirmed": false,
    "proposed_by": "character"
  },
  "requires_confirmation": true
}
```

---

## System Endpoints

### Health Check

```http
GET /api/system/health
```

**Response**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-27T12:30:00Z",
  "version": "1.0.0"
}
```

### System Status

```http
GET /api/system/status
```

**Response**
```json
{
  "llm": {
    "connected": true,
    "model": "mixtral-8x7b-instruct",
    "status": "ready"
  },
  "tts": {
    "engine": "xtts_v2",
    "loaded": true,
    "status": "ready"
  },
  "stt": {
    "engine": "faster-whisper",
    "loaded": true,
    "status": "ready"
  },
  "comfyui": {
    "connected": true,
    "url": "http://localhost:8188",
    "queue_size": 2,
    "status": "ready"
  },
  "memory": {
    "vector_store": "chroma",
    "connected": true,
    "total_entries": 156
  },
  "storage": {
    "database": "sqlite",
    "path": "data/chorus.db",
    "size_mb": 45.2
  }
}
```

### Get Configuration

```http
GET /api/system/config
```

**Response**
```json
{
  "config": {
    "llm": {
      "provider": "local",
      "model": "mixtral-8x7b-instruct",
      "context_window": 8192
    },
    "memory": {
      "implicit_enabled": false,
      "ephemeral_ttl_hours": 24,
      "embedding_model": "all-MiniLM-L6-v2"
    },
    "comfyui": {
      "url": "http://localhost:8188",
      "timeout_seconds": 300
    },
    "paths": {
      "characters": "characters/",
      "workflows": "workflows/",
      "data": "data/"
    }
  }
}
```

### Update Configuration

```http
PUT /api/system/config
Content-Type: application/json

{
  "memory": {
    "implicit_enabled": true
  },
  "comfyui": {
    "timeout_seconds": 600
  }
}
```

Partial updates supported.

**Response**: Updated full config object

---

## Error Responses

All errors follow this format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "field": "Additional context"
    },
    "recoverable": true,
    "degraded_services": ["comfyui"],
    "suggested_action": "Check ComfyUI connection and restart if needed"
  },
  "timestamp": "2025-12-27T12:30:00Z",
  "request_id": "uuid"
}
```

### Standard HTTP Status Codes

- `200 OK` - Success
- `201 Created` - Resource created
- `400 Bad Request` - Invalid input
- `404 Not Found` - Resource not found
- `409 Conflict` - Resource conflict (e.g., duplicate)
- `422 Unprocessable Entity` - Validation failed
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Dependency unavailable

### Error Codes

**General**
- `INVALID_REQUEST` - Malformed request
- `VALIDATION_ERROR` - Input validation failed
- `NOT_FOUND` - Resource not found
- `DUPLICATE` - Resource already exists

**Characters**
- `CHARACTER_NOT_FOUND` - Character ID invalid
- `CHARACTER_VALIDATION_FAILED` - Character config invalid
- `CHARACTER_IN_USE` - Cannot delete, has conversations

**Conversations/Threads**
- `CONVERSATION_NOT_FOUND` - Conversation ID invalid
- `THREAD_NOT_FOUND` - Thread ID invalid
- `CANNOT_DELETE_MAIN_THREAD` - Main thread deletion forbidden

**Memory**
- `MEMORY_NOT_FOUND` - Memory entry not found
- `MEMORY_SCOPE_INVALID` - Invalid scope configuration
- `EMBEDDING_FAILED` - Failed to generate embedding

**LLM**
- `LLM_UNAVAILABLE` - LLM service not available
- `LLM_GENERATION_FAILED` - Generation failed
- `LLM_TIMEOUT` - Generation timed out
- `CONTEXT_WINDOW_EXCEEDED` - Input too large

**Voice**
- `STT_FAILED` - Speech-to-text failed
- `TTS_FAILED` - Text-to-speech failed
- `AUDIO_FORMAT_UNSUPPORTED` - Audio format not supported
- `AUDIO_FILE_CORRUPT` - Audio file unreadable

**Visual**
- `COMFY_UNAVAILABLE` - ComfyUI not connected
- `WORKFLOW_NOT_FOUND` - Workflow file not found
- `GENERATION_FAILED` - Image generation failed
- `GENERATION_TIMEOUT` - Generation exceeded timeout
- `JOB_NOT_FOUND` - Job ID invalid

**System**
- `SYSTEM_LOCKED` - System locked, unlock required
- `CONFIG_INVALID` - Configuration invalid
- `SERVICE_DEGRADED` - Partial service availability

### Graceful Degradation

When a service is unavailable, errors include degradation info:

```json
{
  "error": {
    "code": "COMFY_UNAVAILABLE",
    "message": "ComfyUI connection failed",
    "degraded_services": ["visual_generation"],
    "available_services": ["chat_text", "chat_voice", "memory"],
    "suggested_action": "Image generation unavailable. Text and voice chat still functional."
  }
}
```

---

## File Management

### Generated File Paths

Chorus Engine organizes files by conversation:

```
data/
├── conversations/
│   └── {conversation_id}/
│       ├── audio/
│       │   ├── {message_id}_input.wav
│       │   └── {message_id}_output.wav
│       ├── images/
│       │   └── {job_id}.png
│       └── metadata.json
├── memory/
│   └── chroma_db/
└── cache/
```

### ComfyUI Integration

1. ComfyUI generates to its output folder (absolute path)
2. Chorus Engine moves (not copies) file to conversation folder
3. Database stores relative path within Chorus structure
4. API returns both absolute path and relative path

**Rationale**: 
- Avoid file duplication
- Maintain logical conversation grouping
- Preserve ComfyUI compatibility
- Enable efficient backups

---

## Rate Limiting

v1 has no rate limiting (single-user, local).

Future versions may add:
- Per-endpoint request limits
- Token generation limits
- Concurrent job limits

---

## Versioning

API version included in base path (future):
- v1: `/api/...`
- v2: `/api/v2/...`

v1 endpoints remain stable. Breaking changes require new version.

---

## WebSocket Lifecycle

### Connection Flow

1. Client sends HTTP request to chat endpoint
2. Server responds with WebSocket URL + session ID
3. Client connects to WebSocket with session ID
4. Client sends init message with thread context
5. Server validates and begins generation
6. Server streams tokens/chunks
7. Server sends completion message
8. Connection closes or waits for next message

### Timeout Behavior

- Connection timeout: 60 seconds of inactivity
- Generation timeout: Configurable per model (default: 120 seconds)
- Clients should reconnect on disconnect

### Error Handling

- Parse errors close connection with error message
- Recoverable errors send error message, keep connection open
- Fatal errors close connection

---

## Debug Mode

When system config has `debug: true`:

### Additional Response Fields

```json
{
  "debug": {
    "prompt_assembly": {
      "layers": [...],
      "total_tokens": 2048,
      "memory_retrieved": 5
    },
    "generation_params": {...},
    "timing_breakdown_ms": {
      "memory_retrieval": 45,
      "prompt_assembly": 12,
      "llm_generation": 1250,
      "total": 1307
    }
  }
}
```

### Debug Endpoints

```http
GET /api/debug/prompts/{message_id}
GET /api/debug/memory/embeddings/{memory_id}
GET /api/debug/logs
```

---

## Summary

This API specification:
- Covers all v1 features comprehensively
- Enables seamless text/voice switching
- Supports async visual generation
- Provides flexible memory management
- Includes detailed error handling
- Maintains local-first, transparent operation
- Scales cleanly for future enhancements

Next steps: Token budget management, Memory retrieval algorithm, ComfyUI integration details.
