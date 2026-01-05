# Debugging Guide for Chorus Engine

## Overview

This guide documents the comprehensive debugging system implemented to trace all LLM interactions and troubleshoot intent detection/memory extraction issues.

## Debug Log Structure

All debug logs are organized by conversation in `data/debug_logs/conversations/{conversation_id}/`:

```
data/debug_logs/
  conversations/
    {conversation_id}/
      conversation.jsonl     # All LLM calls for this conversation
      extractions.jsonl      # Memory extraction attempts
```

### Files

- **conversation.jsonl**: Logs every LLM interaction (chat, intent detection, image prompts, etc.)
- **extractions.jsonl**: Detailed logs of background memory extraction (input, prompt, LLM response, filtered results)

## What Was Implemented

### 1. Debug Logger (`chorus_engine/utils/debug_logger.py`)

A comprehensive logging utility that captures EVERY LLM interaction with full context:

- **Per-conversation directories**: Each conversation gets its own folder in `data/debug_logs/conversations/{conversation_id}/`
- **conversation.jsonl**: Contains all LLM interactions for the conversation
- **Full prompt/response capture**: Every LLM call is logged with complete prompt and response
- **Rich metadata**: Model name, temperature, settings, timestamps, interaction type
- **Multiple interaction types tracked**:
  - `chat_stream`: Normal conversation responses
  - `intent_detection`: Intent classification and memory extraction
  - `image_prompt`: Image prompt generation
  - More can be added as needed

### 2. Background Memory Extraction Logging

**File**: `data/debug_logs/conversations/{conversation_id}/extractions.jsonl`

Each extraction attempt logs:
- **Input messages**: Raw messages sent for extraction (with roles)
- **System prompt**: The extraction instructions sent to LLM
- **User content**: The actual user messages being analyzed
- **LLM response**: Raw JSON response from model
- **Extracted memories**: What the LLM extracted (before filtering)
- **Saved memories**: What actually got saved (after defensive filters)
- **Errors**: Any errors during extraction

This provides complete transparency into:
- What the LLM sees vs. what the user actually said
- Which memories get filtered out and why
- Whether system prompts are leaking into extraction

### 3. Enhanced Intent Detection Validation

**Problem**: The 3B intent detection model was extracting character names from greetings as user names. For example:
- User: "Evenin' Sarah. Do you remember my name?" → Extracted "User's name is Sarah" ❌

**Solution**: Enhanced `_validate_extracted_facts()` with pattern detection:

#### Greeting Pattern Detection
```python
greeting_pattern = r"(?:^|\s)(?:hi|hey|hello|good\s+(?:morning|afternoon|evening|night)|evenin[g']?)\s+" + name
```
Detects: "Hi Sarah", "Hey Sarah", "Evenin' Sarah", "Hello Sarah", etc.

#### Address/Vocative Pattern Detection  
```python
address_pattern = r"(?:^|\s|,)" + name + r"(?:,|\s+(?:do|can|will|what|how|why|when|where))"
```
Detects: "Sarah, do you...", "Sarah, can you...", etc.

**Result**: Names appearing in these patterns are now rejected and logged with detailed reasoning.

### 4. Comprehensive Logging Integration

#### Intent Detection Service
- Logs every prompt sent to intent detection model
- Logs every response received
- Logs validation results (accepted/rejected facts)
- All stored in `data/debug_logs/conversations/{conversation_id}/conversation.jsonl`

#### Chat Streaming Endpoint
- Logs every conversation LLM call
- Full messages array (system + memories + history)
- Full response text
- Model settings (temperature, max_tokens)
- Character metadata
- All stored in `data/debug_logs/conversations/{conversation_id}/conversation.jsonl`

## How to Use

### 1. Server Logs (Real-time)

Watch the server logs for validation messages:

```
[INTENT DETECTION - STREAM] Analyzing user message: 'Evenin' Sarah. Do you remember my name?...'
Rejected name extraction - 'Sarah' appears in greeting, not a fact about user: 'Evenin' Sarah...'
[IMPLICIT MEMORY - STREAM] Found 0 facts to save (all rejected)
```

### 2. Debug Log Files (Detailed Analysis)

After a conversation, use the viewer script:

```bash
# View all logs for a conversation
python view_debug_log.py

# (Edit the conversation_id in the script first)
```

The script shows:
- All LLM interactions (conversation.jsonl)
- All extraction attempts (extractions.jsonl)
- Input/output for each call
- Filtered vs. saved memories

### 3. API Endpoints

#### Get Conversation Debug Log
```bash
curl http://localhost:8080/debug/conversation/{conversation_id}
```

Returns JSON with all interactions:
```json
{
  "conversation_id": "abc-123",
  "interaction_count": 15,
  "interactions": [
    {
      "timestamp": "2025-12-31T19:30:45.123456",
      "type": "intent_detection",
      "model": "qwen2.5:3b-instruct",
      "prompt": "You are an intent classifier...",
      "response": "{\"generate_image\": false, ...}",
      "settings": {"temperature": 0.1},
      "metadata": {"character_id": "sarah_v1", "message_preview": "Hi Sarah..."}
    },
    {
      "timestamp": "2025-12-31T19:30:46.789012",
      "type": "chat_stream",
      "model": "CognitiveComputations/dolphin-mistral-nemo:12b",
      "prompt": "[{\"role\": \"system\", \"content\": \"...\"}...]",
      "response": "Hello! I do remember...",
      "settings": {"temperature": 0.9, "max_tokens": 2048},
      "metadata": {"character_id": "sarah_v1", "thread_id": "xyz-789"}
    }
  ]
}
```

#### Clear Conversation Debug Log
```bash
curl -X DELETE http://localhost:8080/debug/conversation/{conversation_id}
```

## Common Issues and Solutions

### Issue: "User's name is [character name]" from greetings

**Symptoms**:
- User says "Hi Sarah" → Extracts "User's name is Sarah"
- User says "Evenin' Sarah, how are you?" → Extracts "User's name is Sarah"

**Debug Steps**:
1. Check server logs for validation warnings:
   ```
   Rejected name extraction - 'Sarah' appears in greeting, not a fact about user
   ```
2. Check debug log to see the exact prompt sent to intent detection model
3. Verify the user's message doesn't actually contain "I'm Sarah" or "My name is Sarah"

**Solution**: Pattern detection now catches these automatically. If still happening:
- Check if new greeting patterns need to be added to `greeting_pattern` regex
- Check if new vocative patterns need to be added to `address_pattern` regex

### Issue: Hallucinated names like "[Anonymous]"

**Symptoms**:
- User says "What are you up to?" → Extracts "User's name is [Anonymous]"
- Placeholder text appears in extractions

**Debug Steps**:
1. Check server logs for hallucination warnings:
   ```
   Rejected hallucinated fact with placeholder: 'User's name is [Anonymous]'
   ```
2. Check debug log to see what the intent model actually returned
3. Verify the prompt doesn't contain the hallucinated text

**Solution**: Validation layer catches common placeholder patterns. If new patterns emerge:
- Add them to `hallucination_patterns` list in `_validate_extracted_facts()`

### Issue: Wrong model being used for chat

**Symptoms**:
- Sarah (dolphin-mistral-nemo) feels like qwen2.5
- Character personality inconsistent

**Debug Steps**:
1. Check server logs for model loading:
   ```
   [CHAT] LLM Settings for Sarah: model=CognitiveComputations/dolphin-mistral-nemo:12b (override)
   [MODEL TRACKING] ensure_model_loaded called: model=...
   [MODEL TRACKING] Ollama ps reports: 1 loaded: ['qwen2.5:3b-instruct']
   [MODEL TRACKING] Unloading ['qwen2.5:3b-instruct'] before loading ...
   ```
2. Check debug log to see which model actually generated the response
3. Look for model switching events

**Solution**: Check the debug log's `model` field in chat interactions to confirm correct model was used.

## Debug Log File Structure

Each line in a `.jsonl` file is a complete JSON object:

```json
{
  "timestamp": "ISO 8601 timestamp",
  "type": "interaction_type",
  "model": "model name/identifier",
  "prompt": "full prompt text or JSON",
  "response": "full response text",
  "settings": {
    "temperature": 0.1,
    "max_tokens": 2048,
    "...": "..."
  },
  "metadata": {
    "character_id": "sarah_v1",
    "conversation_id": "abc-123",
    "...": "..."
  },
  "error": "error message if failed (null otherwise)"
}
```

## Performance Impact

- **Negligible runtime overhead**: Async file writes, no blocking
- **Disk space**: ~1-5KB per LLM interaction (prompts can be large)
- **Can be disabled**: Set `debug_logger.enabled = False` in production

## Future Enhancements

Possible additions:
- Web UI for viewing debug logs
- Filtering by interaction type or model
- Search within prompts/responses
- Export to formatted reports
- Automatic issue detection (e.g., detect model switching)
- Integration with testing framework

## Files Modified

- `chorus_engine/utils/debug_logger.py` (NEW) - Debug logging utility
- `chorus_engine/services/intent_detection_service.py` - Added `re` import, enhanced validation, debug logging
- `chorus_engine/api/app.py` - Added debug logging to chat endpoint, debug API endpoints
- `DEBUGGING_GUIDE.md` (NEW) - This file

## Next Steps

1. **Restart server** to load new code
2. **Start a new conversation** with Sarah
3. **Send test messages**:
   - "Hi Sarah. I'm John."
   - "Evenin' Sarah, how are you?"
   - "What are you up to?"
4. **Check logs** for validation warnings
5. **Retrieve debug log**: `GET /debug/conversation/{conversation_id}`
6. **Analyze** the actual prompts and responses

If issues persist, the debug logs will show EXACTLY what's being sent to each model and what's coming back, making it much easier to diagnose problems.
