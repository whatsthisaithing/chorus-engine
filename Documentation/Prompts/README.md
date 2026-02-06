# System Prompts Specifications - Index

**Purpose**: Comprehensive documentation of all LLM prompts used throughout Chorus Engine  
**Maintained**: Updated as prompts are modified or new prompts added  
**Version**: 1.0 (December 31, 2025)

---

## Overview

Chorus Engine uses carefully crafted prompts to guide LLM behavior across different functions. Each prompt type serves a specific purpose and follows design principles optimized for that use case.

This folder contains detailed specifications for each prompt type, including:
- Full prompt templates
- Design rationale
- Configuration options
- Integration points
- Editing guidelines
- Troubleshooting tips

---

## Prompt Types

### 1. Conversational System Prompts

**File**: [`conversational_system_prompts.md`](conversational_system_prompts.md)

**Purpose**: Define character personality and interaction style during chat

**Key Features**:
- Immersion levels (minimal, balanced, full, unbounded)
- Disclaimer behavior control
- Character personality establishment
- Roleplay boundary management

**Used In**:
- Every chat message generation
- System prompt for LLM conversations
- Character identity reinforcement

**Edit When**:
- Creating new characters
- Adjusting roleplay philosophy
- Changing immersion guidelines
- Refining character behavior patterns

**Location**: `chorus_engine/services/system_prompt_generator.py`

---

### 2. Memory Extraction Prompts

**File**: [`memory_extraction.md`](memory_extraction.md)

**Purpose**: Extract memorable facts about user from conversations

**Key Features**:
- Automatic memory extraction
- Memory types (fact, project, experience, story, relationship)
- Confidence scoring (0.0-1.0)
- Anti-hallucination safeguards
- JSON structured output

**Used In**:
- Background extraction after every user message
- Automatic user profile building
- Character continuity across conversations

**Edit When**:
- Adding new memory categories
- Improving extraction accuracy
- Preventing hallucination patterns
- Adjusting confidence thresholds

**Location**: `chorus_engine/services/background_memory_extractor.py`

---

### 3. Image Generation Prompts

**File**: [`image_generation.md`](image_generation.md)

**Purpose**: Generate detailed image descriptions for Stable Diffusion/ComfyUI

**Key Features**:
- Context-aware prompt generation
- 100-300 word detailed descriptions
- Character depiction rules
- Style integration
- Technical specification inclusion

**Used In**:
- Image request detection and preparation
- Prompt preview before generation
- ComfyUI workflow integration

**Edit When**:
- Improving prompt detail level
- Adjusting character depiction logic
- Adding context extraction examples
- Refining style guidance

**Location**: `chorus_engine/services/image_prompt_service.py`

---

### 4. Video Generation Prompts

**File**: [`video_generation.md`](video_generation.md)

**Purpose**: Generate motion-focused video descriptions for video workflows

**Key Features**:
- Motion/action emphasis
- Camera movement guidance
- Character depiction rules
- JSON structured output

**Used In**:
- Video request detection and preparation
- Prompt preview before generation
- ComfyUI video workflow integration

**Location**: `chorus_engine/services/video_prompt_service.py`

---

### 5. Scene Capture Prompts

**File**: [`scene_capture.md`](scene_capture.md)

**Purpose**: Third-person observer prompts for capturing the current scene

**Key Features**:
- Omniscient observer perspective
- Multi-participant inclusion
- Context weighting for “current moment”
- JSON structured output

**Used In**:
- Scene capture button (manual capture)
- Prompt preview before generation

**Location**: `chorus_engine/services/scene_capture_prompt_service.py`

---

### 6. Conversation Analysis Prompts

**File**: [`conversation_analysis.md`](conversation_analysis.md)

**Purpose**: Analyze full conversations to extract memories and summaries

**Key Features**:
- Full conversation analysis
- Memory extraction + summaries
- Themes and emotional arc
- JSON structured output

**Used In**:
- Manual “Analyze Now” flow
- Conversation summaries for retrieval

**Location**: `chorus_engine/services/conversation_analysis_service.py`

---

## Design Principles

### Across All Prompts

**1. Explicit Instructions**:
- Clear, unambiguous language
- Specific examples (good and bad)
- Repeated critical rules

**2. Structured Output**:
- JSON where appropriate (extraction, image)
- Consistent formatting
- Easy parsing

**3. Anti-Hallucination**:
- "DO NOT" rules for common mistakes
- Counter-examples showing wrong behavior
- Confidence scoring for uncertainty

**4. Context Awareness**:
- Use conversation history intelligently
- Extract relevant details
- Ignore meta-discussion

**5. Maintainability**:
- Single source of truth per prompt type
- Clear editing locations
- Version control via git

---

## Prompt Engineering Best Practices

### Temperature Settings

Different tasks require different temperatures:

| Task | Temperature | Reason |
|------|-------------|--------|
| Conversation | 0.7 (Nova) / 0.5 (Alex) | Creative but coherent |
| Memory Extraction | 0.1 | Consistent, structured output |
| Image Prompts | 0.3 | Consistent, detailed descriptions |

### Length Guidelines

| Prompt Type | Token Budget | Reason |
|-------------|--------------|--------|
| System Prompt | 150-400 tokens | Sent with every message |
| Extraction Prompt | 600-800 tokens | Complex task needs examples |
| Image Prompt | 800-1200 tokens | Detailed instructions + examples |

### Example Quality

**Good Examples**:
- Specific and concrete
- Show desired output format
- Cover common cases
- Demonstrate edge cases

**Bad Examples**:
- Critical for anti-hallucination
- Show what NOT to do
- Explain why it's wrong
- Prevent common mistakes

---

## Editing Workflow

### 1. Identify Need

**When to Edit Prompts**:
- Undesired behavior observed
- Hallucination patterns detected
- New feature requirements
- Performance optimization

### 2. Locate Prompt

**Quick Reference**:
- Conversation: `system_prompt_generator.py`
- Extraction: `background_memory_extractor.py` → `_build_extraction_prompt()`
- Images: `image_prompt_service.py` → `_build_system_prompt()`

### 3. Make Changes

**Best Practices**:
- Add examples before changing instructions
- Test incrementally (one change at a time)
- Document reasoning in commit message
- Update specification document

### 4. Test Thoroughly

**Test Scenarios**:
- Happy path (typical usage)
- Edge cases (unusual inputs)
- Error cases (malformed responses)
- Performance (token count, latency)

### 5. Update Documentation

**Required Updates**:
- Update specification markdown file
- Add "Last Modified" date
- Document reasoning for change
- Update examples if applicable

---

## Token Budget Management

### Context Window Allocation

Chorus Engine uses Qwen2.5:14B-Instruct (32K context):

```
Total: 32,768 tokens

Breakdown (after system prompt + document injection):
- Conversation Context Summaries: ~5% of remaining budget
- Retrieved Memories: ~20% of remaining budget
- Message History: ~50% of remaining budget
- Document Chunks: ~15% of remaining budget
- Reserve: ~10% of remaining budget
```

### Optimization Strategies

**1. Keep System Prompts Lean**:
- Move backstory to core memories (not system prompt)
- Use immersion guidance instead of long instructions
- Target <400 tokens for system prompt

**2. Efficient Extraction Prompts**:
- Don't send full conversation history
- Only analyze recent messages (1-3 at a time)
- Reuse prompt template (not duplicated per message)

**3. Image Prompts Can Be Large**:
- One-time use (not sent repeatedly)
- Complexity justified by output quality
- 800-1200 tokens acceptable

---

## Integration Architecture

### Service Layer

```
PromptAssemblyService
├── SystemPromptGenerator → Conversation prompts
├── MemoryRetrievalService → Fetches memories
└── TokenCounterService → Budget management

BackgroundExtractionManager
└── MemoryExtractionService → Extraction prompts

ImageGenerationOrchestrator
└── ImagePromptService → Image prompts
```

### API Layer

```
POST /messages (streaming)
├── Assemble system prompt
├── Retrieve memories
├── Format history
└── Stream LLM response

Background extraction queue
├── Extract facts
├── Deduplicate
└── Store in vector DB

POST /threads/{id}/detect-image-request
├── Detect request
├── Generate prompt
└── Return preview
```

### Database Layer

```
conversations → Tracks conversation metadata
messages → Stores chat history
memories → Stores extracted facts
images → Links images to messages
vector_store → Semantic search for memories
```

---

## Testing Prompts

### Manual Testing

**Conversation Prompts**:
1. Create conversation with character
2. Send messages testing immersion level
3. Verify personality consistency
4. Check disclaimer behavior

**Extraction Prompts**:
1. Send messages with facts
2. Check extracted memories in UI
3. Verify no hallucinations
4. Test category classification

**Image Prompts**:
1. Request image with context
2. Review generated prompt
3. Check detail level (100-300 words)
4. Verify character depiction

### Automated Testing

**Unit Tests** (future):
```python
def test_extraction_prompt_no_hallucination():
    messages = [Message(role="user", content="Hello Nova")]
    prompt = build_extraction_prompt(messages, "Nova")
    # Assert prompt contains anti-hallucination rules
    assert "DO NOT confuse" in prompt
    assert "Hello Nova" in prompt
```

**Integration Tests**:
```python
def test_memory_extraction_flow():
    # Send message
    # Wait for extraction
    # Check database
    # Verify no duplicates
```

---

## Troubleshooting Guide

### Issue: Hallucinations in Extraction

**Symptoms**: Facts that weren't mentioned appear in memories

**Diagnosis**:
1. Check extraction logs for JSON output
2. Identify hallucination pattern
3. Review prompt for relevant rules

**Fix**:
1. Add counter-example to prompt
2. Strengthen "DO NOT" rules
3. Lower confidence threshold for inclusion

---

### Issue: Inconsistent Character Behavior

**Symptoms**: Character breaks roleplay, changes personality

**Diagnosis**:
1. Check system prompt generation
2. Verify immersion level setting
3. Review character YAML config

**Fix**:
1. Clarify personality in system prompt
2. Adjust immersion guidance
3. Add specific behavior rules

---

### Issue: Poor Image Prompts

**Symptoms**: Vague descriptions, missing context, wrong character appearance

**Diagnosis**:
1. Check context extraction
2. Review character self_description
3. Test with explicit requests

**Fix**:
1. Add context examples to prompt
2. Strengthen character depiction rules
3. Improve detail requirements

---

### Issue: JSON Parsing Failures

**Symptoms**: Extraction/image prompts fail to parse

**Diagnosis**:
1. Check LLM response in logs
2. Look for markdown fences
3. Check for text outside JSON

**Fix**:
1. Reinforce "ONLY JSON" instruction
2. Add parsing fallback logic
3. Strip markdown programmatically

---

## Version History

### Version 1.0 (December 31, 2025)

**Initial Documentation**:
- Conversational system prompts (4 immersion levels)
- Memory extraction prompts (fact/project/experience/story/relationship)
- Image generation prompts (context-aware)
- Created comprehensive specifications

**Design Philosophy**:
- Explicit over implicit instructions
- Examples over abstract rules
- Anti-hallucination safeguards
- Structured output formats

---

## Future Prompt Types

### Planned Additions

**1. Summarization Prompts** (Phase 8):
- Long conversation summarization
- Theme extraction
- Important moment detection

**2. Activity Proposal Prompts** (Phase 7):
- Suggest ambient activities
- Based on conversation context
- Character personality influences

**3. Evaluation Prompts**:
- Response quality assessment
- Fact verification
- Consistency checking

**4. Creative Writing Prompts**:
- Story generation
- Character development
- World-building

---

## Contributing

### Adding New Prompt Type

**Steps**:

1. **Create Implementation**:
   - Add service class to `chorus_engine/services/`
   - Implement prompt building method
   - Add to relevant orchestrator/manager

2. **Write Specification**:
   - Create `prompts_{type}.md` in this folder
   - Follow existing document structure
   - Include full template + examples

3. **Update Index**:
   - Add entry to this README
   - Link to new specification
   - Update architecture diagrams

4. **Test & Document**:
   - Add test scenarios
   - Document integration points
   - Update related docs

### Specification Template

Each prompt specification should include:

- Overview & Purpose
- Architecture & Flow
- Full Prompt Template
- Design Decisions
- Configuration Options
- Editing Guidelines
- Integration Points
- Troubleshooting
- Related Documentation

---

## Related Documentation

### General Architecture

- **API Specification**: `chorus_engine_api_specification_v_1.md`
- **Memory Retrieval**: `chorus_engine_memory_retrieval_algorithm_v_1.md`
- **ComfyUI Integration**: `chorus_engine_comfyui_integration_v_1.md`
- **Configuration**: `chorus_engine_configuration_validation_v_1.md`

### Development Guides

- **Character Development**: `Documentation/Development/character_development_best_practices.md`
- **Phase 4 Complete**: `Documentation/Development/PHASE_4_COMPLETE.md` (Memory extraction)
- **Phase 5 Complete**: `Documentation/Development/PHASE_5_COMPLETE.md` (Image generation)

### Planning Documents

- **Character Schema**: `Documentation/Planning/chorus_engine_character_schema_v_1.md`
- **Development Phases**: `Documentation/Development/DEVELOPMENT_PHASES.md`

---

## Maintenance Schedule

### Regular Reviews

**Monthly** (or after significant changes):
- Review prompt effectiveness
- Check for new hallucination patterns
- Update examples with real usage
- Optimize token usage

**Quarterly**:
- Comprehensive prompt audit
- Performance benchmarking
- User feedback integration
- Documentation updates

### Version Control

All prompts tracked in git:
- Commit messages explain changes
- Tag major prompt revisions
- Document breaking changes
- Maintain change log

---

## Quick Reference

### File Locations

| Prompt Type | Service File | Method | Line |
|-------------|--------------|--------|------|
| Conversation | `system_prompt_generator.py` | `generate()` | ~20 |
| Extraction | `background_memory_extractor.py` | `_build_extraction_prompt()` | ~276 |
| Image | `image_prompt_service.py` | `_build_system_prompt()` | ~178 |
| Video | `video_prompt_service.py` | `_build_system_prompt()` | ~48 |
| Scene Capture | `scene_capture_prompt_service.py` | `_build_system_prompt()` | ~160 |
| Conversation Analysis | `conversation_analysis_service.py` | `_build_analysis_prompt()` | ~250 |

### Key Parameters

| Prompt Type | Temperature | Token Budget | Output Format |
|-------------|-------------|--------------|---------------|
| Conversation | 0.5-0.7 | 150-400 | Natural text |
| Extraction | 0.1 | 600-800 | JSON array |
| Image | 0.3 | 800-1200 | JSON object |
| Video | 0.3 | 100-300 words (prompt rule says max 150) | JSON object |
| Scene Capture | 0.5 | 100-300 words | JSON object |
| Conversation Analysis | 0.1 | Up to 4000 tokens | JSON object |

### Common Issues

| Issue | Prompt Type | Fix Location |
|-------|-------------|--------------|
| Hallucinations | Extraction | Add counter-examples |
| Inconsistent personality | Conversation | Adjust immersion guidance |
| Vague descriptions | Image | Strengthen detail requirements |
| JSON parse errors | Any | Reinforce format rules |

---

**Last Updated**: December 31, 2025  
**Maintainer**: Development Team  
**Status**: Active Documentation
