# Chorus Engine Design Documentation

**Purpose**: Permanent design documentation for Chorus Engine's core systems  
**Audience**: Future developers, contributors, and maintainers  
**Status**: Living documents - updated as systems evolve

---

## Overview

This folder contains polished design documents for Chorus Engine's major systems. Unlike development plans and specifications (which may be archived), these documents capture the **permanent architectural vision** of systems that work well and should be preserved.

Each document includes:
- **Philosophy**: Core principles and design insights
- **Architecture**: Component diagrams and data models
- **Design Decisions**: Why we chose this approach (with alternatives considered)
- **Known Limitations**: Honest assessment of trade-offs
- **Future Enhancements**: Potential improvements to pursue

---

## Design Documents

### 1. [Memory Intelligence System](MEMORY_INTELLIGENCE_SYSTEM.md)

**Phase 8: Memory Intelligence & Conversation Lifecycle**

The crown jewel of Chorus Engine - a comprehensive system for intelligent memory extraction, temporal awareness, type-aware memory management, and conversation understanding.

**Key Features**:
- Five memory types (FACT, PROJECT, EXPERIENCE, STORY, RELATIONSHIP)
- Temporal weighting (conversation-relative, never decays)
- Immersion-level memory profiles (minimal → unbounded)
- Smart conversation summarization (selective preservation)
- Cross-conversation continuity (natural greetings)
- Whole-conversation analysis

**Philosophy**: Memory-as-truth principle - no explicit state, everything inferred from memory.

**Status**: Phase 8 Days 1-9 complete, Day 10 (frontend) in progress

---

### 2. [Vector Store & Semantic Memory System](VECTOR_STORE_SYSTEM.md)

**Phase 3: Vector Memory & Advanced Features**

The foundation of intelligent memory retrieval using semantic embeddings instead of keyword matching.

**Key Features**:
- ChromaDB vector database with persistent storage
- sentence-transformers embeddings (384 dimensions)
- Dual storage (SQL metadata + vector semantics)
- Hybrid ranking (semantic + priority + type + temporal)
- Character-scoped collections
- <200ms retrieval latency

**Philosophy**: Semantic-over-keyword - users think in concepts, not keywords.

**Status**: Complete and battle-tested (Phase 3)

---

### 3. [Character & Immersion System](CHARACTER_IMMERSION_SYSTEM.md)

**Phase 3: Character Management & Immersion Levels**

A unique approach to character personality that balances roleplay immersion with user comfort through four graduated immersion levels.

**Key Features**:
- Four immersion levels (minimal, balanced, full, unbounded)
- Configurable personality boundaries (preferences, experiences, sensations)
- Character-specific LLM preferences and temperatures
- Core memories (immutable backstory)
- Visual identity and workflow configurations
- System prompt generation based on immersion

**Philosophy**: Characters are data, not code. Users choose comfort level.

**Status**: Complete with UI and backend support (Phase 3)

---

### 4. [ComfyUI Workflow Orchestration System](COMFYUI_WORKFLOW_SYSTEM.md)

**Phase 5: Image Generation, Phase 6: Audio Generation**

A workflow-first approach to multimodal generation that treats image and audio generation as composable workflows rather than black boxes.

**Key Features**:
- Workflow types (image, audio, video-future)
- Character-specific workflow libraries
- Placeholder injection (__CHORUS_PROMPT__, __CHORUS_TEXT__, etc.)
- VRAM coordination between LLM and ComfyUI
- Async job submission and polling
- Storage and database recording

**Philosophy**: Explicit workflows over magic generation. Users control every step.

**Status**: Image (Phase 5) and Audio (Phase 6) complete

---

### 5. [Background Memory Extraction System](BACKGROUND_EXTRACTION_SYSTEM.md)

**Phase 4.1: Implicit Memory Extraction**

Non-blocking, intelligent memory extraction that runs in the background without interrupting conversation flow.

**Key Features**:
- Async queue-based processing
- LLM-powered fact extraction
- Semantic deduplication (vector similarity)
- Confidence-based auto-approval
- Privacy mode integration
- Character's loaded model (no VRAM swapping)

**Philosophy**: Extract continuously, never block users, respect privacy.

**Status**: Complete with Phase 7.5 optimization (uses character model)

---

## Document Standards

### What Belongs Here

**Include**:
- Systems that are complete and working well
- Architectural decisions worth preserving
- Trade-offs and limitations honestly documented
- Future enhancement ideas
- Philosophy and rationale (the "why")

**Exclude**:
- Implementation checklists (keep in Development/)
- Temporary experiment notes
- Incomplete systems still being prototyped
- Step-by-step tutorials (keep in USER_GUIDE.md)

### Document Structure

Each design document should follow this general structure:

1. **Overview**: What is this system and why does it exist?
2. **Core Philosophy**: Central insights and principles
3. **Architecture**: Components, data models, flow diagrams
4. **Key Components**: Deep dive into major pieces
5. **Design Decisions & Rationale**: Why we chose this approach
6. **Known Limitations**: Honest assessment of trade-offs
7. **Performance Characteristics**: Latency, storage, accuracy metrics
8. **Testing & Validation**: How we verify it works
9. **Migration Guide**: How to adopt or upgrade
10. **Future Enhancements**: Potential improvements
11. **Conclusion**: Summary of achievements and status

### Maintaining These Documents

**When to Update**:
- Major architectural changes
- New limitations discovered
- Performance improvements implemented
- Future enhancements completed
- Design decisions reconsidered

**How to Update**:
1. Add dated "Update" sections at top
2. Keep original content for history
3. Mark deprecated sections clearly
4. Update "Status" field
5. Increment version number

**Example Update Header**:
```markdown
## Update: January 15, 2026

**Change**: Added multilingual embedding support

**Motivation**: Non-English users needed better semantic search quality

**Impact**: 3x slower embeddings, but now works for 100+ languages

**See**: Future Enhancements → High Priority → Multi-Language Support (now complete)
```

---

## Related Documentation

### Active Development

- **[PHASE_8_PLAN.md](../Development/PHASE_8_PLAN.md)**: Current implementation plan
- **[PHASE_8_CHECKLIST.md](../Development/PHASE_8_CHECKLIST.md)**: Task tracking
- **[PHASE_8_PROGRESS.md](../Development/PHASE_8_PROGRESS.md)**: Daily updates

### Specifications

- **[memory_intelligence_system.md](../Specifications/memory_intelligence_system.md)**: Technical spec
- **[Prompts/](../Specifications/Prompts/)**: LLM prompt specifications

### User Documentation

- **[USER_GUIDE.md](../USER_GUIDE.md)**: End-user documentation
- **[CHARACTER_CONFIGURATION.md](../CHARACTER_CONFIGURATION.md)**: Character YAML reference
- **[WORKFLOW_GUIDE.md](../WORKFLOW_GUIDE.md)**: ComfyUI workflow creation

---

### 6. [Conversation & Thread Architecture](CONVERSATION_THREAD_ARCHITECTURE.md)

**Phase 2-4: Conversation Model & Message Privacy**

A subtle but important three-layer hierarchy (Character → Conversation → Thread → Message) that provides natural conversation organization with message-level privacy control.

**Key Features**:
- Three-layer hierarchy (vs. flat, 2-layer, or 4-layer alternatives)
- Message-level privacy (captured at send time, immutable)
- Privacy mode toggle (affects new messages only)
- Automatic title generation with tracking
- Thread transitions (fork, summarize)
- Character-scoped memories with conversation origin tracking

**Philosophy**: Characters as persistent entities, conversations as long-running relationships, threads as story arcs.

**Status**: Complete and battle-tested (Phases 2-4, Phase 8 enhancements)

---

### 7. [Prompt Assembly & Token Management](PROMPT_ASSEMBLY_TOKEN_MANAGEMENT.md)

**Phase 3: Token Budget Management**

A sophisticated token budgeting system that allocates context window space using priority-based memory selection and recent-first history truncation.

**Key Features**:
- 30/40/30 budget ratios (memory/history/reserve)
- Priority-based memory allocation
- Model-specific tokenizer for exact counts
- Recent-first history truncation (minimum 3 exchanges)
- Selective context preservation (Phase 8)
- Never truncate system prompt
- Summarization integration

**Philosophy**: Priority-based allocation, budget-ratio principle, never-truncate-system-prompt.

**Status**: Complete and excellent (Phase 3 foundation, Phase 8 enhancements)

---

### 8. [LLM Client Abstraction & Multi-Provider Support](LLM_CLIENT_ABSTRACTION.md)

**Mid-Development Refactor (January 2, 2026)**

A provider-agnostic LLM client interface enabling seamless switching between Ollama, LM Studio, and future OpenAI-compatible backends.

**Key Features**:
- BaseLLMClient abstract interface
- Factory pattern for provider selection
- Ollama provider (manual model management)
- LM Studio provider (JIT model loading, OpenAI-compatible)
- Unified LLMResponse type
- Health checking and error handling
- Streaming support

**Philosophy**: Provider-agnostic interface, fail-fast configuration errors, unified response types.

**Status**: Production-ready, tested with multiple providers

---

## Future Design Documents

Systems that will eventually deserve their own design docs:

### Privacy & Scoped Memory

**Why**: Message-level privacy is a unique design that works really well

**What to Cover**:
- Why message-level vs conversation-level
- Privacy at send time (not extraction time)
- Memory scoping (character vs conversation)
- Private message styling

**Status**: Need to write (Phase 4.1 complete)

---

## Contributing to Design Docs

If you're adding a new system or significantly enhancing an existing one, consider writing a design document if:

1. **It's architecturally significant**: Touches multiple components or establishes patterns
2. **It involves trade-offs**: You made choices that future devs should understand
3. **It's non-obvious**: The "why" is as important as the "what"
4. **It's complete and tested**: Design docs are for working systems, not experiments
5. **It has lasting value**: Will be relevant for years, not months

**Process**:
1. Copy template from an existing design doc
2. Fill in all sections thoroughly
3. Be honest about limitations
4. Include code examples and diagrams
5. Review for clarity (imagine reading this in 2 years)
6. Add to this index

---

## Design Philosophy Evolution

These documents also serve as a record of how Chorus Engine's design philosophy has evolved:

**Phase 1-2**: Basic chat with simple memory  
→ *Insight*: Memory needs structure and retrieval

**Phase 3**: Vector embeddings + immersion levels  
→ *Insight*: Semantic search > keywords, characters need personality boundaries

**Phase 4**: Background extraction + privacy  
→ *Insight*: Non-blocking extraction, message-level privacy is right granularity

**Phase 5-6**: ComfyUI workflows  
→ *Insight*: Explicit workflows > black-box generation

**Phase 7**: Intent detection  
→ *Insight*: LLMs can detect intents reliably inline

**Phase 8**: Memory intelligence  
→ *Insight*: Memory-as-truth, temporal weighting, conversation lifecycle management

Each phase built on lessons from the previous, and these documents capture that learning.

---

## Conclusion

These design documents represent the **institutional knowledge** of Chorus Engine. They capture not just what we built, but why we built it that way, what worked, what didn't, and what we'd like to improve.

They're living documents - they should evolve as the system evolves - but they should always represent the stable, well-understood core of what makes Chorus Engine work.

**Keep them updated. Keep them honest. Keep them useful.**

---

**Index Version**: 1.1  
**Last Updated**: January 4, 2026  
**Total Design Docs**: 8 (+ 1 future)
