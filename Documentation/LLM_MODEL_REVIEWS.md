# LLM Model Testing & Reviews

**Hardware**: NVIDIA RTX 5090  
**Last Updated**: January 4, 2026

---

## Overview

This document captures real-world testing experiences with different LLM models used in Chorus Engine. Ratings are based on actual usage with characters, conversations, and memory extraction on a high-end consumer GPU setup.

---

## Rating System

All models rated on a 1-5 star scale:

- ⭐ **1 Star**: Poor - Unusable for this purpose
- ⭐⭐ **2 Stars**: Below Average - Significant issues
- ⭐⭐⭐ **3 Stars**: Average - Acceptable with limitations
- ⭐⭐⭐⭐ **4 Stars**: Good - Reliable with minor issues
- ⭐⭐⭐⭐⭐ **5 Stars**: Excellent - Best in class

### Categories

- **Consistency**: Follows system prompts, maintains character personality, respects immersion boundaries
- **Roleplay/Immersion**: Natural dialogue, emotional depth, stays in character
- **Memory Extraction**: Accuracy extracting facts, appropriate confidence scores, minimal hallucination
- **Performance**: Generation speed (tokens/sec), VRAM usage, model loading time

---

## Model Reviews

---

### Qwen 2.5 Coder 14B - LM Studio

**Version**: qwen/qwen2.5-coder-14b  

#### Overview

Qwen 2.5 Coder 14B is a code-specialized variant of the Qwen 2.5 series, optimized for programming tasks and technical content. Features enhanced instruction following and structured output generation.

#### Testing Notes

This is the system default. It provides great all around performance including for light roleplay, but is definitely censored and more task focused.

#### Ratings

- **Consistency**: ⭐⭐⭐⭐⭐ (5/5)
- **Roleplay/Immersion**: ⭐⭐⭐ 3X/5)
- **Memory Extraction**: ⭐⭐⭐⭐⭐ (5/5)
- **Performance**: ⭐⭐⭐⭐ (4/5)

**Overall**: ⭐⭐⭐⭐ (4/5)

#### Recommended For

- General purpose or technically focused, task-oriented characters
- Very light full or unbounded (roleplay) characters

#### Not Recommended For

- Deep roleplay
- Uncensored use cases

#### Configuration Notes

```yaml
# Character YAML settings that worked well
preferred_llm:
  provider: lmstudio
  model: "qwen/qwen2.5-coder-14b"
  temperature: 0.7
  max_tokens: 2048
```

---

### Cydonia 24B v4.3 Heretic v2 - LM Studio

**Version**: cydonia-24b-v4.3-heretic-v2  

#### Overview

Cydonia 24B Heretic is a community-fine-tuned model focused on creative writing and roleplay scenarios. Known for uncensored responses and enhanced narrative capabilities.

#### Testing Notes

This is a FANTASTIC model for roleplay in virtually ANY situation. Responses naturally provide background narrative, action descriptions, decent dialog. Haven't encountered any limits so far.

#### Ratings

- **Consistency**: ⭐⭐⭐⭐⭐ (4/5)
- **Roleplay/Immersion**: ⭐⭐⭐⭐⭐ (5/5)
- **Memory Extraction**: ⭐⭐⭐⭐⭐ (4/5)
- **Performance**: ⭐⭐⭐⭐ (4/5)

**Overall**: ⭐⭐⭐⭐ (4/5)

#### Recommended For

- Deep roleplay
- Specifically good for truly uncensored conversation

#### Not Recommended For

- Technical tasks
- SFW needs

#### Configuration Notes

```yaml
# Character YAML settings that worked well
preferred_llm:
  provider: lmstudio
  model: "cydonia-24b-v4.3-heretic-v2"
  temperature: 0.9
  max_tokens: 2048
```

---

### Dolphin Mistral Nemo 12B - Ollama

**Version**: CognitiveComputations/dolphin-mistral-nemo:12b  

#### Overview

Dolphin Mistral Nemo is an uncensored fine-tune of Mistral's Nemo 12B model by Cognitive Computations. Designed for balanced performance with reduced content filtering and improved instruction following.

#### Testing Notes

This was a decent first model tested for unbounded (roleplay) characters. Performance was slower than other models due to VRAM needs exceeding even my 5090's capacity when factoring 
other system processes. With better VRAM management (outside of Chorus Engine), would probably perform better. Also had inconsistent performance with memory extraction and conversation 
quality, but might be tuneable. Decent alternative if ollama is preferred over LM Studio.

#### Ratings

- **Consistency**: ⭐⭐⭐ 3X/5)
- **Roleplay/Immersion**: ⭐⭐⭐⭐ (4/5)
- **Memory Extraction**: ⭐⭐⭐ (3/5)
- **Performance**: ⭐⭐⭐ (3/5)

**Overall**: ⭐⭐⭐ (3/5)

#### Recommended For

- Roleplay for ollama users (LM Studio and better models highly recommended)

#### Not Recommended For

- Less than 32gb of VRAM
- Deep roleplay
- Conversations where memory extraction is important

#### Configuration Notes

```yaml
# Character YAML settings that worked well
preferred_llm:
  provider: ollama
  model: "CognitiveComputations/dolphin-mistral-nemo:12b"
  temperature: 0.8
  max_tokens: 2048
```

---

<!-- Copy the template below for each model you test -->

---

### [Model Name] - [Provider]

**Version**: [Model version/quantization]  

#### Overview

[Brief description of the model - what it's designed for, notable features, etc.]

#### Testing Notes

[Your experience using this model - what worked well, what didn't, specific examples, etc.]

#### Ratings

- **Consistency**: ⭐⭐⭐⭐⭐ (5/5)
- **Roleplay/Immersion**: ⭐⭐⭐⭐ (4/5)
- **Memory Extraction**: ⭐⭐⭐⭐⭐ (5/5)
- **Performance**: ⭐⭐⭐⭐ (4/5)

**Overall**: ⭐⭐⭐⭐ (4.5/5)

#### Recommended For

- [Use case 1]
- [Use case 2]
- [Use case 3]

#### Not Recommended For

- [Use case where it struggles]
- [Another limitation]

#### Configuration Notes

```yaml
# Character YAML settings that worked well
llm:
  model: "model_name"
  temperature: 0.7
  max_response_tokens: 2048
  context_window: 8192
```