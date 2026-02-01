# Heartbeat System Design

**Phase**: Phase D (Conversation Memory Enrichment)  
**Created**: February 1, 2026  
**Status**: Implemented

---

## Overview

The Heartbeat System provides intelligent background processing during idle periods. It automatically detects when the system is truly idle (no active users, no LLM calls, no image generation) and uses that time to perform beneficial background work—currently focused on analyzing stale conversations to extract memories and generate summaries.

The key innovation is the **multi-layer idle detection** that ensures background tasks never interfere with user interactions, while still making productive use of idle time.

---

## Core Philosophy

### The Invisible Processing Principle

**Central Insight**: Background work should be completely invisible to users. If they never know it's happening, we've succeeded.

**The Problem with Naive Background Processing**:
- ❌ Scheduled jobs (cron-style) may run during active use
- ❌ Immediate processing blocks user responses
- ❌ Fixed intervals waste resources when busy, miss opportunities when idle
- ❌ Resource competition degrades user experience

**The Solution**:
```
User sends message → System ACTIVE → Background tasks DEFER
       ↓
Response generated → Response sent
       ↓
User goes quiet → Countdown begins
       ↓
5 minutes no activity + No LLM calls + No ComfyUI jobs + GPU idle
       ↓
System IDLE → Background tasks RUN (one at a time)
       ↓
User returns → Tasks pause gracefully
```

**Why Invisible Processing Works**:
- Zero impact on user experience
- Productive use of idle hardware
- Graceful handling of activity spikes
- Self-healing memory system

---

### The Progressive Idle Detection Principle

**Central Insight**: A single "is idle" check isn't enough. Real safety requires multiple independent signals.

**The Idle Detection Stack**:
```
Level 1: Time Since Last Activity
         └─ Has it been 5+ minutes since any API call?
                    ↓ yes
Level 2: Active LLM Operations  
         └─ Are there any in-flight LLM requests?
                    ↓ no
Level 3: Active ComfyUI Jobs
         └─ Are there any pending/running image/video generations?
                    ↓ no
Level 4: GPU Utilization (Optional)
         └─ Is the GPU compute usage below 15%?
                    ↓ yes
         
         SAFE TO PROCESS ✓
```

**Why Multiple Checks Work**:
- Time alone misses streaming responses
- LLM check misses image generation
- ComfyUI check misses external GPU activity
- GPU check catches gaming, training, external Ollama, etc.
- Each layer catches what others miss

---

### The Non-Blocking Batch Principle

**Central Insight**: Process tasks in small batches with frequent idle re-checks.

**The Problem with Greedy Processing**:
- Processing all 50 stale conversations blocks for hours
- User returns mid-batch → wait forever for completion
- Large batches are all-or-nothing

**The Solution**:
```
Queue: [conv1, conv2, conv3, conv4, conv5, ...]
              ↓
Batch size: 3 conversations per cycle
              ↓
Process conv1 → Re-check idle → Still idle? Continue
Process conv2 → Re-check idle → Still idle? Continue
Process conv3 → Re-check idle → User returned! STOP
              ↓
[conv4, conv5, ...] remain queued for next idle period
```

**Why Small Batches Work**:
- Quick response to user activity
- Predictable batch duration
- Progress survives interruption
- Graceful degradation under load

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    API Request Flow                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  Activity Middleware   │ ← Records all API activity
        │  (excluded paths:      │   (except heartbeat endpoints)
        │   /heartbeat/*)        │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │    IdleDetector        │
        │  ┌──────────────────┐  │
        │  │ last_activity_at │  │ ← Timestamp of last API call
        │  │ llm_call_count   │  │ ← Active LLM requests counter
        │  │ comfy_job_count  │  │ ← Active ComfyUI jobs counter
        │  └──────────────────┘  │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   HeartbeatService     │ ← Background loop (60s interval)
        │  ┌──────────────────┐  │
        │  │ _is_safe_to_process │ ← Multi-layer idle check
        │  │ _task_queue      │  │ ← Priority queue of tasks
        │  │ _task_finder     │  │ ← Auto-discovers new tasks
        │  │ _handlers        │  │ ← Task type → handler map
        │  └──────────────────┘  │
        └────────────┬───────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────────┐
│ StaleConversation│    │ ConversationAnalysis│
│   Finder         │    │   TaskHandler       │
│ ┌─────────────┐ │    │ ┌─────────────────┐ │
│ │ stale_hours │ │    │ │ analyze_conv()  │ │
│ │ min_messages│ │    │ │ save_analysis() │ │
│ └─────────────┘ │    │ └─────────────────┘ │
└─────────────────┘    └─────────────────────┘
```

### Data Flow

```
                    System Start
                         │
                         ▼
        ┌────────────────────────────────┐
        │ HeartbeatService.start()       │
        │ - Spawns async heartbeat loop  │
        │ - Registers task handlers      │
        │ - Sets task finder callback    │
        └────────────────┬───────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │ Heartbeat Loop (every 60s)     │
        │ 1. Check _is_safe_to_process() │
        │ 2. If queue empty → call finder│
        │ 3. Process batch (up to 3)     │
        │ 4. Re-check idle between tasks │
        └────────────────┬───────────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
     Queue Empty?              Queue Has Tasks?
         │                          │
         ▼                          ▼
  ┌──────────────────┐    ┌─────────────────────┐
  │ Task Finder      │    │ _process_batch()    │
  │ - Query DB for   │    │ - Pop highest prio  │
  │   stale convos   │    │ - Execute handler   │
  │ - Queue tasks    │    │ - Update stats      │
  └──────────────────┘    │ - Mark analyzed_at  │
                          └─────────────────────┘
```

---

## Configuration

### HeartbeatConfig (system.yaml)

```yaml
heartbeat:
  # Master switch
  enabled: true
  
  # Timing
  interval_seconds: 60        # How often the loop runs
  idle_threshold_minutes: 5   # Minutes before system considered idle
  resume_grace_seconds: 2     # Pause after activity before resuming
  
  # Conversation Analysis
  analysis_stale_hours: 24    # Hours since activity before eligible
  analysis_min_messages: 10   # Minimum messages to be worth analyzing
  analysis_batch_size: 3      # Conversations per batch cycle
  
  # GPU Check (NVIDIA only, optional)
  gpu_check_enabled: false    # Enable GPU utilization check
  gpu_max_utilization_percent: 15  # Skip if GPU busier than this
```

### Configuration Rationale

| Setting | Default | Rationale |
|---------|---------|-----------|
| `interval_seconds: 60` | Frequent enough to catch idle windows, rare enough to not waste resources |
| `idle_threshold_minutes: 5` | Long enough that brief pauses don't trigger, short enough to use genuine idle time |
| `analysis_stale_hours: 24` | Gives user time to continue conversation naturally before analyzing |
| `analysis_min_messages: 10` | Ensures there's enough content for meaningful analysis |
| `analysis_batch_size: 3` | ~15-30 seconds per batch, allows quick response to user return |
| `gpu_max_utilization_percent: 15` | Idle GPUs sit at 0-5%, active workloads spike to 80%+ |

---

## Idle Detection Deep Dive

### IdleDetector Class

```python
class IdleDetector:
    """Tracks system activity state for safe background processing."""
    
    # Configuration
    idle_threshold_minutes: float = 5.0
    
    # State tracking
    last_activity_at: datetime      # Last API request timestamp
    _llm_call_counter: int = 0      # In-flight LLM requests
    _comfy_job_counter: int = 0     # Active ComfyUI jobs
    
    # Excluded paths (don't count as activity)
    _excluded_paths = {
        "/health", "/heartbeat/status", "/heartbeat/pause", 
        "/heartbeat/resume", "/heartbeat/queue"
    }
```

### Activity Recording

The middleware records activity for all requests **except** excluded paths:

```python
@app.middleware("http")
async def activity_tracking_middleware(request: Request, call_next):
    path = request.url.path
    idle_detector = app_state.get("idle_detector")
    
    if idle_detector and not idle_detector.should_exclude_path(path):
        idle_detector.record_activity()  # Updates last_activity_at
    
    response = await call_next(request)
    return response
```

**Why Exclude Heartbeat Paths?**
The UI polls `/heartbeat/status` every 5 seconds to show the status indicator. Without exclusion, this polling would constantly reset the idle timer, preventing background processing from ever running.

### LLM Call Tracking

```python
# In LLM client before generate():
idle_detector.start_llm_call()

# In LLM client after generate():
idle_detector.end_llm_call()
```

Counter-based (not boolean) because multiple concurrent LLM calls can be in flight.

### ComfyUI Job Tracking

```python
# When job starts:
idle_detector.start_comfy_job()

# When job completes or fails:
idle_detector.end_comfy_job()
```

---

## GPU Utilization Check

### Purpose

The GPU check is a **final safety gate** for users who:
- Game while leaving the Chorus Engine running
- Run external ComfyUI instances
- Train LoRA models
- Use Ollama/LM Studio outside of Chorus

### Implementation

```python
def get_gpu_utilization() -> Optional[int]:
    """Get GPU compute utilization % via nvidia-smi."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", 
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=2
    )
    # Returns 0-100 percentage, or None if unavailable
```

**Key Design Decisions**:
1. **Utilization, not memory**: Memory stays high when models are loaded but idle. Utilization drops to 0-5% when idle.
2. **NVIDIA only**: Uses nvidia-smi. Returns None on AMD/Intel (check skipped gracefully).
3. **Multi-GPU**: Takes max utilization across all GPUs.
4. **Opt-in**: Disabled by default (`gpu_check_enabled: false`).

### Utilization Thresholds

| GPU State | Typical Utilization |
|-----------|---------------------|
| Idle (model loaded, no inference) | 0-5% |
| Chorus LLM inference | 80-100% (brief spikes) |
| Gaming | 40-100% (sustained) |
| Training | 90-100% (sustained) |
| ComfyUI generation | 80-100% (sustained) |

Default threshold of 15% comfortably separates "idle" from "active".

---

## Task System

### Task Priority

```python
class TaskPriority(Enum):
    CRITICAL = 1  # Must run ASAP (e.g., failed Discord delivery)
    HIGH = 2      # Important but can wait for idle
    NORMAL = 3    # Standard background tasks
    LOW = 4       # Can be deferred indefinitely
```

Conversation analysis tasks use `LOW` priority—they're beneficial but never urgent.

### Task Lifecycle

```
Task Created → PENDING
       │
       ▼ (picked up by batch processor)
    RUNNING
       │
    ┌──┴──┐
    │     │
    ▼     ▼
COMPLETED  FAILED
              │
              ▼ (if retriable)
           PENDING (retry_count++)
```

### Task Handler Interface

```python
class BackgroundTaskHandler(ABC):
    @property
    @abstractmethod
    def task_type(self) -> str:
        """Unique identifier for this task type."""
        pass
    
    @abstractmethod
    async def execute(
        self, 
        task: BackgroundTask, 
        app_state: Dict[str, Any]
    ) -> TaskResult:
        """Execute the task, return result."""
        pass
    
    def should_retry(self, task: BackgroundTask, error: Exception) -> bool:
        """Determine if task should retry on failure."""
        return task.retry_count < task.max_retries
```

---

## Conversation Analysis Integration

### StaleConversationFinder

Discovers conversations needing analysis:

```sql
SELECT * FROM conversations 
WHERE (is_private != 'true' OR is_private IS NULL)
  AND (last_analyzed_at IS NULL OR last_analyzed_at < :cutoff_time)
  AND updated_at < :cutoff_time
ORDER BY updated_at ASC
LIMIT :batch_size * 2
```

Then filters in Python for `message_count >= min_messages`.

### ConversationAnalysisTaskHandler

1. Loads conversation and character from app_state
2. Calls `ConversationAnalysisService.analyze_conversation()`
3. Calls `ConversationAnalysisService.save_analysis()` to persist:
   - Extracted memories (facts, projects, experiences, etc.)
   - Conversation summary
   - Vector store embeddings
4. Updates `conversation.last_analyzed_at`

### Analysis Pipeline

```
Conversation (messages)
       │
       ▼
Build LLM Prompt
(includes character's memory extraction profile)
       │
       ▼
LLM Analysis → JSON Response
{
  "summary": "...",
  "themes": [...],
  "memories": [
    {"content": "...", "type": "FACT", ...},
    {"content": "...", "type": "PROJECT", ...}
  ]
}
       │
       ▼
Save to Database
- Memories → memories table
- Summary → conversation_summaries table
- Embeddings → ChromaDB vector store
       │
       ▼
Update last_analyzed_at
(prevents re-analysis until stale again)
```

---

## UI Integration

### Status Indicator

A small colored dot in the sidebar header shows heartbeat status:

| Color | Meaning |
|-------|---------|
| 🔵 Blue (pulsing) | Processing background task |
| 🟢 Green | Idle, ready for background work |
| 🟡 Yellow | Paused (user paused manually) |
| 🔴 Red | Waiting (system active, not idle) |
| ⚫ Gray | Disabled or unavailable |

### Status Polling

```javascript
// Poll every 5 seconds
setInterval(async () => {
    const response = await fetch('/heartbeat/status');
    const status = await response.json();
    updateStatusIndicator(status);
}, 5000);
```

Polling `/heartbeat/status` is excluded from activity tracking to avoid resetting idle timer.

### Click-to-Show-Details

Clicking the status indicator shows a toast with:
- Current state (idle/processing/paused/waiting)
- Time until idle (if waiting)
- Queue length (if tasks pending)
- GPU utilization (if check enabled)

---

## Error Handling

### Graceful Degradation

| Failure Mode | Behavior |
|--------------|----------|
| nvidia-smi not available | GPU check returns None, skipped gracefully |
| LLM timeout during analysis | Task marked failed, retries later |
| Database error | Task fails, conversation retried next cycle |
| User activity during batch | Batch stops, remaining tasks stay queued |

### Retry Logic

```python
def should_retry(self, task: BackgroundTask, error: Exception) -> bool:
    error_str = str(error).lower()
    
    # Permanent failures - don't retry
    if "not found" in error_str:
        return False
    if "too short" in error_str:
        return False
    
    # Transient failures - retry up to max_retries
    return task.retry_count < task.max_retries
```

---

## Diagnostic Tools

### check_stale_conversations.py

Utility script to diagnose conversation eligibility:

```bash
python utilities/check_stale_conversations.py --from-config --all
```

Output shows:
- ✅ Eligible conversations (with details)
- ❌ Private conversations (skipped)
- ⚠️ Too few messages (below threshold)
- ⏰ Too recent activity (not stale yet)
- 🔄 Recently analyzed (not due yet)

---

## Future Enhancements

### Potential Task Types

The heartbeat system is designed to be extensible. Future task types could include:

1. **Memory Consolidation**: Merge related memories, remove duplicates
2. **Vector Store Maintenance**: Re-index embeddings, cleanup orphans
3. **Conversation Archival**: Compress very old conversations
4. **Character Statistics**: Generate usage analytics
5. **Proactive Notifications**: Character-initiated outreach (Discord)

### Potential Optimizations

1. **Adaptive Batch Sizing**: Increase batch size during extended idle periods
2. **Priority Aging**: Bump priority of long-queued tasks
3. **Smart Scheduling**: Learn user activity patterns to predict idle windows
4. **Multi-GPU Support**: Distribute tasks across available GPUs

---

## Design Decisions Log

### Why 60-second interval?

**Considered alternatives**:
- 10 seconds: Too aggressive, wastes CPU on constant checking
- 300 seconds: Misses short idle windows

**Decision**: 60 seconds balances responsiveness with efficiency. Most users don't leave for exactly 5 minutes—they either stay (under 5 min) or leave for extended periods (5+ min). 60-second checks catch the latter quickly without overhead.

### Why counter-based LLM tracking?

**Considered alternatives**:
- Boolean flag: Doesn't handle concurrent calls
- Queue inspection: Invasive, requires access to LLM client internals

**Decision**: Simple increment/decrement counter. Thread-safe, non-invasive, handles any number of concurrent calls.

### Why opt-in GPU check?

**Considered alternatives**:
- Always on: Breaks on AMD/Intel systems
- Auto-detect: Complexity for marginal benefit

**Decision**: Opt-in (`gpu_check_enabled: false` default). Power users who need it can enable it; others aren't affected by missing nvidia-smi.

### Why utilization over memory?

**Considered alternatives**:
- Memory threshold: "Require 4GB free"
- Combined: Memory AND utilization

**Decision**: Pure utilization. Memory is misleading—Ollama keeps models loaded in VRAM indefinitely, showing high "usage" even when completely idle. Utilization correctly identifies actual GPU activity.

---

## Files Reference

| File | Purpose |
|------|---------|
| `chorus_engine/services/heartbeat_service.py` | Main service, loop, task queue |
| `chorus_engine/services/idle_detector.py` | Activity tracking, idle state |
| `chorus_engine/services/conversation_analysis_task.py` | Task handler, stale finder |
| `chorus_engine/config/models.py` | HeartbeatConfig dataclass |
| `config/system.yaml` | User configuration |
| `web/js/app.js` | Status indicator polling |
| `web/js/system_settings.js` | UI settings editor |
| `utilities/check_stale_conversations.py` | Diagnostic script |
