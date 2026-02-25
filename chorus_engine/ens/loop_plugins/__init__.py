"""Loop-kind plugin interfaces and registry."""

from chorus_engine.ens.loop_plugins.contracts import (
    LoopKindPlugin,
    LoopStepPlan,
    PassExecutionResult,
    StepExecutionAggregate,
    StepPassPlan,
    StepOutcomePolicy,
    StepOutcomeResolution,
)
from chorus_engine.ens.loop_plugins.registry import get_loop_plugin, has_loop_plugin, registered_loop_kinds

__all__ = [
    "LoopKindPlugin",
    "LoopStepPlan",
    "StepPassPlan",
    "StepOutcomePolicy",
    "StepOutcomeResolution",
    "PassExecutionResult",
    "StepExecutionAggregate",
    "get_loop_plugin",
    "has_loop_plugin",
    "registered_loop_kinds",
]
