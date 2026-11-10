"""En-Masse prototype: a small simulator for exploring synchronization policies
for multi-agent execution, from conservative to optimistic.

This is an early prototype. It simulates agents; it does not run them.
"""

from .checker import check
from .engine import Engine, simulate
from .policies import (
    Conservative,
    Hybrid,
    Optimistic,
    Policy,
    Serial,
    Unsynchronized,
    gated_optimistic,
)
from .workload import AgentTask, Step, generate

__all__ = [
    "AgentTask",
    "Conservative",
    "Engine",
    "Hybrid",
    "Optimistic",
    "Policy",
    "Serial",
    "Step",
    "Unsynchronized",
    "check",
    "gated_optimistic",
    "generate",
    "simulate",
]

__version__ = "0.0.1"
