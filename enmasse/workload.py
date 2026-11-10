"""Synthetic multi-agent workloads.

A workload is a set of agents. Each agent runs a fixed sequence of steps. A step
stands for one "think, then act" cycle: an LLM call followed by a tool action.

The model is deliberately coarse. It captures only what a synchronization policy
needs to see:

* how long a step takes and what it costs (tokens),
* which shared objects the step reads and writes,
* whether the step has an irreversible external effect (for example, sending
  an email), and
* what each agent declares in advance about the objects it may write
  (its footprint), which is the lookahead available to conservative policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Step:
    """One think-and-act cycle of an agent."""

    duration: float                 # seconds of wall-clock time
    tokens: float                   # cost of the step, in tokens
    reads: frozenset[int]           # shared objects read at the start of the step
    writes: frozenset[int]          # shared objects written when the step completes
    irreversible: bool = False      # the step has an external effect that cannot be undone


@dataclass
class AgentTask:
    """The work of one agent.

    ``rank`` is the agent's position in the reference serial order. A run is
    correct if it is equivalent to running the agents one after another in rank
    order. Rank plays the role of virtual time in this prototype.

    ``declared_extra`` holds objects that the agent declares it may write but
    never does. It models a coarse, over-approximate footprint (for example
    "may touch anything under src/"). The larger it is, the worse the lookahead.
    """

    rank: int
    steps: list[Step]
    declared_extra: frozenset[int] = field(default_factory=frozenset)


def generate(
    rng: np.random.Generator,
    n_agents: int = 8,
    n_steps: int = 12,
    n_objects: int = 12,
    contention: float = 0.2,
    write_prob: float = 0.5,
    irreversible_prob: float = 0.1,
    footprint_slack: int = 0,
    mean_duration: float = 10.0,
    sigma: float = 0.5,
    tokens_per_second: float = 300.0,
) -> list[AgentTask]:
    """Draw a random workload.

    contention         probability that a step touches a shared object
    write_prob         probability that a step which touches a shared object also writes it
    irreversible_prob  probability that a step has an irreversible external effect
    footprint_slack    number of objects each agent declares but never writes
    """
    mu = np.log(mean_duration) - 0.5 * sigma**2   # so that the mean is mean_duration
    tasks: list[AgentTask] = []
    for rank in range(n_agents):
        steps: list[Step] = []
        written: set[int] = set()
        for _ in range(n_steps):
            duration = float(rng.lognormal(mu, sigma))
            reads: frozenset[int] = frozenset()
            writes: frozenset[int] = frozenset()
            if rng.random() < contention:
                obj = int(rng.integers(n_objects))
                reads = frozenset({obj})
                if rng.random() < write_prob:
                    writes = frozenset({obj})
                    written.add(obj)
            steps.append(
                Step(
                    duration=duration,
                    tokens=duration * tokens_per_second,
                    reads=reads,
                    writes=writes,
                    irreversible=bool(rng.random() < irreversible_prob),
                )
            )
        candidates = [o for o in range(n_objects) if o not in written]
        k = min(footprint_slack, len(candidates))
        extra = rng.choice(candidates, size=k, replace=False) if k else []
        tasks.append(
            AgentTask(rank=rank, steps=steps, declared_extra=frozenset(int(o) for o in extra))
        )
    return tasks
