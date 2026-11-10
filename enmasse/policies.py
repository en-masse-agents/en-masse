"""Synchronization policies.

A policy answers one question each time an agent is about to start a step:

    RUN    start now and treat the step as final (no checkpoint needed)
    SPEC   start now, speculatively (take a checkpoint; the step may be rolled back)
    WAIT   do not start yet; ask again after the next event

The engine supplies the facts a policy may use:

    engine.is_safe(agent, step)      True if no agent earlier in the serial order
                                     can still write anything this step reads, and
                                     all earlier steps of this agent are final
    engine.uncommitted(agent)        number of this agent's steps that are still
                                     speculative
    engine.all_earlier_finished(agent)

Two class attributes tell the engine how to treat the policy:

    repairs      the runtime detects stale reads and rolls agents back
    multiversion the runtime serves each read the value its serial position
                 should see, so writes by later agents are never visible to
                 earlier ones
"""

from __future__ import annotations

import math

RUN, SPEC, WAIT = "run", "spec", "wait"


class Policy:
    name = "policy"
    repairs = True
    multiversion = True

    def decide(self, engine, agent: int, step: int) -> str:
        raise NotImplementedError


class Serial(Policy):
    """One agent at a time, in serial order. The reference for correctness and cost."""

    name = "serial"

    def decide(self, engine, agent, step):
        return RUN if engine.all_earlier_finished(agent) else WAIT


class Unsynchronized(Policy):
    """All agents run freely with no synchronization. Fast and unsafe."""

    name = "unsynchronized"
    repairs = False
    multiversion = False

    def decide(self, engine, agent, step):
        return RUN


class Conservative(Policy):
    """Start a step only when it is provably safe.

    Safety is judged from declared footprints, so the quality of those
    declarations decides how much concurrency this policy can find.
    """

    name = "conservative"

    def decide(self, engine, agent, step):
        return RUN if engine.is_safe(agent, step) else WAIT


class Optimistic(Policy):
    """Always start. Repair by rollback when a read turns out to be stale.

    Irreversible effects are released as soon as their step completes, so a
    later rollback can leave an effect in the world that should not exist.
    """

    name = "optimistic"

    def decide(self, engine, agent, step):
        return RUN if engine.is_safe(agent, step) else SPEC


class Hybrid(Policy):
    """Conservative when safe, speculative otherwise, within a bound.

    window  maximum number of speculative (uncommitted) steps an agent may hold.
            0 gives the conservative policy; infinity gives optimistic execution.
    gate    if True, a step with an irreversible effect waits until it is safe,
            so irreversible effects are never rolled back.
    """

    def __init__(self, window: float = 2, gate: bool = True):
        self.window = window
        self.gate = gate
        w = "inf" if math.isinf(window) else str(int(window))
        self.name = f"hybrid(w={w})" if gate else f"hybrid(w={w},ungated)"

    def decide(self, engine, agent, step):
        if engine.is_safe(agent, step):
            return RUN
        if self.gate and engine.step(agent, step).irreversible:
            return WAIT
        if engine.uncommitted(agent) < self.window:
            return SPEC
        return WAIT


def gated_optimistic() -> Hybrid:
    """Optimistic execution that still holds irreversible effects until safe."""
    p = Hybrid(window=math.inf, gate=True)
    p.name = "optimistic+gate"
    return p
