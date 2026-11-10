"""Discrete-event engine that runs a workload under a synchronization policy.

Reference semantics
-------------------
Agents are ranked. A run is correct if it is equivalent to executing the agents
one after another in rank order (the serial order). Concretely, every read must
observe exactly the writes of earlier-ranked agents, all of them and nothing
else.

Commit point
------------
Each agent has a commit point: the index of its first step that is not yet
final. A step is *safe* when

  (a) every earlier step of the same agent is final, and
  (b) no unfinished earlier-ranked agent can still write an object the step
      reads. "Can still write" is judged from what that agent has declared:
      the writes of its not-yet-final steps plus its declared-but-unused
      objects.

A safe step can never be invalidated, so it needs no checkpoint and its
irreversible effects can be released at once. Steps that are not safe may
still be executed speculatively if the policy allows it.

Repair
------
When an agent completes a write, every later-ranked agent that has already
read that object holds a stale value. If the policy repairs, those agents are
rolled back to the stale step. Rolling back undoes the agent's later writes,
which can in turn invalidate agents ranked after it (a cascade).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from .policies import RUN, SPEC, WAIT, Policy
from .workload import AgentTask, Step


@dataclass
class Execution:
    """One attempt at one step. A step may be attempted several times."""

    exec_id: int
    agent: int
    step: int
    start: float
    start_stamp: int
    duration: float
    tokens: float
    speculative: bool
    end: float | None = None
    end_stamp: int | None = None
    completed: bool = False
    valid: bool = True          # False once rolled back or cancelled
    released: bool = False      # an irreversible effect was released to the world


class _AgentState:
    def __init__(self, task: AgentTask):
        self.task = task
        self.n = len(task.steps)
        self.pc = 0                       # number of completed, currently valid steps
        self.commit = 0                   # index of the first step that is not final
        self.running: Execution | None = None
        self.epoch = 0                    # bumps on rollback; stale completion events are ignored
        self.blocked_until = 0.0
        self.step_exec: list[int | None] = [None] * self.n
        self.done_before: list[bool] = [False] * self.n
        # suffix_writes[s] = objects written by steps s, s+1, ...
        self.suffix_writes: list[frozenset[int]] = [frozenset()] * (self.n + 1)
        acc: frozenset[int] = frozenset()
        for s in range(self.n - 1, -1, -1):
            acc = acc | task.steps[s].writes
            self.suffix_writes[s] = acc

    @property
    def finished(self) -> bool:
        return self.commit == self.n


class Engine:
    def __init__(
        self,
        tasks: list[AgentTask],
        policy: Policy,
        checkpoint_cost: float = 0.5,
        undo_cost: float = 1.0,
        redo_factor: float = 1.0,
    ):
        """
        checkpoint_cost  seconds added to every speculative step (state saving)
        undo_cost        seconds an agent is delayed per undone step that had written
        redo_factor      cost of re-executing a step relative to its first execution.
                         1.0 means full redo; a smaller value models an agent that
                         repairs its earlier work instead of repeating it.
        """
        self.tasks = sorted(tasks, key=lambda t: t.rank)
        self.policy = policy
        self.checkpoint_cost = checkpoint_cost
        self.undo_cost = undo_cost
        self.redo_factor = redo_factor

        self.agents = [_AgentState(t) for t in self.tasks]
        self.now = 0.0
        self._stamp = 0
        self._seq = 0
        self._heap: list[tuple[float, int, str, int, int]] = []

        # trace, consumed by the checker
        self.executions: list[Execution] = []

        # metrics
        self.tokens = 0.0
        self.wasted_tokens = 0.0
        self.rollbacks = 0
        self.speculative_steps = 0
        self.leaks = 0                    # irreversible effects released, then rolled back

    # ---- facts offered to policies -------------------------------------------------

    def step(self, agent: int, step: int) -> Step:
        return self.agents[agent].task.steps[step]

    def uncommitted(self, agent: int) -> int:
        a = self.agents[agent]
        return a.pc - a.commit

    def all_earlier_finished(self, agent: int) -> bool:
        return all(self.agents[i].finished for i in range(agent))

    def _may_still_write(self, agent: int) -> frozenset[int]:
        a = self.agents[agent]
        if a.finished:
            return frozenset()
        return a.suffix_writes[a.commit] | a.task.declared_extra

    def _reads_are_final(self, agent: int, step: int) -> bool:
        reads = self.step(agent, step).reads
        if not reads:
            return True
        for i in range(agent):
            if reads & self._may_still_write(i):
                return False
        return True

    def is_safe(self, agent: int, step: int) -> bool:
        return self.agents[agent].commit == step and self._reads_are_final(agent, step)

    # ---- event loop ----------------------------------------------------------------

    def run(self) -> dict:
        self._dispatch()
        while self._heap:
            time, _, kind, agent, epoch = heapq.heappop(self._heap)
            self.now = time
            if kind == "done":
                a = self.agents[agent]
                if epoch != a.epoch or a.running is None:
                    continue              # the step was cancelled by a rollback
                self._complete(agent)
            self._advance_commits()
            self._dispatch()
        if not all(a.pc == a.n for a in self.agents):
            raise RuntimeError("simulation stalled before all agents finished")
        self._advance_commits()
        baseline = sum(s.tokens for t in self.tasks for s in t.steps)
        serial_time = sum(s.duration for t in self.tasks for s in t.steps)
        return {
            "policy": self.policy.name,
            "makespan": self.now,
            "serial_makespan": serial_time,
            "speedup": serial_time / self.now,
            "tokens": self.tokens,
            "serial_tokens": baseline,
            "token_ratio": self.tokens / baseline,
            "wasted_tokens": self.wasted_tokens,
            "rollbacks": self.rollbacks,
            "speculative_steps": self.speculative_steps,
            "leaks": self.leaks,
        }

    def _push(self, time: float, kind: str, agent: int, epoch: int) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (time, self._seq, kind, agent, epoch))

    def _next_stamp(self) -> int:
        self._stamp += 1
        return self._stamp

    def _dispatch(self) -> None:
        for j, a in enumerate(self.agents):
            if a.running is not None or a.pc == a.n:
                continue
            if a.blocked_until > self.now:
                self._push(a.blocked_until, "wake", j, a.epoch)
                continue
            decision = self.policy.decide(self, j, a.pc)
            if decision == WAIT:
                continue
            self._start(j, speculative=(decision == SPEC))

    def _start(self, agent: int, speculative: bool) -> None:
        a = self.agents[agent]
        s = a.pc
        step = a.task.steps[s]
        factor = self.redo_factor if a.done_before[s] else 1.0
        duration = step.duration * factor + (self.checkpoint_cost if speculative else 0.0)
        ex = Execution(
            exec_id=len(self.executions),
            agent=agent,
            step=s,
            start=self.now,
            start_stamp=self._next_stamp(),
            duration=duration,
            tokens=step.tokens * factor,
            speculative=speculative,
        )
        self.executions.append(ex)
        a.running = ex
        if speculative:
            self.speculative_steps += 1
        self._push(self.now + duration, "done", agent, a.epoch)

    def _complete(self, agent: int) -> None:
        a = self.agents[agent]
        ex = a.running
        assert ex is not None
        step = a.task.steps[ex.step]
        ex.completed = True
        ex.end = self.now
        ex.end_stamp = self._next_stamp()
        ex.released = step.irreversible
        a.running = None
        a.step_exec[ex.step] = ex.exec_id
        a.done_before[ex.step] = True
        a.pc += 1
        self.tokens += ex.tokens
        if step.writes and self.policy.repairs:
            self._invalidate_readers(agent, step.writes)

    # ---- repair --------------------------------------------------------------------

    def _first_dependent_step(self, agent: int, objs: frozenset[int] | set[int]) -> int | None:
        """Index of the earliest valid step of ``agent`` that read any of ``objs``."""
        a = self.agents[agent]
        for idx in range(a.pc):
            if a.task.steps[idx].reads & objs:
                return idx
        if a.running is not None and a.task.steps[a.pc].reads & objs:
            return a.pc
        return None

    def _invalidate_readers(self, writer: int, objs: frozenset[int] | set[int]) -> None:
        for k in range(writer + 1, len(self.agents)):
            idx = self._first_dependent_step(k, objs)
            if idx is not None:
                self._rollback(k, idx)

    def _rollback(self, agent: int, to_step: int) -> None:
        a = self.agents[agent]
        if to_step < a.commit:
            raise RuntimeError(
                "a final step was invalidated: the declared footprints were not sound"
            )
        self.rollbacks += 1
        if a.running is not None:
            ex = a.running
            partial = ex.tokens * min(1.0, (self.now - ex.start) / ex.duration)
            self.tokens += partial
            self.wasted_tokens += partial
            ex.valid = False
            ex.end = self.now
            a.running = None
        a.epoch += 1
        undone: set[int] = set()
        undone_writes = 0
        for idx in range(a.pc - 1, to_step - 1, -1):
            ex = self.executions[a.step_exec[idx]]
            ex.valid = False
            a.step_exec[idx] = None
            step = a.task.steps[idx]
            self.wasted_tokens += ex.tokens
            if step.writes:
                undone |= step.writes
                undone_writes += 1
            if ex.released:
                self.leaks += 1
        a.pc = to_step
        a.blocked_until = max(a.blocked_until, self.now + self.undo_cost * undone_writes)
        if undone:
            self._invalidate_readers(agent, undone)

    # ---- commit --------------------------------------------------------------------

    def _advance_commits(self) -> None:
        changed = True
        while changed:
            changed = False
            for j, a in enumerate(self.agents):
                while a.commit < a.pc and self._reads_are_final(j, a.commit):
                    a.commit += 1
                    changed = True


def simulate(tasks: list[AgentTask], policy: Policy, **engine_kwargs) -> tuple[dict, Engine]:
    """Run one workload under one policy. Returns (metrics, engine)."""
    engine = Engine(tasks, policy, **engine_kwargs)
    metrics = engine.run()
    return metrics, engine
