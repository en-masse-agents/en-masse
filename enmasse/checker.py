"""Policy-independent correctness checker.

The checker looks only at the trace an engine leaves behind (the list of step
executions). It does not use any policy logic, so the same checker judges every
policy, and could judge a trace recorded from real agents.

It checks four things.

completeness
    Every step of every agent has exactly one valid, completed execution.

causality (stale reads)
    A valid read must not be followed by a write to the same object from an
    earlier-ranked agent. If it is, the reader acted on a value that the serial
    order says it should not have seen.

causality (future reads), only when the runtime has no versioned reads
    A valid read must not be preceded by a write to the same object from a
    later-ranked agent.

commit safety
    No irreversible effect may have been released by an execution that was
    later rolled back.

What it does not check: the final value of each object (write-write order),
and anything about the quality of the agents' output.
"""

from __future__ import annotations

from .engine import Engine


def check(engine: Engine) -> dict:
    tasks = engine.tasks
    valid = [e for e in engine.executions if e.valid and e.completed]

    # completeness
    seen: dict[tuple[int, int], int] = {}
    for e in valid:
        seen[(e.agent, e.step)] = seen.get((e.agent, e.step), 0) + 1
    incomplete = 0
    for j, t in enumerate(tasks):
        for s in range(len(t.steps)):
            if seen.get((j, s), 0) != 1:
                incomplete += 1

    # collect valid reads (at step start) and valid writes (at step end), per object
    reads: dict[int, list[tuple[int, int]]] = {}    # obj -> [(agent, stamp)]
    writes: dict[int, list[tuple[int, int]]] = {}
    for e in valid:
        step = tasks[e.agent].steps[e.step]
        for o in step.reads:
            reads.setdefault(o, []).append((e.agent, e.start_stamp))
        for o in step.writes:
            writes.setdefault(o, []).append((e.agent, e.end_stamp))

    stale_reads = 0
    future_reads = 0
    for o, rs in reads.items():
        ws = writes.get(o, [])
        for reader, r_stamp in rs:
            if any(w < reader and w_stamp > r_stamp for w, w_stamp in ws):
                stale_reads += 1
            if not engine.policy.multiversion and any(
                w > reader and w_stamp < r_stamp for w, w_stamp in ws
            ):
                future_reads += 1

    # commit safety
    leaked = sum(1 for e in engine.executions if e.released and not e.valid)

    violations = incomplete + stale_reads + future_reads + leaked
    return {
        "incomplete_steps": incomplete,
        "stale_reads": stale_reads,
        "future_reads": future_reads,
        "leaked_irreversible": leaked,
        "violations": violations,
        "ok": violations == 0,
    }
