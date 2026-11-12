# En-Masse

**En**abling **M**ulti-**A**gent **S**ynchronization-**S**pace **E**xploration

A framework for exploring synchronization policies for multi-agent LLM
execution, from conservative (block until provably safe) to optimistic
(speculate, checkpoint and roll back).

**Status: early prototype.** The repository currently contains a simulation
engine, six policies and a policy-independent checker. It simulates agents and
does not run them. Workloads are synthetic. The numbers below show that the
instrument works and that the expected trade-offs appear. They are not
measurements of real LLM agents.

## Why

Agents that act concurrently on shared state (files, repositories, databases,
deployed services, messages) need a synchronization policy. Existing
multi-agent runtimes each implement one strategy and evaluate it on their own
workloads. The best strategy depends on the workload: its contention, how
precisely agents can declare what they will read and write, and its fraction of
irreversible actions.

Parallel discrete-event simulation (PDES) has studied this design space for
decades. En-Masse brings that body of work to agent execution, so that a user
can compare policies for a given workflow in simulation before spending tokens
on live agents.

## The model in brief

Every agent action has a **virtual time** (its position in a reference serial
order), an **effect class** (reversible, compensable or irreversible) and a
**declared footprint** (a superset of what it may read and write).

Each agent holds a **commit horizon**. A step is *safe* when no uncommitted
step with an earlier virtual time can still conflict with its footprint.

- A safe step runs irreversibly. It needs no checkpoint and its effects are
  released at once.
- For any other step, the **policy** decides. The step either waits, or runs
  speculatively behind a checkpoint with its irreversible effects withheld.

A policy is therefore a rule for advancing the horizon and bounding
speculation. This one interface expresses the whole range:

| Policy | Rule |
|---|---|
| `Serial` | One agent at a time (the reference) |
| `Conservative` | Run only safe steps |
| `Hybrid(window=w)` | Run safe steps. Otherwise speculate, holding at most `w` uncommitted steps. Hold irreversible effects until safe |
| `gated_optimistic()` | `Hybrid` with an unbounded window |
| `Optimistic` | Always run and repair by rollback. Irreversible effects are not held |
| `Unsynchronized` | Always run, with no repair |

A **policy-independent checker** reads the execution trace alone and reports
stale reads, reads of values from the future, and irreversible effects that
were released and then rolled back. The same checker judges every policy.

## Install

```
git clone git@github.com:en-masse-agents/en-masse.git
cd en-masse
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[experiments,dev]"
python -m pytest -q
```

## Quick start

```python
import numpy as np
from enmasse import generate, simulate, check, Hybrid

tasks = generate(np.random.default_rng(0), contention=0.3, footprint_slack=3)
metrics, engine = simulate(tasks, Hybrid(window=2))
print(metrics["speedup"], metrics["token_ratio"], check(engine))
```

To reproduce the results and the figure:

```
python experiments/run_sweeps.py
python experiments/plot_figure.py
```

## First results

![results](results/figure_policies.png)

Setting: 8 agents, 12 steps each, 12 shared objects, 10% of steps irreversible,
200 random workloads per point, every policy on the same workloads. Shaded
bands are 95% confidence intervals. Contention is the share of steps that touch
shared state. "Coarse footprints" means each agent declares 3 extra objects it
does not use.

Coarse footprints, contention 0.3:

| Policy | Speedup over serial | Token cost | Violations per run |
|---|---|---|---|
| Conservative | 1.94x | 1.00x | 0 |
| Hybrid, window 2 | 2.21x | 1.08x | 0 |
| Gated optimistic | 2.87x | 1.23x | 0 |
| Optimistic, not gated | 4.06x | 1.32x | 2.9 leaked irreversible effects |
| Unsynchronized | 6.54x | 1.00x | 12.2 stale or future reads |

What the results show:

1. **The policies form a frontier and not a ranking.** Each step toward
   optimism buys speed and costs tokens.
2. **Footprint precision decides whether speculation pays.** With exact
   footprints the conservative policy already finds all the concurrency there
   is (3.99x), and speculation only adds cost. With coarse footprints,
   speculation buys speed, and the window selects a point on the curve.
3. **The checker makes the price of unsafe speed visible.** Releasing
   irreversible effects without gating is faster but leaks them in 83% of
   runs. The conservative, hybrid and gated policies had no violation in any
   run.

The full safety table is in `results/summary.md`, and the raw sweeps are in
`results/sweep_contention.csv` and `results/sweep_window.csv`.

## What is modelled, and what is not

Modelled: step durations and token cost, reads and writes on shared objects,
irreversible effects, declared footprints, checkpoint cost, undo cost, rollback
with cascades, and an optional discount for repairing instead of redoing.

Not modelled yet: real agents and tools, nondeterministic re-execution, LLM
serving capacity and rate limits, messages between agents, human approval
delays, failures, and calibration against real traces.


## Layout

```
enmasse/workload.py     synthetic workload generator
enmasse/policies.py     the policies
enmasse/engine.py       discrete-event engine, commit points, rollback
enmasse/checker.py      policy-independent correctness checker
experiments/            sweeps and the figure
tests/                  unit tests
results/                CSV output, figure, safety summary
```

## Citing

If you use En-Masse, please cite it using the metadata in `CITATION.cff`
(GitHub shows a "Cite this repository" button).

## License

Code in this repository is licensed under the Apache License, Version 2.0 (see
`LICENSE`). Documentation, course material and datasets released by the project
are licensed under CC BY 4.0.

## Contact

Neha Karanjkar, Indian Institute of Technology Goa.
<https://nehakaranjkar.github.io>
