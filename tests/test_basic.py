"""Basic checks. Run with:  python -m pytest -q   (from the repository root)"""

import math

import numpy as np
import pytest

from enmasse import (
    Conservative,
    Hybrid,
    Optimistic,
    Serial,
    Unsynchronized,
    check,
    gated_optimistic,
    generate,
    simulate,
)


def workload(seed, **kw):
    return generate(np.random.default_rng(seed), **kw)


def test_serial_matches_sum_of_durations():
    tasks = workload(1, contention=0.5)
    m, eng = simulate(tasks, Serial())
    assert m["makespan"] == pytest.approx(m["serial_makespan"])
    assert m["token_ratio"] == pytest.approx(1.0)
    assert check(eng)["ok"]


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("contention", [0.0, 0.2, 0.6, 1.0])
def test_safe_policies_pass_the_checker(seed, contention):
    tasks = workload(seed, contention=contention, footprint_slack=3)
    for policy in (Serial(), Conservative(), Hybrid(window=2), gated_optimistic()):
        m, eng = simulate(tasks, policy)
        result = check(eng)
        assert result["ok"], (policy.name, result)
        assert m["leaks"] == 0


@pytest.mark.parametrize("seed", range(40))
def test_conservative_never_rolls_back(seed):
    tasks = workload(seed, contention=0.6, footprint_slack=2)
    m, _ = simulate(tasks, Conservative())
    assert m["rollbacks"] == 0
    assert m["speculative_steps"] == 0
    assert m["token_ratio"] == pytest.approx(1.0)


@pytest.mark.parametrize("seed", range(20))
def test_hybrid_window_zero_is_conservative(seed):
    tasks = workload(seed, contention=0.4, footprint_slack=3)
    a, _ = simulate(tasks, Conservative())
    b, _ = simulate(tasks, Hybrid(window=0))
    assert a["makespan"] == pytest.approx(b["makespan"])
    assert a["tokens"] == pytest.approx(b["tokens"])


def test_no_contention_gives_full_concurrency():
    tasks = workload(3, contention=0.0, irreversible_prob=0.0)
    longest = max(sum(s.duration for s in t.steps) for t in tasks)
    for policy in (Conservative(), Optimistic(), Hybrid(window=math.inf)):
        m, _ = simulate(tasks, policy)
        assert m["makespan"] == pytest.approx(longest)


def test_checker_flags_unsynchronized_runs():
    flagged = 0
    for seed in range(40):
        tasks = workload(seed, contention=0.8)
        _, eng = simulate(tasks, Unsynchronized())
        if not check(eng)["ok"]:
            flagged += 1
    assert flagged > 30


def test_ungated_optimism_can_leak_irreversible_effects():
    leaks = 0
    for seed in range(40):
        tasks = workload(seed, contention=0.6, irreversible_prob=0.3)
        m, eng = simulate(tasks, Optimistic())
        result = check(eng)
        assert result["stale_reads"] == 0          # repair keeps reads consistent
        assert result["leaked_irreversible"] == m["leaks"]
        leaks += m["leaks"]
    assert leaks > 0


def test_runs_are_deterministic():
    tasks = workload(7, contention=0.5, footprint_slack=2)
    a, _ = simulate(tasks, Hybrid(window=3))
    b, _ = simulate(tasks, Hybrid(window=3))
    assert a == b
