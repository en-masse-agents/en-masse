"""Run the two parameter sweeps behind the results figure.

Usage (from the repository root):

    python experiments/run_sweeps.py            # 200 workloads per setting
    python experiments/run_sweeps.py --seeds 50 # quicker

Every policy is run on the same set of random workloads (paired comparison),
and every run is passed through the policy-independent checker.

Outputs (CSV, in results/):
    sweep_contention.csv   metrics against contention, coarse footprints
    sweep_window.csv       metrics against the speculation window, for three
                           levels of footprint quality
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from enmasse import (
    Conservative,
    Hybrid,
    Optimistic,
    Unsynchronized,
    check,
    gated_optimistic,
    generate,
    simulate,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"

WORKLOAD = dict(n_agents=8, n_steps=12, n_objects=12, write_prob=0.5, irreversible_prob=0.1)
ENGINE = dict(checkpoint_cost=0.5, undo_cost=1.0, redo_factor=1.0)

METRICS = ["speedup", "token_ratio", "rollbacks", "violations", "stale_reads",
           "future_reads", "leaked_irreversible", "unsafe_run"]


def run_setting(policies, n_seeds, **workload_overrides):
    """Return {policy name: {metric: (mean, standard error)}} over n_seeds workloads."""
    samples: dict[str, list[list[float]]] = {}
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        tasks = generate(rng, **{**WORKLOAD, **workload_overrides})
        for make in policies:
            policy = make()
            metrics, engine = simulate(tasks, policy, **ENGINE)
            verdict = check(engine)
            row = [
                metrics["speedup"],
                metrics["token_ratio"],
                metrics["rollbacks"],
                verdict["violations"],
                verdict["stale_reads"],
                verdict["future_reads"],
                verdict["leaked_irreversible"],
                0.0 if verdict["ok"] else 1.0,
            ]
            samples.setdefault(policy.name, []).append(row)
    out = {}
    for name, rows in samples.items():
        arr = np.asarray(rows)
        mean = arr.mean(axis=0)
        sem = arr.std(axis=0, ddof=1) / math.sqrt(len(arr))
        out[name] = {m: (float(mean[i]), float(sem[i])) for i, m in enumerate(METRICS)}
    return out


def write_csv(path: Path, key_fields: list[str], records: list[dict]) -> None:
    fields = key_fields + ["policy", "n"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "sem")]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def flatten(keys: dict, n: int, result: dict) -> list[dict]:
    records = []
    for name, metrics in result.items():
        rec = {**keys, "policy": name, "n": n}
        for m, (mean, sem) in metrics.items():
            rec[f"{m}_mean"] = round(mean, 6)
            rec[f"{m}_sem"] = round(sem, 6)
        records.append(rec)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=200)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    # Sweep 1: contention, with coarse footprints (3 falsely declared objects per agent)
    policies = [
        Conservative,
        lambda: Hybrid(window=2),
        gated_optimistic,
        Optimistic,
        Unsynchronized,
    ]
    records = []
    for contention in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        result = run_setting(policies, args.seeds, contention=contention, footprint_slack=3)
        records += flatten({"contention": contention, "footprint_slack": 3}, args.seeds, result)
        print(f"contention={contention:<4}  " + "  ".join(
            f"{k}: x{v['speedup'][0]:.2f}" for k, v in result.items()))
    write_csv(RESULTS / "sweep_contention.csv", ["contention", "footprint_slack"], records)

    # Sweep 2: speculation window, for three levels of footprint quality
    records = []
    windows = [0, 1, 2, 4, 8, math.inf]
    for slack in [0, 3, 6]:
        policies = [(lambda w=w: Hybrid(window=w)) for w in windows]
        result = run_setting(policies, args.seeds, contention=0.3, footprint_slack=slack)
        for w, (name, metrics) in zip(windows, result.items()):
            records += flatten(
                {"contention": 0.3, "footprint_slack": slack,
                 "window": "inf" if math.isinf(w) else w},
                args.seeds, {name: metrics})
        print(f"slack={slack}  " + "  ".join(
            f"{k}: x{v['speedup'][0]:.2f}/{v['token_ratio'][0]:.2f}" for k, v in result.items()))
    write_csv(RESULTS / "sweep_window.csv", ["contention", "footprint_slack", "window"], records)
    print(f"wrote CSV files to {RESULTS}")


if __name__ == "__main__":
    main()
