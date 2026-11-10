# Safety summary (from the policy-independent checker)

Mean violations per run, 200 workloads per setting, coarse footprints.

| contention | policy | stale or future reads | leaked irreversible effects | runs with any violation |
|---|---|---|---|---|
| 0.1 | conservative | 0.00 | 0.00 | 0% |
| 0.1 | hybrid(w=2) | 0.00 | 0.00 | 0% |
| 0.1 | optimistic+gate | 0.00 | 0.00 | 0% |
| 0.1 | optimistic | 0.00 | 0.24 | 18% |
| 0.1 | unsynchronized | 1.35 | 0.00 | 60% |
| 0.3 | conservative | 0.00 | 0.00 | 0% |
| 0.3 | hybrid(w=2) | 0.00 | 0.00 | 0% |
| 0.3 | optimistic+gate | 0.00 | 0.00 | 0% |
| 0.3 | optimistic | 0.00 | 2.89 | 83% |
| 0.3 | unsynchronized | 12.15 | 0.00 | 100% |
| 0.6 | conservative | 0.00 | 0.00 | 0% |
| 0.6 | hybrid(w=2) | 0.00 | 0.00 | 0% |
| 0.6 | optimistic+gate | 0.00 | 0.00 | 0% |
| 0.6 | optimistic | 0.00 | 8.05 | 98% |
| 0.6 | unsynchronized | 39.73 | 0.00 | 100% |
