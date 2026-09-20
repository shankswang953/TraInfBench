# Trajectory methods

| Method | Implementation in this repository | Guide |
| --- | --- | --- |
| TrajectoryNet | Calls the upstream CNF package; seed/input/time and evaluation adapters | [Usage](TrajectoryNet/README.md) |
| MIOFlow | Calls upstream GAGA and MIOFlow APIs; frozen normalization and decoder adapters | [Usage](MIOFlow/README.md) |
| CytoBridge | Calls upstream velocity/growth training APIs; balanced/unbalanced benchmark runners | [Usage](CytoBridge/README.md) |
| TIGON | Local vectorized implementation based on pinned public TIGON, with equivalence checks | [Usage](TIGON/README.md) |
| scMultiNODE | Calls upstream AE/QGW/fusion/ODE; signed-input and native-grid trajectory adapters; four dataset protocols | [Usage](scMultiNODE/README.md) |

Input data and checkpoints are external. Use each guide's current entrypoints rather than guessing from historical iteration counts in filenames. No COATI training is included. Package patches are the exact local changes used by the research environment; see each guide before applying them.
