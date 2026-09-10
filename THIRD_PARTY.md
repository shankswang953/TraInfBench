# Third-party provenance

Upstream source code, example datasets and checkpoints are not vendored. Exact source revisions are recorded in [external/versions.json](external/versions.json).

- TrajectoryNet and MIOFlow: Krishnaswamy Lab. The small local benchmark patches retain the respective upstream notices in `model/TrajectoryNet/UPSTREAM_LICENSE.txt` and `model/MIOFlow/UPSTREAM_LICENSE.txt`.
- CytoBridge: zhenyiizhang/CytoBridge. The local patch is covered by the upstream GPLv3 notice in `model/CytoBridge/UPSTREAM_LICENSE.txt`.
- TIGON: yutongo/TIGON. `model/TIGON/run_tigon_moscot_ae.py` and its helpers are the benchmark's adapted/vectorized implementation; upstream MIT attribution is retained in `model/TIGON/UPSTREAM_LICENSE.txt`.
- COATI/scMultiSim manuscript plots copied from the author's adjacent research workspace are identified in `common/source_manifest.json`. COATI core code, training and data are not distributed here.

No repository-wide license is asserted over third-party components. Consult each linked project and retained notice for its terms.
