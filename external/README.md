# External software (not included)

Follow each project's installation instructions. Clone into the following local paths if using source installations; Git ignores all checkout contents.

| Method | Upstream | Local checkout |
| --- | --- | --- |
| TrajectoryNet | https://github.com/KrishnaswamyLab/TrajectoryNet | `external/TrajectoryNet` |
| MIOFlow | https://github.com/KrishnaswamyLab/MIOFlow | `external/MIOFlow` |
| CytoBridge | https://github.com/zhenyiizhang/CytoBridge | `external/CytoBridge` |
| TIGON | https://github.com/yutongo/TIGON | `external/TIGON_upstream_1ed92cf` |

For the exact benchmark, check out the revision in [versions.json](versions.json), apply the method's `benchmark.patch` if provided, and follow the upstream installation instructions. The patch contains only local changes, not a copy of the package. Do not apply it twice.

```bash
# Example, run from repository root after cloning MIOFlow:
git -C external/MIOFlow checkout ec4b8ba06784aeba0f6e2be6c6204a8e806e1c77
git -C external/MIOFlow apply --check ../../model/MIOFlow/benchmark.patch
git -C external/MIOFlow apply ../../model/MIOFlow/benchmark.patch
python -m pip install -e external/MIOFlow
```

TIGON's benchmark implementation is in `model/TIGON`, separate from the official checkout used for equivalence checks. The exact solver dependency is installed locally with:

```bash
python -m pip install --target external/TIGON_upstream_1ed92cf/_deps TorchDiffEqPack==1.0.1
```

Optional COATI manuscript assets go under [COATI/](COATI/README.md). `external/COATI_WORKSPACE` denotes the outer legacy workspace for the few original scripts that refer to outer-workspace assets; inspect the dataset asset inventory before running those scripts.
