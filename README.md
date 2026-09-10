# TraInfBench

Lightweight trajectory-inference benchmark adapters and figure-analysis code for **SyncOTML-8**.

| Directory | Contents |
| --- | --- |
| [model/](model/README.md) | TrajectoryNet, MIOFlow, CytoBridge and TIGON: training, trajectory generation and short usage guides |
| [comparison/](comparison/README.md) | Paper comparisons organized by dataset, with a figure-to-code index |
| [common/](common/README.md) | Shared plotting, evaluation and path helpers |
| [external/](external/README.md) | Where to install upstream packages; no upstream source is included |
| [data/](data/README.md) | Dataset sources and expected local inputs; no data are included |

Use Python 3.10 and install the relevant method following its README. Run commands from the repository root. Shell launchers use `PYTHON=python` by default; override `PYTHON` to use an existing environment.

```bash
python -m pip install -r requirements.txt
python common/check_repository.py
python model/MIOFlow/train_mioflow_10000.py --help
```

Start with [the figure index](comparison/FIGURES.md) to find a paper panel, then read that dataset's README. Reproduction requires the original prepared embeddings, frozen maps, trajectories or summary tables described there. These assets, checkpoints and generated figures are intentionally excluded. Public raw data alone do not replace the exact frozen benchmark representations.

The repository includes benchmark adapters and necessary local package patches. It does not include third-party source trees or COATI training code. Comparison routines may read precomputed COATI outputs; legacy optional analyses that reload COATI checkpoints require its separately obtained source under `external/COATI`.

Full training jobs can be long. Existing launcher overwrite guards are retained (`OVERWRITE=0` by default). Full/LOO representation policies, native versus conditional trajectory initialization, and TIGON adaptations are documented per method; do not exchange these protocols silently.

The local pre-organization `scripts/`, data, results and long README are preserved outside the published tree. [Source provenance](common/source_manifest.json) records each selected original file and its SHA-256. See [third-party notices](THIRD_PARTY.md).
