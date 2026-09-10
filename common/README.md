# Shared helpers

`benchmark_runtime.py` configures paths for the reorganized standalone scripts. `trainfbench_plot_style.py` supplies the shared method colors and publication style. Other files support shared input preparation, trajectory evaluation and UMAP projection.

`source_manifest.json` records selected source files and original hashes. `check_repository.py` validates Python/shell syntax, local dependencies, figure targets and the absence of large/data files without starting training.

Run `python common/check_repository.py` from the repository root. For scientific validation, use the per-method setup/equivalence checks with the required dependencies installed.
