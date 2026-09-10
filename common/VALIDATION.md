# Release validation (2026-09-10)

- Parsed all 220 Python/shell/R source entries: Python AST and Bash syntax checks passed (R script inventoried; not executed).
- Verified all selected source-manifest and 40 figure-group targets exist; all README links resolve locally.
- Imported the TrajectoryNet, MIOFlow, CytoBridge and TIGON command-line entrypoints with `--help` in the documented shared Python 3.10 environment.
- TIGON equivalence check against public commit `1ed92cfcc250415fc01b4d344a308b0680cc9635` passed. Seeded AE/network outputs match exactly; action maximum absolute difference was 2.38e-7 and KDE difference was 1.12e-8 (tolerance 1e-6).
- Applied each TrajectoryNet/MIOFlow/CytoBridge patch to temporary copies of its recorded upstream revision. The resulting Python files match the existing local benchmark installation byte-for-byte.
- Replayed the scMultiSim terminal comparison, palate external-peak contrast table and human three-stage mass-allocation table; inspected the images against the paper panels. Outputs were saved only in a temporary directory.
- Exported only the proposed publishable files into a clean directory; repository checks and the TIGON trainer help entrypoint passed without the original `scripts/` tree.
- Confirmed raw data, result directories, checkpoints, upstream checkouts and the historical workspace scripts are ignored. Release source/documentation is approximately 3.1 MB before Git compression.

No full training jobs were started. Full scientific reproduction of every figure is not claimed: it requires the prepared datasets, frozen maps, trajectories and summary tables documented per dataset. Some original plotters evaluate pretrained COATI fields and need the separately provided source; no COATI training code is shipped.
