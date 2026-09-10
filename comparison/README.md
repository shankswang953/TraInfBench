# Comparisons by dataset

Start with [FIGURES.md](FIGURES.md), which maps the supplied PDF's main and supplementary panels to the selected scripts. The machine-readable mapping is [figure_manifest.json](figure_manifest.json).

- [Gaussian bump](gaussian_bump/README.md): Fig. 2a.
- [Bifurcation](bifurcation/README.md): Fig. 2b and S1a.
- [scMultiSim](scmultisim/README.md): Fig. 2c, S1 and S2.
- [Gastrulation](gastrulation/README.md): Figs. 3-4, S3-S8 and mapping references.
- [Pancreas](pancreas/README.md): Fig. 5, S9-S12 and mapping references.
- [Palate](palate/README.md): Fig. 6, S13-S15 and mapping references.
- [Human cerebral organoids](human_cerebral/README.md): Fig. 7, S16 and mapping references.
- [Across datasets](multi_dataset/README.md): S17-S18.

The folders contain paper-facing plot entrypoints and the helpers they import. They do not contain raw data, checkpoint binaries or regenerated figures. Restore the matching summary tables/trajectory caches before plotting. Scripts called `evaluate_*`, `analyze_*` or `recompute_*` may additionally need fitted external models and frozen mappings. They are not a request to retrain COATI; this release does not provide its training stack.

Run from the repository root, for example:

```bash
python comparison/scmultisim/plot_synthetic_truth_method_columns_a4.py --help
python comparison/palate/plot_palate_external_peak_contrast_table.py --help
```

Use a new output directory when replaying a plot. Existing training overwrite guards are preserved; some historical plotting scripts overwrite their requested figure filenames.
