# Optional manuscript assets

Place the original TraInf **inner project** here (the directory containing `Gastrulation/`, `MouseBrain/`, `moscot/`, `humanCerebral/`, `Synthetic/`, `GaussToy/`, `ToySplit/` and optional `src/`). Alternatively create a local symlink at this path before restoring assets. The contents are ignored by Git.

This release does not publish or train COATI. Its cached predictions and frozen RNA-to-ATAC maps are external inputs. Obtain the manuscript's exact prepared assets from the author; no public download URL is assumed. Public biological dataset links are in `data/README.md`.

Plot-only table readers need only their listed tables. Some historical analysis and toy visualization scripts reload pretrained maps or vector fields, so those optional scripts additionally require the corresponding original source modules and checkpoints. Do not treat them as a standalone COATI implementation.
