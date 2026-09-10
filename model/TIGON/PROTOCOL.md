# TIGON Gastrulation reproduction protocol v2

## Status and scope

This protocol is the publication-facing TIGON implementation for the
Gastrulation Full, LOO-time1, and LOO-time2 tasks.  Earlier runs labelled
`official_stationary`, `paper_fixed_sample`, `safe_reproduction_v1`, or a
sigma-floor experiment are diagnostics and must not be presented as the
literal public-code reproduction.

The reference implementation is the unmodified public TIGON repository at
commit `1ed92cfcc250415fc01b4d344a308b0680cc9635`, frozen under
`external/TIGON_upstream_1ed92cf`.

## Representation

- Select the 3,000 highly variable genes recorded in the complete processed
  Gastrulation dataset.
- Use the non-negative log-normalized expression matrix without per-gene
  z-scoring.  This follows the data actually loaded by TIGON's public AE
  notebook; its distributed `AE_EMT_normalized.csv` is not zero-mean or
  unit-variance despite a contradictory trainer docstring.
- Use the public AE architecture and settings: 3000-300-10-300-3000, ReLU,
  batch normalization, dropout 0.2, seed 4232, batch size 128, Adam learning
  rate 1e-3, weight decay 1e-4, and the public early-stopping rule.
- Reset Torch RNG to seed 4232 after model/DataLoader construction, matching
  the public `Trainer` before its first shuffled epoch.
- Freeze one full-data AE for Full and LOO.  The LOO trajectory model never
  sees held-out-stage cells, but the representation is transductive.  This is
  disclosed and applied consistently across the benchmark.
- Scale each frozen AE coordinate to [-2, 2] using complete-dataset statistics.
  This is a declared dataset-specific coordinate normalization, not a feature
  of the public EMT notebook.  It places the 10D coordinates on the scale of
  TIGON's released EMT input and is fitted once, never refitted for LOO.

## Trajectory model and objective

The v2 runner uses the public TIGON choices:

- velocity network: four 16-unit Tanh hidden layers;
- growth network: three 16-unit Tanh hidden layers;
- public Xavier weight initialization with constructor-initialized biases;
- public Python-RNG center sampling and Torch Gaussian noise, covariance 0.02;
- checkpoint and restore Torch, Python, and NumPy RNG states for exact resume;
- short- and long-term raw KDE-density MSE, multiplied by 1e4;
- exact divergence in the continuity equation;
- public TorchDiffEqPack Dopri5 with rtol 1e-3 and atol 1e-5;
- public midpoint action integration;
- literal public `trans_loss` semantics, including its nested growth integral
  evaluated at zero spatial state;
- Adam learning rate 3e-3 and weight decay 0.01;
- the public conditional sigma schedule: start at 1, test `sigma > 0.02`, and
  halve without a post-update floor when the long-density threshold is met.

The benchmark changes `num_samples` from the public EMT default 100 to 1024
and training length from 5,000 to 20,000 iterations to match the comparison
budget requested for all methods.  These alter compute, not the TIGON loss.

## Checkpoint and evaluation policy

- Save checkpoints every 500 iterations and at every sigma-annealing boundary,
  then select the checkpoint with the lowest mean forward distribution
  discrepancy over observed training stages only.
- Never use a held-out LOO stage for checkpoint selection.
- The primary direction is forward: sample the initial TIGON KDE (covariance
  0.02), then push initial particles to each later time, especially terminal.
- Backward later-to-initial integration is only a numerical diagnostic and is
  not a benchmark prediction.
- Report native AE10 metrics as a method-space diagnostic.
- Decode through the frozen AE and frozen expression-to-common-PCA50 bridge for
  the primary cross-method metrics.  The bridge is evaluation-only and is not
  used to train TIGON.
- Report both the strict final checkpoint and the pre-specified observed-stage
  selected checkpoint so bandwidth-induced late-training degradation is
  visible rather than hidden.

## Independent checks

`model/TIGON/check_tigon_upstream_public_equivalence.py` compares the benchmark
primitives with the frozen public source.  It requires exact agreement for the
seeded AE, trajectory network, velocity, growth, and KDE sampling, and numerical
agreement for the action integrand and vectorized KDE.

The public EMT 5-step tutorial is also rerun before Gastrulation.  With the
public RNG, initialization, and ODE backend, the local first loss is 850.128,
the same as the unmodified public run; the five-step endpoints are 75.852 and
75.871, respectively.

## Claims that are safe to make

Use “a numerically verified, vectorized reimplementation of public TIGON commit
`1ed92cf...`, with a TIGON-style 10D AE and declared coordinate normalization.”
Do not write “unmodified TIGON was run directly on Gastrulation,” because the
vectorized KDE, dataset-specific representation, 1024-sample budget, and
observed-stage checkpoint rule are explicit benchmark adaptations.

If a fixed sigma or sigma floor is reported, label it as a bandwidth-sensitivity
or stabilized adaptation.  It must not replace the no-floor v2 run without
being identified as a hyperparameter modification.
