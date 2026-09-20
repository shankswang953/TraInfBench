"""Frozen dataset contracts for the shared scMultiNODE benchmark runner.

Dataset-specific representation names and physical clocks live here. Changing
datasets never changes upstream objectives, sampling, architecture or solvers.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetProtocol:
    name: str
    data_subdirectory: str
    input_space: str
    times: tuple[float, ...]
    stages: tuple[str, ...]
    keys: tuple[str, ...]
    # modality, source matrix, frozen scalar file, dimension
    modalities: tuple[tuple[str, str, str, int], ...]
    splits: dict[str, list[int]]
    representation_policy: str


GASTRULATION = DatasetProtocol(
    name="gastrulation", data_subdirectory="Gastrulation/data",
    input_space="frozen_normalized_rna50_atac14",
    times=(0., 1., 2., 2.5), stages=("E7.5", "E8.0", "E8.5", "E8.75"),
    keys=("time0", "time1", "time2", "time3"),
    modalities=(("rna", "rna_pca_by_time.npz", "primal_norm_params.pt", 50),
                ("atac", "atac_lsi_by_time_14D.npz", "secondary_norm_params.pt", 14)),
    splits={"full": [0, 1, 2, 3], "loo1": [0, 2, 3], "loo2": [0, 1, 3]},
    representation_policy="Frozen common all-data PCA/LSI/scalars; no held-out cells in any scMultiNODE training stage. Not end-to-end inductive raw-data LOO.",
)

PANCREAS = DatasetProtocol(
    name="moscot_pancreas", data_subdirectory="moscot/data",
    input_space="frozen_normalized_rna50_atac_poissonvi22",
    times=(0., 1., 2.), stages=("E14.5", "E15.5", "E16.5"),
    keys=("time_0", "time_1", "time_2"),
    modalities=(("rna", "rna_pca_by_time.npz", "primal_norm_params.pt", 50),
                ("atac", "atac_poissonvi_time_data.npz", "secondary_norm_params_poissonvi.pt", 22)),
    splits={"full": [0, 1, 2], "loo1": [0, 2]},
    representation_policy="Frozen common all-data RNA PCA50 / ATAC PoissonVI22 and scalar normalization; no held-out cells in any scMultiNODE AE, QGW, fusion or dynamics training stage. Not end-to-end inductive raw-data LOO.",
)


HUMAN_CEREBRAL = DatasetProtocol(
    name="human_cerebral_7time_d4_d21_no_d16",
    data_subdirectory="humanCerebral/Data/selected_4_7_9_11_12_18_21",
    input_space="frozen_normalized_rna30_atac_lsi12",
    times=(0., .3, .5, .7, .8, 1.4, 1.7),
    stages=("D4", "D7", "D9", "D11", "D12", "D18", "D21"),
    keys=("age_4", "age_7", "age_9", "age_11", "age_12", "age_18", "age_21"),
    modalities=(("rna", "rna_pca30_by_time.npz", "primal_norm_params.pt", 30),
                ("atac", "atac_lsi12_by_time.npz", "secondary_norm_params_lsi12.pt", 12)),
    splits={"full": [0, 1, 2, 3, 4, 5, 6]},
    representation_policy="Same seven-time normalized RNA PCA30 and unique computationally paired metacells as current balanced CytoBridge, plus their matched normalized ATAC LSI12. Frozen source representations/scalars; all seven selected times observed. No heldout; no biological-mass row replication or paired supervision.",
)


PALATE = DatasetProtocol(
    name="palate", data_subdirectory="MouseBrain/data",
    input_space="frozen_normalized_rna40_atac_lsi15",
    times=(0., 1., 1.5, 2.), stages=("E12.5", "E13.5", "E14.0", "E14.5"),
    keys=("time_0", "time_1", "time_2", "time_3"),
    modalities=(("rna", "rna_pca_by_time.npz", "primal_norm_params.pt", 40),
                ("atac", "atac_lsi15_by_time.npz", "secondary_norm_params_lsi15.pt", 15)),
    splits={"full": [0, 1, 2, 3], "loo1": [0, 2, 3], "loo2": [0, 1, 3]},
    representation_policy="Same observed cells and frozen RNA PCA40 / ATAC LSI15 as the palate benchmark, divided once by its common scalar normalization. CytoBridge trains in raw RNA coordinates and is evaluated after the same scaling. Held-out cells excluded before all scMultiNODE AE, QGW, fusion and dynamics training; frozen upstream representations/scalars mean conditional, not raw-data inductive, LOO. No pairing or cell-type supervision.",
)
