#!/usr/bin/env python
"""Numerically verify vectorized TIGON primitives against public commit code."""

from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tigon_official_ae_common import OFFICIAL_SOURCE_COMMIT, load_tigon_module


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UPSTREAM = ROOT / "external" / "TIGON_upstream_1ed92cf"
DEFAULT_OUTPUT = ROOT / "results" / "tigon_upstream_public_equivalence"


class PublicModelAdapter(nn.Module):
    def __init__(self, public_model: nn.Module):
        super().__init__()
        self.public_model = public_model

    def velocity(self, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.public_model.hyper_net1(time, state)

    def growth(self, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.public_model.hyper_net2(time, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    upstream_dir = args.upstream_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(upstream_dir / "_deps"))
    sys.path.insert(0, str(upstream_dir))
    import utility as upstream
    from AE.models import AutoEncoder as UpstreamAutoEncoder

    from tigon_official_ae_common import OfficialAutoEncoder

    local = load_tigon_module()
    device = torch.device("cpu")

    torch.manual_seed(4232)
    upstream_ae = UpstreamAutoEncoder(
        in_dim=17,
        n_layers=1,
        n_hidden=300,
        n_latent=10,
        activate_type="relu",
        dropout=0.2,
        norm=True,
        seed=4232,
    ).to(device)
    torch.manual_seed(4232)
    local_ae = OfficialAutoEncoder(
        in_dim=17,
        n_layers=1,
        n_hidden=300,
        n_latent=10,
        dropout=0.2,
        norm=True,
        seed=4232,
    ).to(device)
    upstream_ae_parameters = list(upstream_ae.parameters())
    local_ae_parameters = list(local_ae.parameters())
    if len(upstream_ae_parameters) != len(local_ae_parameters):
        raise AssertionError("TIGON AE parameter counts differ")
    ae_parameter_max_abs = max(
        float(torch.max(torch.abs(left - right)).detach().cpu())
        for left, right in zip(upstream_ae_parameters, local_ae_parameters)
    )
    upstream_ae.eval()
    local_ae.eval()
    ae_input = torch.randn(11, 17, dtype=torch.float32, device=device)
    ae_latent_max_abs = float(
        torch.max(
            torch.abs(
                upstream_ae.get_latent_representation(ae_input, tensor=True)
                - local_ae.encode(ae_input)
            )
        )
        .detach()
        .cpu()
    )

    # Public and benchmark network construction/initialization must produce
    # the same seeded tensors, including the constructor-initialized biases.
    torch.manual_seed(31)
    upstream_model = upstream.UOT(
        in_out_dim=3,
        hidden_dim=16,
        n_hiddens=4,
        activation="Tanh",
    ).to(device)
    upstream_model.apply(upstream.initialize_weights)
    torch.manual_seed(31)
    local_model = local.TIGONModel(3, hidden_dim=16, hidden_layers=4).to(device)
    local_model.apply(local._xavier_initialize)
    upstream_parameters = list(upstream_model.parameters())
    local_parameters = list(local_model.parameters())
    if len(upstream_parameters) != len(local_parameters):
        raise AssertionError("TIGON network parameter counts differ")
    parameter_max_abs = max(
        float(torch.max(torch.abs(left - right)).detach().cpu())
        for left, right in zip(upstream_parameters, local_parameters)
    )
    comparison_state = torch.randn(13, 3, dtype=torch.float32, device=device)
    comparison_time = torch.tensor(0.7, dtype=torch.float32, device=device)
    velocity_max_abs = float(
        torch.max(
            torch.abs(
                upstream_model.hyper_net1(comparison_time, comparison_state)
                - local_model.velocity(comparison_time, comparison_state)
            )
        )
        .detach()
        .cpu()
    )
    growth_max_abs = float(
        torch.max(
            torch.abs(
                upstream_model.hyper_net2(comparison_time, comparison_state)
                - local_model.growth(comparison_time, comparison_state)
            )
        )
        .detach()
        .cpu()
    )

    # Reproduce the two independent RNG streams used by public Sampling.
    torch.manual_seed(37)
    random.seed(37)
    sampling_centers = torch.randn(29, 3, dtype=torch.float32, device=device)
    torch_state = torch.get_rng_state()
    python_state = random.getstate()
    upstream_sample = upstream.Sampling(
        17,
        [0],
        0,
        [sampling_centers],
        0.02,
        device,
    )
    torch.set_rng_state(torch_state)
    random.setstate(python_state)
    local_sample = local._sample_gaussian_mixture(
        sampling_centers,
        17,
        0.02,
        device,
    )
    sampling_max_abs = float(
        torch.max(torch.abs(upstream_sample - local_sample)).detach().cpu()
    )

    torch.manual_seed(11)
    public_model = upstream.UOT(
        in_out_dim=3,
        hidden_dim=16,
        n_hiddens=4,
        activation="Tanh",
    ).to(device)
    public_model.apply(upstream.initialize_weights)
    adapter = PublicModelAdapter(public_model)

    state = torch.randn(17, 3, dtype=torch.float32, device=device)
    zeros = torch.zeros(17, 1, dtype=torch.float32, device=device)
    time = torch.tensor(0.9, dtype=torch.float32, device=device)
    step_size = 0.1
    upstream_rate = upstream.trans_loss(
        time,
        (state.clone(), zeros.clone(), zeros.clone()),
        public_model,
        device,
        step_size,
    )[2]
    public_exact_rate = local._TIGONUpstreamPublicActionDynamics(
        adapter,
        growth_action_weight=1.0,
        start_time=0.0,
        midpoint_step_size=step_size,
    )(
        time,
        (state.clone(), zeros.clone(), zeros.clone()),
    )[2]
    paper_fixed_rate = local._TIGONPaperFixedSampleActionDynamics(
        adapter,
        growth_action_weight=1.0,
        start_time=0.0,
        midpoint_step_size=step_size,
    )(
        time,
        (state.clone(), zeros.clone(), zeros.clone()),
    )[2]
    action_max_abs = float(
        torch.max(torch.abs(upstream_rate - public_exact_rate)).detach().cpu()
    )
    action_max_rel = float(
        torch.max(
            torch.abs(upstream_rate - public_exact_rate)
            / torch.clamp(torch.abs(upstream_rate), min=1e-12)
        )
        .detach()
        .cpu()
    )
    fixed_sample_difference = float(
        torch.max(torch.abs(upstream_rate - paper_fixed_rate)).detach().cpu()
    )

    torch.manual_seed(19)
    centers = torch.randn(41, 3, dtype=torch.float32, device=device)
    points = torch.randn(23, 3, dtype=torch.float32, device=device)
    covariance = 0.25
    upstream_density = upstream.MultimodalGaussian_density(
        points,
        [0],
        0,
        [centers],
        covariance,
        device,
    )
    vectorized_density = torch.exp(
        local._kde_log_density(
            points,
            centers,
            covariance,
            chunk_size=13,
        )
    )
    density_max_abs = float(
        torch.max(torch.abs(upstream_density - vectorized_density)).detach().cpu()
    )
    density_max_rel = float(
        torch.max(
            torch.abs(upstream_density - vectorized_density)
            / torch.clamp(torch.abs(upstream_density), min=1e-12)
        )
        .detach()
        .cpu()
    )

    report = {
        "status": "passed",
        "upstream_commit": OFFICIAL_SOURCE_COMMIT,
        "autoencoder": {
            "parameter_tensors": len(upstream_ae_parameters),
            "parameter_max_absolute_difference": ae_parameter_max_abs,
            "latent_max_absolute_difference": ae_latent_max_abs,
            "tolerance": 1e-7,
        },
        "network": {
            "parameter_tensors": len(upstream_parameters),
            "parameter_max_absolute_difference": parameter_max_abs,
            "velocity_max_absolute_difference": velocity_max_abs,
            "growth_max_absolute_difference": growth_max_abs,
            "tolerance": 1e-7,
        },
        "sampling": {
            "samples": 17,
            "covariance": 0.02,
            "max_absolute_difference": sampling_max_abs,
            "tolerance": 1e-7,
        },
        "action": {
            "time": float(time),
            "midpoint_step_size": step_size,
            "max_absolute_difference": action_max_abs,
            "max_relative_difference": action_max_rel,
            "paper_fixed_sample_max_difference_from_public": (
                fixed_sample_difference
            ),
            "tolerance": 1e-6,
        },
        "kde": {
            "dimension": 3,
            "centers": len(centers),
            "evaluation_points": len(points),
            "covariance": covariance,
            "max_absolute_difference": density_max_abs,
            "max_relative_difference": density_max_rel,
            "tolerance": 1e-6,
        },
    }
    if (
        ae_parameter_max_abs > 1e-7
        or ae_latent_max_abs > 1e-7
        or parameter_max_abs > 1e-7
        or velocity_max_abs > 1e-7
        or growth_max_abs > 1e-7
        or sampling_max_abs > 1e-7
        or action_max_abs > 1e-6
        or density_max_abs > 1e-6
    ):
        report["status"] = "failed"
        raise AssertionError(json.dumps(report, indent=2))
    (output_dir / "equivalence_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
