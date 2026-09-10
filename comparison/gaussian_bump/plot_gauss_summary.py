"""One A4-wide summary: the two spaces, the two methods, and the growth sweep.

Left    observed primary / secondary embeddings and T's reconstruction
Middle  primary paths against the bump, and the same paths lifted by T
Right   growth field d(lnw)/dt across the growth penalty beta

Composing the three as one figure keeps a single font size and one set of
colours; the standalone scripts remain the source for each part.
"""

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)

import argparse, importlib.util, os, sys
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib.pyplot as plt
import numpy as np, torch
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

HERE = (_REPO / "external/COATI/GaussToy")
sys.path.insert(0, str(HERE.parent))
from src.Neural import MLPVectorField
from src.utility import forward_ode
BENCH = Path("common/trainfbench_plot_style.py")
_s = importlib.util.spec_from_file_location("trainfbench_plot_style", BENCH)
style = importlib.util.module_from_spec(_s); sys.modules["trainfbench_plot_style"] = style
_s.loader.exec_module(style)
GREY = "#B3B3B3"; STEPS = 61; SIGMA = 0.08


def load_T():
    spec = importlib.util.spec_from_file_location("bt", HERE / "GaussianBump/train_bump_T.py")
    m = importlib.util.module_from_spec(spec); sys.modules["bt"] = m
    try: spec.loader.exec_module(m)
    except SystemExit: pass
    ck = torch.load(HERE / "GaussianBump/checkpoint/T_bump_mlp.pt", map_location="cpu",
                    weights_only=False)
    T = m.FourierBumpMLP(**{k: v for k, v in ck["config"].items() if k != "class_name"})
    T.load_state_dict(ck["state_dict"]); T.eval(); return T


def integrate(path, beta, z0):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck["func_state_dict"]
    unbalanced = any(k.startswith("growth_net") for k in state)
    f = MLPVectorField(dim=2, hidden_dim=256, n_layers=2, activation="leaky_relu",
                       unbalanced=unbalanced, alpha_growth=beta)
    f.load_state_dict(state); f.eval()
    out = forward_ode(f, z0, torch.device("cpu"), unbalancedModel=unbalanced,
                      viz_timesteps=STEPS, t_start=0.0, t_end=1.0, method="rk4")
    if unbalanced:
        lnw = out[1].squeeze(-1).numpy()
        rate = np.gradient(lnw, np.linspace(0, 1, STEPS), axis=0)
        return out[0].numpy(), rate, float(np.exp(lnw[-1]).sum())
    return out[0].numpy(), None, 1.0


def setup3d(ax, elev, azim, zoom=1.22):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1.05)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_zticks([0, 1])
    # Tick numbers removed: every 3-D axis spans 0..1, identical to the 2-D
    # panels beside them, and the z labels are drawn outside the axes bbox
    # where they collide with the next block's y label.
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.tick_params(pad=-4); ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 0.8), zoom=zoom)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width_mm", type=float, default=185.0)
    ap.add_argument("--font_size", type=float, default=10.0)
    ap.add_argument("--iteration", type=int, default=3000)
    ap.add_argument("--betas", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    ap.add_argument("--n_paths", type=int, default=110)
    ap.add_argument("--output", default="figures/gauss_summary.png")
    a = ap.parse_args()

    pri = np.load(HERE / "data/rna_2d_by_time.npz")
    sec = np.load(HERE / "data/secondary_3d_bump_by_time.npz")
    z0 = torch.tensor(pri["time0"])
    observed = np.concatenate([pri["time0"], pri["time1"]])
    T = load_T()
    g = np.linspace(0, 1, 121).astype(np.float32); XX, YY = np.meshgrid(g, g)
    dd = np.stack([XX, YY], -1) - np.array([0.5, 0.5], dtype=np.float32)
    Z = np.exp(-(dd * dd).sum(-1) / (2 * SIGMA ** 2))
    with torch.no_grad():
        pushed = T(torch.tensor(observed), 0.0).numpy()
        Z_hat = T(torch.tensor(np.stack([XX.ravel(), YY.ravel()], 1)),
                  0.0).numpy()[:, 2].reshape(XX.shape)

    balanced = {}
    for label, ckpt in (
        ("OT(RNA)", f"BalancedRNAOnly/checkpoint_long/"
                    f"ckpt_s0_e1.0_m50.0_d0.0_iter{a.iteration}.pth"),
        ("COATI bal.", f"GaussianBump/BalancedSync/checkpoint_long/"
                  f"ckpt_s0_e1.0_m50.0_d0.0_a0.35_iter{a.iteration}.pth"),
    ):
        traj, _, _ = integrate(HERE / ckpt, 1.0, z0)
        with torch.no_grad():
            lifted = torch.stack([T(torch.tensor(traj[i]), 0.0) for i in range(STEPS)]).numpy()
        balanced[label] = (traj, lifted)

    unbal = {}
    for beta in a.betas:
        for label, ckpt in (
            ("UOT(RNA)", f"UnbalancedRNAOnly/ckpt_bio_ag{beta}/"
                         f"ckpt_s0_e1.0_m50.0_d0.0_iter{a.iteration}.pth"),
            ("COATI unbal.", f"GaussianBump/UnbalancedSync/ckpt_bio_ag{beta}/"
                             f"ckpt_s0_e1.0_m50.0_d0.0_a0.35_iter{a.iteration}.pth"),
        ):
            unbal[(label, beta)] = integrate(HERE / ckpt, beta, z0)
    limit = max(np.abs(r).max() for _, r, _ in unbal.values())

    style.apply_nature_rc(font_size=a.font_size)
    plt.rcParams.update({"font.weight": "normal", "axes.titleweight": "normal",
                         "axes.labelweight": "normal",
                         # Match the size convention already used by
                         # TraInfBench/scripts/plot_gastrulation_rostral_sync_ablation.py:
                         # 12 pt titles and axis labels, 10 pt ticks and legend.
                         "axes.titlesize": 12.0,
                         "axes.labelsize": 12.0,
                         "xtick.labelsize": 10.0,
                         "ytick.labelsize": 10.0,
                         "legend.fontsize": 10.0,
                         # mathtext has its own font set and ignores font.family;
                         # without this every $...$ renders in DejaVu, not Arial.
                         "mathtext.fontset": "custom",
                         "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic",
                         "mathtext.bf": "Arial:bold"})
    w = a.width_mm / 25.4
    figure = plt.figure(figsize=(w, w * 0.40))
    outer = figure.add_gridspec(1, 3, width_ratios=[2.0, 2.0, 3.0], wspace=0.20,
                                left=0.030, right=0.930, top=0.955, bottom=0.055)
    pick = np.linspace(0, len(pri["time0"]) - 1, a.n_paths, dtype=int)
    blue, orange = style.NATURE_CUD["sky_blue"], style.NATURE_CUD["vermillion"]
    amber = style.NATURE_CUD["orange"]

    # ---- embeddings: 2x2 of panels with a full-width legend row beneath ----
    ga = outer[0].subgridspec(3, 2, wspace=0.30, hspace=0.30,
                              height_ratios=[1, 1, 0.02])
    ax = figure.add_subplot(ga[0, 0])
    for colour, key in ((blue, "time0"), (orange, "time1")):
        ax.scatter(pri[key][:, 0], pri[key][:, 1], s=0.8, color=colour, alpha=0.55,
                   linewidths=0, rasterized=True)
    ax.set_title("Primary", pad=2); ax.set_aspect("equal")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])

    ax = figure.add_subplot(ga[0, 1], projection="3d")
    ax.plot_surface(XX, YY, Z, color=GREY, linewidth=0, alpha=0.30, rstride=5,
                    cstride=5, antialiased=True)
    for colour, key in ((blue, "time0"), (orange, "time1")):
        ax.scatter(sec[key][:, 0], sec[key][:, 1], sec[key][:, 2], s=0.7,
                   color=colour, alpha=0.85, linewidths=0, rasterized=True)
    ax.set_title("Secondary", pad=-2); setup3d(ax, 30, -45)

    ax = figure.add_subplot(ga[1, 0], projection="3d")
    ax.plot_surface(XX, YY, Z_hat, color=amber, linewidth=0, alpha=0.30, rstride=5,
                    cstride=5, antialiased=True)
    ax.scatter(pushed[:, 0], pushed[:, 1], pushed[:, 2], s=0.8, marker="x",
               color=amber, alpha=0.65, linewidths=0.25, rasterized=True)
    ax.set_title(r"$T$(Primary)", pad=-2); setup3d(ax, 30, -45)

    # Legend on its own full-width row: inside a quadrant it collides with the
    # next block's y label no matter how wide the gap is made.
    ax = figure.add_subplot(ga[1, 1]); ax.axis("off")
    ax.legend(handles=[
        Line2D([], [], linestyle="none", marker="o", markersize=3, color=blue,
               label="Initial"),
        Line2D([], [], linestyle="none", marker="o", markersize=3, color=orange,
               label="Terminal"),
        Line2D([], [], linestyle="none", marker="x", markersize=3, color=amber,
               label=r"$T$(Pri.)"),
    ], frameon=False, loc="center", ncol=1, labelspacing=0.35,
        handletextpad=0.35, borderpad=0.0)

    # ---- panel b -------------------------------------------------------
    gb = outer[1].subgridspec(2, 2, wspace=0.30, hspace=0.14)
    for row, (label, method) in enumerate((("OT(RNA)", "RNA-only"),
                                           ("COATI bal.", "BalancedSync"))):
        colour = style.method_style(method).color
        traj, lifted = balanced[label]
        ax = figure.add_subplot(gb[row, 0])
        ax.scatter(observed[:, 0], observed[:, 1], s=0.5, color=GREY, alpha=0.4,
                   linewidths=0, rasterized=True)
        ax.contour(XX, YY, Z, levels=[0.05, 0.3, 0.7], colors="black", linewidths=0.3)
        for i in pick:
            ax.plot(traj[:, i, 0], traj[:, i, 1], color=colour, linewidth=0.25,
                    alpha=0.4, rasterized=True)
        ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_ylabel(label, labelpad=2)
        if row == 0:
            ax.set_title("Primary", pad=2)

        ax = figure.add_subplot(gb[row, 1], projection="3d")
        ax.plot_surface(XX, YY, Z, color=GREY, linewidth=0, alpha=0.30, rstride=6,
                        cstride=6, antialiased=True)
        for i in pick:
            ax.plot(lifted[:, i, 0], lifted[:, i, 1], lifted[:, i, 2], color=colour,
                    linewidth=0.25, alpha=0.55)
        if row == 0:
            ax.set_title(r"Secondary $T(x)$", pad=-2)
        setup3d(ax, 40, -35)

    # ---- panel c -------------------------------------------------------
    gc = outer[2].subgridspec(2, len(a.betas), wspace=0.30, hspace=0.14)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    for col, beta in enumerate(a.betas):
        for row, label in enumerate(("UOT(RNA)", "COATI unbal.")):
            ax = figure.add_subplot(gc[row, col])
            traj, rate, mass = unbal[(label, beta)]
            ax.scatter(observed[:, 0], observed[:, 1], s=0.4, color=GREY, alpha=0.3,
                       linewidths=0, rasterized=True)
            ax.contour(XX, YY, Z, levels=[0.05, 0.3, 0.7], colors="black",
                       linewidths=0.25)
            segs = [np.stack([traj[:-1, i], traj[1:, i]], axis=1) for i in pick]
            vals = [0.5 * (rate[:-1, i] + rate[1:, i]) for i in pick]
            coll = LineCollection(np.concatenate(segs), cmap="coolwarm", norm=norm,
                                  linewidths=0.3, alpha=0.8, rasterized=True)
            coll.set_array(np.concatenate(vals))
            ax.add_collection(coll)
            ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xticks([0, 1] if row == 1 else []); ax.set_yticks([0, 1] if col == 0 else [])
            # Two lines: "Terminal Mass 2.03" on one line overflows a 28 mm
            # panel at 8 pt, and the bare number is not self-explanatory.
            ax.text(0.02, 0.03, f"T.M.={mass:.2f}", transform=ax.transAxes, va="bottom")
            if row == 0:
                ax.set_title(rf"$\beta={beta:g}$", pad=2)
            if col == 0:
                ax.set_ylabel(label, labelpad=2)
    # Dedicated colourbar axis: passing ax=<panel> would steal that panel's
    # space and squeeze the last beta column.
    cax = figure.add_axes([0.947, 0.10, 0.007, 0.78])
    bar = figure.colorbar(coll, cax=cax)
    bar.outline.set_visible(False); bar.set_label(r"$d\ln w/dt$", labelpad=1)
    bar.set_ticks([-limit, 0, limit])
    bar.ax.set_yticklabels([f"{-limit:.1f}", "0", f"{limit:.1f}"])

    out = HERE / a.output
    figure.savefig(out, dpi=400, bbox_inches="tight")
    figure.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure); print(f"saved {out}")


if __name__ == "__main__":
    main()
