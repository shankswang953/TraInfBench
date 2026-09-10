"""One A4-wide summary of the ToySplit toy.

Left    the two observed spaces and T's pushforward
Middle  balanced OT(RNA) against COATI, in both spaces
Right   unbalanced runs across the growth penalty beta, primary space only,
        coloured by the local log-mass rate

The three blocks do not all come from the same training grid; see the note the
script prints when it finishes.
"""

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)

import argparse, glob, importlib.util, os, sys
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib.pyplot as plt
import numpy as np, torch
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

HERE = (_REPO / "external/COATI/ToySplit")
sys.path.insert(0, str(HERE.parent))
from src.Neural import MLPVectorField
from src.utility import forward_ode
BENCH = Path("common/trainfbench_plot_style.py")
_s = importlib.util.spec_from_file_location("trainfbench_plot_style", BENCH)
style = importlib.util.module_from_spec(_s); sys.modules["trainfbench_plot_style"] = style
_s.loader.exec_module(style)
GREY = "#B3B3B3"; STEPS = 41
LABELS = [f"time_{i}" for i in range(5)]


def load_film(checkpoint):
    spec = importlib.util.spec_from_file_location("film", HERE / "data/TrainT/film_model.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    st = torch.load(HERE / checkpoint, map_location="cpu", weights_only=False)
    T = m.FiLMMLP(**st["config"]); T.load_state_dict(st["state_dict"]); T.eval()
    return T, st.get("meta", {}).get("time_points", list(range(5)))


def integrate(path, beta, z0, grid):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck["func_state_dict"]
    unbalanced = any(k.startswith("growth_net") for k in state)
    net = MLPVectorField(dim=2, hidden_dim=256, n_layers=2, activation="leaky_relu",
                         unbalanced=unbalanced, alpha_growth=beta)
    net.load_state_dict(state); net.eval()
    out = forward_ode(net, z0, torch.device("cpu"), unbalancedModel=unbalanced,
                      viz_timesteps=STEPS, t_start=float(grid[0]),
                      t_end=float(grid[-1]), method="rk4")
    if unbalanced:
        lnw = out[1].squeeze(-1).numpy()
        rate = np.gradient(lnw, np.linspace(0, 1, STEPS), axis=0)
        return out[0].numpy(), rate, float(np.exp(lnw[-1]).sum())
    return out[0].numpy(), None, 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--betas", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    ap.add_argument("--width_mm", type=float, default=185.0)
    ap.add_argument("--font_size", type=float, default=10.0)
    ap.add_argument("--n_paths", type=int, default=150)
    ap.add_argument("--output", default="outputs/toysplit_summary.png")
    a = ap.parse_args()

    pri = np.load(HERE / "data/main_exp_data.npz")
    sec = np.load(HERE / "data/aux_exp_data.npz")
    z0 = torch.tensor(pri["time_0"])
    observed = np.concatenate([pri[l] for l in LABELS])
    observed_sec = np.concatenate([sec[l] for l in LABELS])
    T_new, t_new = load_film("data/TrainT/outputs/T_FiLM_norm.pt")
    T_old, t_old = load_film("data/TrainT/outputs/T_FiLM.pt")
    with torch.no_grad():
        pushed = np.concatenate([T_new(torch.tensor(pri[l]), float(t)).numpy()
                                 for t, l in zip(t_new, LABELS)])

    balanced = {}
    for label, ckpt in (
        ("OT(RNA)", "BalancedRNAOnly/outputs/ckpt_long_d1/ckpt_s0_e0.1_m100.0_d1.0_iter10000.pth"),
        ("COATI bal.", "BalancedSync/outputs/ckpt_long_a09/ckpt_s0_e0.1_m100.0_d10.0_a0.9_iter10000.pth"),
    ):
        traj, _, _ = integrate(HERE / ckpt, 1.0, z0, [0.0, 4.0])
        with torch.no_grad():
            lifted = np.stack([T_old(torch.tensor(traj[i]),
                                     float(t_old[0] + (t_old[-1] - t_old[0]) * i / (STEPS - 1))).numpy()
                               for i in range(STEPS)])
        balanced[label] = (traj, lifted)

    unbal = {}
    for beta in a.betas:
        for label, pat in (("UOT(RNA)", f"UnbalancedRNAOnly/outputs/ckpt_n_b{beta}/*iter5000.pth"),
                           ("COATI unbal.", f"UnbalancedSync/outputs/ckpt_nT_b{beta}/*iter5000.pth")):
            unbal[(label, beta)] = integrate(glob.glob(str(HERE / pat))[0], beta, z0,
                                             [0.0, 1.0])
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
    figure = plt.figure(figsize=(w, w * 0.34))
    outer = figure.add_gridspec(1, 3, width_ratios=[2.0, 2.0, 3.0], wspace=0.30,
                                left=0.033, right=0.930, top=0.945, bottom=0.075)
    pick = np.linspace(0, len(pri["time_0"]) - 1, a.n_paths, dtype=int)
    blue, orange = style.NATURE_CUD["sky_blue"], style.NATURE_CUD["vermillion"]
    amber = style.NATURE_CUD["orange"]
    xlim, ylim = (-0.03, 1.03), (0.30, 0.72)

    def dress(ax, ticks_x=True, ticks_y=True):
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xticks([0, 1] if ticks_x else [])
        ax.set_yticks([0.4, 0.6] if ticks_y else [])

    # ---- embeddings ----
    ga = outer[0].subgridspec(3, 2, wspace=0.26, hspace=0.30,
                              height_ratios=[1, 1, 0.02])
    for cell, (source, title) in zip((ga[0, 0], ga[0, 1]),
                                     ((pri, "Primary"), (sec, "Secondary"))):
        ax = figure.add_subplot(cell)
        for i, l in enumerate(LABELS):
            colour = blue if i == 0 else orange if i == len(LABELS) - 1 else GREY
            ax.scatter(source[l][:, 0], source[l][:, 1], s=0.6, color=colour,
                       alpha=0.6, linewidths=0, rasterized=True)
        ax.set_title(title, pad=2); dress(ax, ticks_x=False)
    ax = figure.add_subplot(ga[1, 0])
    ax.scatter(observed_sec[:, 0], observed_sec[:, 1], s=0.6, color=GREY, alpha=0.5,
               linewidths=0, rasterized=True)
    ax.scatter(pushed[:, 0], pushed[:, 1], s=0.7, marker="x", color=amber,
               alpha=0.6, linewidths=0.25, rasterized=True)
    ax.set_title(r"$T$(Primary)", pad=2); dress(ax)
    ax = figure.add_subplot(ga[1, 1]); ax.axis("off")
    ax.legend(handles=[
        Line2D([], [], linestyle="none", marker="o", markersize=3, color=blue, label="Initial"),
        Line2D([], [], linestyle="none", marker="o", markersize=3, color=orange, label="Terminal"),
        Line2D([], [], linestyle="none", marker="o", markersize=3, color=GREY, label="Interm."),
        Line2D([], [], linestyle="none", marker="x", markersize=3, color=amber, label=r"$T$(Pri.)"),
    ], frameon=False, loc="center", ncol=1, labelspacing=0.35,
        handletextpad=0.35, borderpad=0.0)

    # ---- balanced ----
    gb = outer[1].subgridspec(2, 2, wspace=0.28, hspace=0.16)
    for row, (label, method) in enumerate((("OT(RNA)", "RNA-only"),
                                           ("COATI bal.", "BalancedSync"))):
        colour = style.method_style(method).color
        traj, lifted = balanced[label]
        for col, (paths, backdrop) in enumerate(((traj, observed), (lifted, observed_sec))):
            ax = figure.add_subplot(gb[row, col])
            ax.scatter(backdrop[:, 0], backdrop[:, 1], s=0.4, color=GREY, alpha=0.35,
                       linewidths=0, rasterized=True)
            segs = [np.stack([paths[:-1, i], paths[1:, i]], axis=1) for i in pick]
            ax.add_collection(LineCollection(np.concatenate(segs), colors=colour,
                                             linewidths=0.22, alpha=0.35, rasterized=True))
            dress(ax, ticks_x=(row == 1), ticks_y=(col == 0))
            if row == 0:
                ax.set_title(["Primary", r"Secondary $T$"][col], pad=2)
            if col == 0:
                ax.set_ylabel(label, labelpad=2)

    # ---- unbalanced ----
    gc = outer[2].subgridspec(2, len(a.betas), wspace=0.28, hspace=0.16)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    for col, beta in enumerate(a.betas):
        for row, label in enumerate(("UOT(RNA)", "COATI unbal.")):
            ax = figure.add_subplot(gc[row, col])
            traj, rate, mass = unbal[(label, beta)]
            ax.scatter(observed[:, 0], observed[:, 1], s=0.4, color=GREY, alpha=0.3,
                       linewidths=0, rasterized=True)
            segs = [np.stack([traj[:-1, i], traj[1:, i]], axis=1) for i in pick]
            vals = [0.5 * (rate[:-1, i] + rate[1:, i]) for i in pick]
            coll = LineCollection(np.concatenate(segs), cmap="coolwarm", norm=norm,
                                  linewidths=0.25, alpha=0.8, rasterized=True)
            coll.set_array(np.concatenate(vals)); ax.add_collection(coll)
            dress(ax, ticks_x=(row == 1), ticks_y=(col == 0))
            ax.text(0.02, 0.04, f"T.M.={mass:.2f}", transform=ax.transAxes, va="bottom")
            if row == 0:
                ax.set_title(rf"$\beta={beta:g}$", pad=2)
            if col == 0:
                ax.set_ylabel(label, labelpad=2)
    cax = figure.add_axes([0.945, 0.12, 0.007, 0.76])
    bar = figure.colorbar(coll, cax=cax); bar.outline.set_visible(False)
    bar.set_label(r"$d\ln w/dt$", labelpad=1)
    bar.set_ticks([-limit, 0, limit])
    bar.ax.set_yticklabels([f"{-limit:.1f}", "0", f"{limit:.1f}"])

    out = HERE / a.output
    figure.savefig(out, dpi=400, bbox_inches="tight")
    figure.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(f"saved {out}")
    print("NOTE: the balanced block was trained on the 0..4 grid with T_FiLM.pt; "
          "the unbalanced block on the 0..1 grid with T_FiLM_norm.pt.")


if __name__ == "__main__":
    main()
