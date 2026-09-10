"""Single source of truth for TraInfBench publication figure styling.

The categorical colours are based on the colour-blind-safe palette explicitly
recommended by the Nature research figure guide.  Method identity is never
encoded by colour alone: markers and line styles are fixed here as well.
"""
from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[1]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


from dataclasses import asdict, dataclass

import matplotlib as mpl


# Nature / Okabe-Ito colour-blind-safe categorical colours.
NATURE_CUD = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}


@dataclass(frozen=True)
class MethodStyle:
    color: str
    marker: str
    linestyle: str = "-"
    markerfacecolor: str | None = None
    markeredgecolor: str | None = None
    linewidth: float = 1.8
    markersize: float = 5.0

    def matplotlib_kwargs(self) -> dict[str, object]:
        result = asdict(self)
        result["markerfacecolor"] = (
            self.color if self.markerfacecolor is None else self.markerfacecolor
        )
        result["markeredgecolor"] = (
            self.color if self.markeredgecolor is None else self.markeredgecolor
        )
        return result


# Semantic rules:
#   * the default benchmark contains only COATI balanced and unbalanced;
#   * BOT/UOT ablations are deliberately absent and are styled only when an
#     analysis explicitly requests them;
#   * balanced/unbalanced variants share a marker family, while colour remains
#     strongly separated so that bar charts are also unambiguous;
#   * every external method retains one colour and marker everywhere.
METHOD_STYLES: dict[str, MethodStyle] = {
    "COATI balanced": MethodStyle(
        color=NATURE_CUD["blue"],
        marker="o",
    ),
    "COATI unbalanced": MethodStyle(
        color=NATURE_CUD["vermillion"],
        marker="D",
    ),
    "MIOFlow": MethodStyle(
        color=NATURE_CUD["orange"],
        marker="s",
    ),
    "CytoBridge balanced": MethodStyle(
        color=NATURE_CUD["bluish_green"],
        marker="^",
    ),
    "CytoBridge unbalanced": MethodStyle(
        color=NATURE_CUD["reddish_purple"],
        marker="v",
    ),
    "TIGON": MethodStyle(
        color=NATURE_CUD["sky_blue"],
        marker="p",
    ),
    "TrajectoryNet": MethodStyle(
        color=NATURE_CUD["yellow"],
        marker="X",
        markeredgecolor=NATURE_CUD["black"],
        linewidth=2.0,
    ),
    "Balanced RNA-only": MethodStyle(
        color=NATURE_CUD["sky_blue"],
        marker="o",
    ),
    "Unbalanced RNA-only": MethodStyle(
        color=NATURE_CUD["black"],
        marker="D",
    ),
    "Observed / real": MethodStyle(
        color=NATURE_CUD["black"],
        marker="*",
        linewidth=2.0,
        markersize=6.5,
    ),
    # Explicit balanced-OT ablations. RNA uses a distinct light blue so the
    # baseline remains separable from dark-blue COATI balanced and green
    # CytoBridge balanced without relying on an open marker or hatch pattern.
    "OT baseline RNA": MethodStyle(
        color=NATURE_CUD["sky_blue"],
        marker="^",
    ),
    "OT baseline ATAC": MethodStyle(
        color=NATURE_CUD["reddish_purple"],
        marker="s",
        markerfacecolor="none",
    ),
}


METHOD_ALIASES = {
    "BSOT": "COATI balanced",
    "BalancedSync": "COATI balanced",
    "COATI bal.": "COATI balanced",
    "COATI-BSOT": "COATI balanced",
    "USOT": "COATI unbalanced",
    "UnbalancedSync": "COATI unbalanced",
    "COATI unbal.": "COATI unbalanced",
    "COATI-USOT": "COATI unbalanced",
    "CytoBridge bal.": "CytoBridge balanced",
    "CytoBridge unbal.": "CytoBridge unbalanced",
    "CytoBridge": "CytoBridge balanced",
    "Real": "Observed / real",
    "Observed": "Observed / real",
    "Ground truth": "Observed / real",
    "BOT-RNA": "OT baseline RNA",
    "RNA-only": "OT baseline RNA",
    "BOT-ATAC": "OT baseline ATAC",
    "ATAC-only": "OT baseline ATAC",
    "RNA-only balanced": "Balanced RNA-only",
    "RNA-only unbalanced": "Unbalanced RNA-only",
}


# Gastrulation cell types use a separate, experiment-level palette generated
# with glasbey 0.3.0:
#   create_palette(
#       palette_size=32, colorblind_safe=True,
#       cvd_type="deuteranomaly", cvd_severity=100,
#       optimize_palette_search_radius=50,
#       lightness_bounds=(32, 78), chroma_bounds=(20, 60),
#   )
# See https://glasbey.readthedocs.io/en/latest/color_vision_deficiency.html.
# The moderate lightness/chroma bounds implement BioRender's recommendation to
# avoid pure, over-saturated colours.  Colours deliberately do *not* form
# lineage gradients.  With 32 categories, colour alone is never sufficient;
# atlas figures must retain direct labels and/or panel subsets.
GASTRULATION_CELLTYPE_GROUPS: dict[str, tuple[str, ...]] = {
    "Progenitor / primitive streak": (
        "Epiblast",
        "Primitive Streak",
        "Anterior Primitive Streak",
        "Caudal epiblast",
        "PGC",
    ),
    "Neural / ectoderm": (
        "Rostral neurectoderm",
        "Caudal neurectoderm",
        "Forebrain/Midbrain/Hindbrain",
        "Spinal cord",
        "Neural crest",
        "Surface ectoderm",
    ),
    "Axial / paraxial mesoderm": (
        "NMP",
        "Nascent mesoderm",
        "Mixed mesoderm",
        "Caudal Mesoderm",
        "Paraxial mesoderm",
        "Somitic mesoderm",
        "Notochord",
    ),
    "Other mesoderm": (
        "Pharyngeal mesoderm",
        "Intermediate mesoderm",
        "Cardiomyocytes",
        "Mesenchyme",
        "Allantois",
    ),
    "Haemato-endothelial": (
        "Haematoendothelial progenitors",
        "Endothelium",
        "Blood progenitors 1",
        "Blood progenitors 2",
        "Erythroid1",
        "Erythroid2",
        "Erythroid3",
    ),
    "Endoderm": (
        "Def. endoderm",
        "Gut",
    ),
    "Other / missing": (
        "Unannotated",
    ),
}


GASTRULATION_CELLTYPE_COLORS: dict[str, str] = {
    # Five frequently co-plotted posterior states receive the first five,
    # maximally separated colours.
    "Caudal epiblast": "#5559C6",
    "NMP": "#AA5100",
    "Spinal cord": "#FBBE31",
    "Paraxial mesoderm": "#D76D9A",
    "Somitic mesoderm": "#39C6FF",
    # Remaining individual cell types: severe-deuteranomaly-optimised Glasbey.
    "Epiblast": "#0C755D",
    "Primitive Streak": "#D78661",
    "Anterior Primitive Streak": "#9ADBE3",
    "PGC": "#FBC292",
    "Rostral neurectoderm": "#826D92",
    "Caudal neurectoderm": "#6D9671",
    "Forebrain/Midbrain/Hindbrain": "#798AB6",
    "Neural crest": "#D29A00",
    "Surface ectoderm": "#79E7CA",
    "Nascent mesoderm": "#D79AD7",
    "Mixed mesoderm": "#457124",
    "Caudal Mesoderm": "#2D6D8A",
    "Notochord": "#04927D",
    "Pharyngeal mesoderm": "#8E9210",
    "Intermediate mesoderm": "#D2B65D",
    "Cardiomyocytes": "#E78EA6",
    "Mesenchyme": "#10A29E",
    "Allantois": "#49DFFF",
    "Haematoendothelial progenitors": "#416D6D",
    "Endothelium": "#A66D41",
    "Blood progenitors 1": "#7D9AFF",
    "Blood progenitors 2": "#49D78A",
    "Erythroid1": "#417DB2",
    "Erythroid2": "#75C2BA",
    "Erythroid3": "#7179E7",
    "Def. endoderm": "#DB71D7",
    "Gut": "#AE4559",
    # Missing annotation is not a biological category.
    "Unannotated": "#B3B3B3",
}


GASTRULATION_CELLTYPE_ALIASES = {
    "Definitive endoderm": "Def. endoderm",
    "Caudal mesoderm": "Caudal Mesoderm",
    "Unknown": "Unannotated",
    "unknown": "Unannotated",
    "nan": "Unannotated",
}


# Mouse secondary-palate cell types.  This palette is the single source of
# truth for every palate composition and trajectory figure.  The colours are
# drawn from the same severe-deuteranomaly-optimised Glasbey pool used for the
# gastrulation atlas, then reassigned to the palate's seven biological states.
# Anterior/posterior receive a robust lightness contrast, while osteogenic,
# dental, and perimysial use distinct teal, pink, and green identities.
PALATE_CELLTYPE_ORDER: tuple[str, ...] = (
    "CNC-derived progenitors",
    "anterior palatal mesenchymal",
    "posterior palatal mesenchymal",
    "osteogenic",
    "dental mesenchymal",
    "perimysial",
    "intermidiate cells",
)

PALATE_CELLTYPE_LABELS: dict[str, str] = {
    "CNC-derived progenitors": "CNC progenitor",
    "anterior palatal mesenchymal": "Anterior",
    "posterior palatal mesenchymal": "Posterior",
    "osteogenic": "Osteogenic",
    "dental mesenchymal": "Dental",
    "perimysial": "Perimysial",
    "intermidiate cells": "Intermediate",
}

PALATE_CELLTYPE_COLORS: dict[str, str] = {
    "CNC-derived progenitors": "#798AB6",
    "anterior palatal mesenchymal": "#FBBE31",
    "posterior palatal mesenchymal": "#AA5100",
    "osteogenic": "#10A29E",
    "dental mesenchymal": "#D76D9A",
    "perimysial": "#0C755D",
    "intermidiate cells": "#B3B3B3",
}


def palate_celltype_color(name: str) -> str:
    if name not in PALATE_CELLTYPE_COLORS:
        raise KeyError(
            f"No fixed palate cell-type colour for {name!r}. Add it to "
            "PALATE_CELLTYPE_COLORS rather than assigning a local colour."
        )
    return PALATE_CELLTYPE_COLORS[name]


# scMultiSim synthetic populations.  Only four categories exist, so they are
# taken directly from the Okabe-Ito pool rather than from Glasbey: the two
# source branches keep the strongest separation (blue vs orange) and the two
# terminal states that split out of `4_5` take the remaining two hues.  Note
# that `4_5` is observed only at time1/time2 -- it becomes `5_2`/`5_3`.
SYNTHETIC_POPULATION_ORDER: tuple[str, ...] = ("4_1", "4_5", "5_2", "5_3")

SYNTHETIC_POPULATION_COLORS: dict[str, str] = {
    "4_1": NATURE_CUD["blue"],
    "4_5": NATURE_CUD["orange"],
    "5_2": NATURE_CUD["bluish_green"],
    "5_3": NATURE_CUD["reddish_purple"],
}


def synthetic_population_color(name: str) -> str:
    if name not in SYNTHETIC_POPULATION_COLORS:
        raise KeyError(
            f"No fixed synthetic population colour for {name!r}. Add it to "
            "SYNTHETIC_POPULATION_COLORS rather than assigning a local colour."
        )
    return SYNTHETIC_POPULATION_COLORS[name]


# Observed snapshot time is ordinal, not categorical, so it uses a
# perceptually uniform sequential map instead of the categorical pool.  Keeping
# the name here means every figure that colours by time samples the same ramp.
SEQUENTIAL_TIME_CMAP = "viridis"
# Endpoints are trimmed: pure viridis yellow is illegible on white, and the
# darkest end is hard to separate from black annotation.
SEQUENTIAL_TIME_RANGE = (0.08, 0.88)


def time_colors(n_times: int) -> list[str]:
    """Evenly spaced colours along the shared sequential time ramp."""

    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(SEQUENTIAL_TIME_CMAP)
    lo, hi = SEQUENTIAL_TIME_RANGE
    if n_times == 1:
        positions = [0.5 * (lo + hi)]
    else:
        positions = [
            lo + (hi - lo) * index / (n_times - 1) for index in range(n_times)
        ]
    return [mpl.colors.to_hex(cmap(position)) for position in positions]


# Coarse states used in posterior-lineage composition plots.  These are
# deliberately maximally separated rather than shaded by lineage: the task is
# to compare five trajectories, not to display a 32-cell atlas.
GASTRULATION_FATE_STYLES: dict[str, MethodStyle] = {
    "Caudal epiblast": MethodStyle(
        color=NATURE_CUD["bluish_green"],
        marker="o",
    ),
    "NMP": MethodStyle(
        color=NATURE_CUD["reddish_purple"],
        marker="D",
    ),
    "Posterior neural-spinal": MethodStyle(
        color=NATURE_CUD["blue"],
        marker="^",
    ),
    "Paraxial-somitic": MethodStyle(
        color=NATURE_CUD["vermillion"],
        marker="s",
    ),
    "Other": MethodStyle(
        color=NATURE_CUD["black"],
        marker="X",
    ),
}


def canonical_method_name(name: str) -> str:
    return METHOD_ALIASES.get(name, name)


def method_style(name: str) -> MethodStyle:
    canonical = canonical_method_name(name)
    if canonical not in METHOD_STYLES:
        raise KeyError(
            f"No fixed method style for {name!r} (canonical={canonical!r}). "
            "Add it to METHOD_STYLES rather than assigning a local colour."
        )
    return METHOD_STYLES[canonical]


def gastrulation_celltype_color(name: str) -> str:
    canonical = GASTRULATION_CELLTYPE_ALIASES.get(name, name)
    if canonical not in GASTRULATION_CELLTYPE_COLORS:
        raise KeyError(
            f"No fixed gastrulation cell-type colour for {name!r} "
            f"(canonical={canonical!r}). Add it to "
            "GASTRULATION_CELLTYPE_COLORS rather than assigning a local colour."
        )
    return GASTRULATION_CELLTYPE_COLORS[canonical]


def apply_nature_rc(font_size: float = 7.0) -> None:
    """Apply Nature-compatible editable-vector defaults.

    Nature's final assembled artwork uses 5--7 pt text.  Callers that create
    standalone panels for later down-scaling may explicitly request 12 pt.
    """

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            # Mathtext has its own font set and ignores font.family, so without
            # these four any $W_2$ or $t$ in a label silently falls back to
            # DejaVu Sans and the figure ships with two typefaces embedded.
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
