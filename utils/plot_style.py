"""utils/plot_style.py

Portfolio-wide figure styling.  Three palettes are defined:
  CMAP       — healthy / baseline condition
  CMAP_DMG   — outer race damaged condition  (dark:salmon_r)
  CMAP_IR    — inner race damaged condition  (ch:start=.4,rot=.3)

Usage (Section 0 of any notebook)::

    plot_style = _load_module("plot_style", "utils/plot_style.py")
    from plot_style import (apply_style, FigSize,
                            CMAP, C1, C2, C3, FAULT_COLORS,
                            CMAP_DMG, D1, D2, D3, FAULT_COLORS_DMG,
                            CMAP_IR,  I1, I2, I3, FAULT_COLORS_IR)
    apply_style()
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Palettes — single source of truth for all colours
# ---------------------------------------------------------------------------

CMAP     = sns.color_palette("ch:start=.2,rot=-.3",  as_cmap=True)  # healthy
CMAP_DMG = sns.color_palette("dark:salmon_r",         as_cmap=True)  # OR damage
CMAP_IR  = sns.color_palette("ch:start=.4,rot=.3",   as_cmap=True)  # IR damage


def blues(n: int, lo: float = 0.35, hi: float = 0.95) -> list:
    """Return n evenly-spaced RGBA colours from CMAP (healthy palette).

    Args:
        n:   Number of colours to return.
        lo:  Lower bound on the colormap (0 = lightest, 1 = darkest).
        hi:  Upper bound on the colormap.

    Returns:
        List of n RGBA tuples, light-to-dark.
    """
    return [CMAP(v) for v in np.linspace(lo, hi, n)]


def salmons(n: int, lo: float = 0.35, hi: float = 0.95) -> list:
    """Return n evenly-spaced RGBA colours from CMAP_DMG (OR damage palette).

    Args:
        n:   Number of colours to return.
        lo:  Lower bound on the colormap.
        hi:  Upper bound on the colormap.

    Returns:
        List of n RGBA tuples.
    """
    return [CMAP_DMG(v) for v in np.linspace(lo, hi, n)]


def greens(n: int, lo: float = 0.35, hi: float = 0.95) -> list:
    """Return n evenly-spaced RGBA colours from CMAP_IR (IR damage palette).

    Args:
        n:   Number of colours to return.
        lo:  Lower bound on the colormap.
        hi:  Upper bound on the colormap.

    Returns:
        List of n RGBA tuples.
    """
    return [CMAP_IR(v) for v in np.linspace(lo, hi, n)]


# Three standard line colours — healthy
C1, C2, C3 = blues(3)

# Three standard line colours — OR damage
D1, D2, D3 = salmons(3)

# Three standard line colours — IR damage
I1, I2, I3 = greens(3)

# Fault-frequency marker colours (4 shades) — healthy, OR damage, IR damage
FAULT_COLORS: dict = dict(zip(
    ["FTF", "BSF", "BPFO", "BPFI"],
    blues(4, lo=0.30, hi=0.95),
))

FAULT_COLORS_DMG: dict = dict(zip(
    ["FTF", "BSF", "BPFO", "BPFI"],
    salmons(4, lo=0.30, hi=0.95),
))

FAULT_COLORS_IR: dict = dict(zip(
    ["FTF", "BSF", "BPFO", "BPFI"],
    greens(4, lo=0.30, hi=0.95),
))

# ---------------------------------------------------------------------------
# Figure sizes  (CLAUDE.md §11.3) — scaled down ~20 % from original
# ---------------------------------------------------------------------------


class FigSize:
    """Standard figure dimensions used across all notebooks."""

    DEFAULT            = (6,   4)    # bar charts, general
    HEATMAP            = (5,   3.5)  # correlation / confusion matrix (small)
    HEATMAP_LARGE      = (6,   5)    # confusion matrix (large)
    FEATURE_IMPORTANCE = (6,   4.5)  # wide horizontal bar
    COUNT              = (4,   3)    # small count / distribution plot
    MULTI_PANEL        = (8,   6)    # large grid of subplots
    WIDE_TALL          = (8,   5)    # DSP multi-channel subplots


# ---------------------------------------------------------------------------
# Global activator
# ---------------------------------------------------------------------------


def apply_style() -> None:
    """Apply colour cycle, whitegrid theme, and default figure size.

    Call once at the bottom of Section 0 imports, replacing the bare
    sns.set_theme / plt.rcParams lines.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = FigSize.DEFAULT
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=blues(6))
