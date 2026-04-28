"""
Centralized plot parameters for all spectroxide notebooks and figures.

This file is the single source of truth for all styling constants.
Import and use these in every notebook to ensure visual consistency::

    from spectroxide import apply_style, C, SINGLE_COL, DOUBLE_COL
    from spectroxide.plot_params import *
    apply_style()

Or simply::

    from spectroxide import *
    apply_style()
"""

# ── Color palette (Paul Tol "bright", colorblind-friendly) ──────────
C = {
    "blue": "#0077BB",
    "orange": "#EE7733",
    "teal": "#009988",
    "red": "#CC3311",
    "magenta": "#EE3377",
    "gray": "#BBBBBB",
    "cyan": "#33BBEE",
    "purple": "#AA3377",
    "black": "#000000",
}

# ── Figure dimensions (APS two-column journals) ────────────────────
SINGLE_COL = 3.375  # inches (86 mm)
DOUBLE_COL = 7.0  # inches (178 mm)

# ── Font sizes (pt) ────────────────────────────────────────────────
FONT_SIZE = 9  # base font size
LABEL_SIZE = 10  # axis labels
LEGEND_SIZE = 7  # legend text
TICK_SIZE = 8  # tick labels
TITLE_SIZE = 9  # panel titles (same as base)
ANNOT_SIZE = 7  # annotations and inset text

# ── Line widths (pt) ──────────────────────────────────────────────
LW = 1.2  # primary data lines
LW_THIN = 0.8  # secondary/reference lines
LW_THICK = 1.8  # emphasized curves
LW_AXIS = 0.5  # zero lines, axhline
LW_SPINE = 0.6  # axes spines
LW_TICK_MAJOR = 0.6  # major tick marks
LW_TICK_MINOR = 0.4  # minor tick marks

# ── Marker sizes ──────────────────────────────────────────────────
MS = 4  # default marker size
MS_SMALL = 3  # small markers (dense data)

# ── Residual panel defaults ───────────────────────────────────────
RESID_YLIM = (-15, 15)  # default residual y-limits [%]
RESID_BAND = 5  # ±% target band
RESID_BAND_ALPHA = 0.08  # band transparency
RESID_MASK_FRAC = 0.03  # mask below this fraction of peak

# ── Figure DPI ────────────────────────────────────────────────────
SCREEN_DPI = 150
SAVE_DPI = 300

# ── rcParams dict (applied by apply_style) ────────────────────────
RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": FONT_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth": LW_SPINE,
    "xtick.major.width": LW_TICK_MAJOR,
    "ytick.major.width": LW_TICK_MAJOR,
    "xtick.minor.width": LW_TICK_MINOR,
    "ytick.minor.width": LW_TICK_MINOR,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "lines.linewidth": LW,
    "lines.markersize": MS,
    "figure.dpi": SCREEN_DPI,
    "savefig.dpi": SAVE_DPI,
    "savefig.bbox": "tight",
    "legend.frameon": False,
    "legend.borderpad": 0.3,
    "legend.handlelength": 1.5,
}

# ── LaTeX preamble (matches paper/paper.tex custom commands) ──────
LATEX_PREAMBLE = r"""
\usepackage{amsmath,amssymb}
\newcommand{\dc}{\text{DC}}
\newcommand{\br}{\text{BR}}
\newcommand{\Te}{T_\mathrm{e}}
\newcommand{\Tz}{T_z}
\newcommand{\te}{\theta_\mathrm{e}}
\newcommand{\tz}{\theta_z}
\newcommand{\npl}{n_\mathrm{pl}}
\newcommand{\dn}{\Delta n}
\newcommand{\drho}{\Delta\rho/\rho}
\newcommand{\re}{\rho_\mathrm{e}}
\newcommand{\xe}{x_\mathrm{e}}
\newcommand{\Gbb}{G_\mathrm{bb}}
\newcommand{\Gth}{G_\mathrm{th}}
\newcommand{\Jmu}{J_\mu}
\newcommand{\Jbb}{J_\mathrm{bb}^*}
\newcommand{\Jy}{J_y}
\newcommand{\Ysz}{Y_\mathrm{SZ}}
\newcommand{\kc}{\kappa_\mathrm{c}}
\newcommand{\bmu}{\beta_\mu}
\newcommand{\cosmotherm}{\textsc{CosmoTherm}}
\newcommand{\spectroxide}{\textsc{spectroxide}}
"""
