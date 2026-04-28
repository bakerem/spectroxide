"""
Shared matplotlib styling for spectroxide notebooks and figures.

Usage::

    from spectroxide import apply_style, C, SINGLE_COL, DOUBLE_COL
    apply_style()

This enables ``text.usetex=True`` with a LaTeX preamble that defines
the same custom commands as ``paper/paper.tex``, so plot labels can use
``$\\spectroxide$``, ``$\\Gbb$``, ``$\\Te$``, etc.

All plot parameters are defined in ``plot_params.py``.
"""

import shutil

import matplotlib.pyplot as plt
from .plot_params import (
    C,
    SINGLE_COL,
    DOUBLE_COL,
    RCPARAMS,
    LATEX_PREAMBLE,
)  # noqa: F401


def apply_style(*, usetex=True):
    """Apply publication-quality matplotlib style.

    Parameters
    ----------
    usetex : bool
        If True (default), use LaTeX for all text rendering with the
        custom preamble.  Set False to fall back to mathtext (faster
        but no custom commands).

    Raises
    ------
    RuntimeError
        If ``usetex=True`` but ``latex`` or ``dvipng`` are not found on
        the system PATH.  The error message includes installation
        instructions.
    """
    if usetex and shutil.which("latex") is None:
        raise RuntimeError(
            "LaTeX not found on PATH.  Publication-quality plots require "
            "a LaTeX installation with dvipng.\n\n"
            "Install via conda:\n"
            "    conda install -c conda-forge texlive-core dvipng cm-super\n\n"
            "Or on Debian/Ubuntu:\n"
            "    sudo apt install texlive-latex-extra dvipng cm-super\n\n"
            "Or on macOS (Homebrew):\n"
            "    brew install --cask mactex-no-gui\n\n"
            "To skip LaTeX rendering, use: apply_style(usetex=False)"
        )
    style = dict(RCPARAMS)
    style["text.usetex"] = usetex
    if usetex:
        style["text.latex.preamble"] = LATEX_PREAMBLE
    plt.rcParams.update(style)
