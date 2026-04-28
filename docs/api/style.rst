Plotting utilities (``spectroxide.style``, ``spectroxide.plot_params``)
=======================================================================

Matplotlib style helpers and plot parameter constants for
publication-quality figures. Not re-exported at the top level — import
explicitly:

.. code-block:: python

   from spectroxide.style import apply_style, C, SINGLE_COL, DOUBLE_COL
   from spectroxide.plot_params import FONT_SIZE, LW

   apply_style()                                        # apply rcParams

   import matplotlib.pyplot as plt
   fig, ax = plt.subplots(figsize=SINGLE_COL)           # single-column width
   ax.plot(x, y, color=C[0], lw=LW)                     # use palette + lw

.. currentmodule:: spectroxide.style

Style
-----

.. autosummary::
   :nosignatures:

   apply_style

.. autofunction:: spectroxide.style.apply_style


Plot parameters (``spectroxide.plot_params``)
---------------------------------------------

Module-level constants providing a single source of truth for figure
styling. Import and reuse to keep visuals consistent across notebooks.

.. automodule:: spectroxide.plot_params
