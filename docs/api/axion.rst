Axion helpers (``spectroxide.axion``)
=====================================

.. currentmodule:: spectroxide.axion

Pure-Python narrow-width-approximation (NWA) helpers for resonant
``γ ↔ a`` axion–photon conversion, following Cyr, Chluba & Manoj
(2024, arXiv:2411.13701). Mirror the Rust ``src/axion.rs`` routines and
reuse the plasma-frequency machinery of :mod:`spectroxide.dark_photon`
(the resonance condition ``m_a = ω_pl`` is identical). Not re-exported at
the top level — import explicitly:

.. code-block:: python

   from spectroxide.axion import gamma_con_axion, kappa_ev

   # Conversion parameter γ_con for g_aγγ [GeV⁻¹], B_rms [nG], m_a [eV]
   gc, z_res = gamma_con_axion(g_agamma=1e-10, b_rms=1.0, m_ev=1e-7)

Two differences from the dark-photon case:

1. The conversion probability carries ``x`` in the numerator,
   ``P(x) = 1 − exp(−γ_con·x)`` (Eq. 2), so high-frequency (Wien-tail)
   photons convert preferentially — opposite to the dark photon's ``1/x``.
2. The ``γ_con`` prefactor uses ``κ² (1+z)⁴ T_CMB(z)`` with
   ``κ = g_aγγ B_rms`` (Eq. 3b), instead of ``ε² m²``.

.. note::

   This is the **monopole / plasma-frequency** treatment
   (``m_γ² ≈ ω_pl²``), valid for ``m_a ≳ few×10⁻¹⁰ eV`` in the
   fully-ionized era. The frequency-dependent HI/HeI/HeII corrections to
   the photon mass (paper Sec. II B, Eqs. 7–12) — which shift the
   conversion redshift and produce multiple crossings near recombination —
   are **not** modeled.

.. autosummary::
   :nosignatures:

   kappa_ev
   gamma_con_axion

.. autofunction:: spectroxide.axion.kappa_ev
.. autofunction:: spectroxide.axion.gamma_con_axion
