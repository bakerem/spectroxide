Analytic Green's function (``spectroxide.greens``)
==================================================

.. currentmodule:: spectroxide.greens


.. important::

   This module is the **analytic three-component approximation** of
   Chluba (2013, MNRAS 436, 2232; arXiv:1304.6120), implemented in pure
   Python. It is *not* the PDE solver — for production work prefer the
   :doc:`PDE solver <solver>`. Use this module for fast estimates,
   pedagogical exploration, and the spectral templates ``M(x)``,
   ``Y_SZ(x)``, ``G_bb(x)`` that downstream code consumes.

The Chluba ansatz decomposes the distortion into three channels — μ, y,
and a temperature shift — weighted by redshift-dependent visibility /
branching functions:

.. math::

   G_{th}(x, z_h) = \frac{3}{\kappa_c}\, J_\mu(z_h)\, J_{bb}^*(z_h)\, M(x)
       + \tfrac{1}{4} J_y(z_h)\, Y_{SZ}(x)
       + \tfrac{1}{4} \bigl(1 - J_{bb}^*(z_h)\bigr)\, G_{bb}(x).

The visibility coefficients ``J_μ``, ``J_y``, ``J_bb*`` are fits to
PDE output. See :func:`j_mu`, :func:`j_y`, :func:`j_bb_star` for the
specific functional forms and references.

.. note::

   **Accuracy vs. PDE** — ``<5%`` deep μ-era (z_h > 2 × 10⁵), ``<1%``
   y-era (z_h < 10⁴), ``~8–13%`` shape error in the μ↔y transition
   (3 × 10⁴–10⁵). For the transition region prefer the PDE-based
   numerical Green's function in :doc:`greens_table`.

Quick example
-------------

.. code-block:: python

   import numpy as np
   from spectroxide.greens import greens_function, distortion_from_heating

   # Distortion from a delta-function injection at z_h = 2e5
   x = np.logspace(-2, 1.5, 200)
   dn = greens_function(x, z_h=2e5)            # per unit Δρ/ρ

   # Distortion from a heating history dQ/dz(z)
   dq_dz = lambda z: 1e-9 / (1.0 + z)          # toy decaying source
   dn_total = distortion_from_heating(x, dq_dz, z_min=1e3, z_max=1e7)

.. automodule:: spectroxide.greens
   :no-members:


Constants and presets
---------------------

Module-level constants used across the Green's-function machinery.

.. autosummary::
   :nosignatures:

   Z_MU
   BETA_MU
   KAPPA_C
   G3_PLANCK
   G2_PLANCK
   G1_PLANCK
   ALPHA_RHO
   X_BALANCED

Cosmology presets
~~~~~~~~~~~~~~~~~

All routines that take a ``cosmo`` argument expect a plain ``dict`` with
keys ``h``, ``omega_b``, ``omega_m``, ``y_p``, ``t_cmb``, ``n_eff``. Three
presets are provided:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Preset
     - Parameters
   * - ``DEFAULT_COSMO``
     - Chluba (2013): h=0.71, Ω_b=0.044, Ω_m=0.26, Y_p=0.24, T_CMB=2.726, N_eff=3.046.
   * - ``PLANCK2015_COSMO``
     - Planck 2015 (matches CosmoTherm DI files): h=0.6727, Ω_b=0.04917, Ω_m=0.3139, Y_p=0.2467.
   * - ``PLANCK2018_COSMO``
     - Planck 2018 (Planck VI 2020, TT,TE,EE+lowE+lensing): h=0.6736, Ω_b=0.04930, Ω_m=0.3153, Y_p=0.2454.

.. autodata:: spectroxide.greens.Z_MU
.. autodata:: spectroxide.greens.BETA_MU
.. autodata:: spectroxide.greens.KAPPA_C
.. autodata:: spectroxide.greens.G3_PLANCK
.. autodata:: spectroxide.greens.G2_PLANCK
.. autodata:: spectroxide.greens.G1_PLANCK
.. autodata:: spectroxide.greens.ALPHA_RHO
.. autodata:: spectroxide.greens.X_BALANCED


Spectral shapes
---------------

Dimensionless spectral templates ``M(x)``, ``Y_SZ(x)``, ``G_bb(x)`` that
multiply the μ, y, and temperature-shift coefficients in the Green's
function decomposition.

.. autosummary::
   :nosignatures:

   planck
   g_bb
   mu_shape
   y_shape
   temperature_shift_shape

.. code-block:: python

   from spectroxide.greens import mu_shape, y_shape
   x = np.linspace(0.1, 10, 100)
   M_x  = mu_shape(x)       # μ-distortion shape (peaks at x ~ 3.83)
   Y_x  = y_shape(x)        # SZ y-distortion shape

.. autofunction:: spectroxide.greens.planck
.. autofunction:: spectroxide.greens.g_bb
.. autofunction:: spectroxide.greens.mu_shape
.. autofunction:: spectroxide.greens.y_shape
.. autofunction:: spectroxide.greens.temperature_shift_shape


Visibility functions
--------------------

Redshift-dependent branching functions that route injected energy into
μ-, y-, or temperature-shift channels.

.. autosummary::
   :nosignatures:

   j_bb
   j_bb_star
   j_mu
   j_y

.. autofunction:: spectroxide.greens.j_bb
.. autofunction:: spectroxide.greens.j_bb_star
.. autofunction:: spectroxide.greens.j_mu
.. autofunction:: spectroxide.greens.j_y


Energy injection
----------------

Three-component Green's function and convolutions over a heating history
``dQ/dz``.

.. autosummary::
   :nosignatures:

   greens_function
   distortion_from_heating
   mu_from_heating
   y_from_heating

.. code-block:: python

   from spectroxide.greens import mu_from_heating, y_from_heating

   # Decaying dark-matter-like heating: dQ/dz ∝ exp(-Γt(z))
   import numpy as np
   from spectroxide.greens import cosmic_time
   dq_dz = lambda z: 1e-7 * np.exp(-1e-15 * cosmic_time(z))

   mu = mu_from_heating(dq_dz, z_min=1e3, z_max=5e6)
   y  = y_from_heating(dq_dz,  z_min=1e3, z_max=5e6)

The ``cosmo`` keyword (where supported) folds the post-recombination
Compton visibility into the y-channel automatically (via the bundled
free-electron history).

.. autofunction:: spectroxide.greens.greens_function
.. autofunction:: spectroxide.greens.distortion_from_heating
.. autofunction:: spectroxide.greens.mu_from_heating
.. autofunction:: spectroxide.greens.y_from_heating


Photon injection
----------------

Monochromatic photon injection at frequency ``x_inj`` and redshift
``z_h`` (Chluba 2015). Includes critical frequencies for DC/BR
absorption and the photon survival probability ``P_s``.

.. autosummary::
   :nosignatures:

   x_c_dc
   x_c_br
   x_c
   photon_survival_probability
   greens_function_photon
   mu_from_photon_injection
   distortion_from_photon_injection

.. code-block:: python

   from spectroxide.greens import greens_function_photon, mu_from_photon_injection

   # Δn(x_obs) for ΔN/N injected at x_inj = 0.5, z_h = 5e5
   x_obs = np.logspace(-2, 1.5, 200)
   dn = greens_function_photon(x_obs, x_inj=0.5, z_h=5e5)

   # Total μ from a fractional photon-number injection of 1e-6
   mu = mu_from_photon_injection(x_inj=0.5, z_h=5e5, delta_n_over_n=1e-6)

.. autofunction:: spectroxide.greens.x_c_dc
.. autofunction:: spectroxide.greens.x_c_br
.. autofunction:: spectroxide.greens.x_c
.. autofunction:: spectroxide.greens.photon_survival_probability
.. autofunction:: spectroxide.greens.greens_function_photon
.. autofunction:: spectroxide.greens.mu_from_photon_injection
.. autofunction:: spectroxide.greens.distortion_from_photon_injection


For the flat ΛCDM background quantities (Hubble rate, densities,
recombination history) and cosmology presets, see
:doc:`cosmology`.


Decomposition utilities
-----------------------

Decompose an arbitrary Δn(x) into (μ, y, ΔT/T) components and convert
to intensity units (MJy/sr).

.. autosummary::
   :nosignatures:

   decompose_distortion
   delta_n_to_delta_I

.. code-block:: python

   from spectroxide.greens import decompose_distortion, delta_n_to_delta_I

   mu, y, dT_T = decompose_distortion(x, dn)
   dI = delta_n_to_delta_I(x, dn)            # intensity in MJy/sr

:func:`decompose_distortion` is the entry point — it dispatches via a
``method=`` keyword to the non-linear blackbody-temperature solve
(``"be"``, default) or the linear Gram-Schmidt fit (``"gs"``).  Both
underlying routines are private (``_decompose_nonlinear_be`` /
``_decompose_gram_schmidt``); reach them through ``decompose_distortion``.

.. autofunction:: spectroxide.greens.decompose_distortion
.. autofunction:: spectroxide.greens.delta_n_to_delta_I


Number-conservation stripping
-----------------------------

FIRAS measures the CMB spectrum with the absolute temperature as a free
parameter, so a uniform shift :math:`\Delta T/T` is unobservable.
CosmoTherm therefore defines the *distortion* as the number-conserving
part of :math:`\Delta n` (the part satisfying
:math:`\int x^2 \Delta n\,dx = 0`); any nonzero photon-number
perturbation is absorbed into :math:`\alpha \cdot G_{bb}(x)`.
``strip_gbb`` performs this projection in occupation-number space.

.. autosummary::
   :nosignatures:

   spectroxide.cosmotherm.strip_gbb

.. autofunction:: spectroxide.cosmotherm.strip_gbb


Convenience wrapper
-------------------

:func:`spectroxide.solver.run_single` is a thin wrapper around the
analytic Green's function that bundles single-burst and custom-heating
calculations into a single dict-returning call. It is documented on the
:doc:`PDE-solver page <solver>` for proximity with the other ``solver``
entry points; despite living there, it does **not** invoke the Rust
PDE.
