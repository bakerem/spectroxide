PDE solver (``spectroxide.solver``)
===================================

.. currentmodule:: spectroxide.solver


Python wrapper around the Rust ``spectroxide`` binary. Runs the full
photon-Boltzmann PDE (Kompaneets + double Compton + bremsstrahlung)
with adaptive redshift stepping and parses the JSON output.

This page documents the **PDE solver only**. For the analytic Green's
function approximation see :doc:`greens`; for the precomputed
PDE-based numerical Green's function tables see :doc:`greens_table`.

.. note::

   The Rust binary must be built once before any of these entry
   points work::

      cargo build --release


When to call which function
---------------------------

Four PDE entry points cover the common patterns. Pick by *what you're
scanning over*:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Use case
     - Function
   * - One PDE solve for a custom injection scenario, custom heating
       history, or tabulated photon source.
     - :func:`solve` — returns a structured :class:`SolverResult`.
   * - Many injection redshifts at fixed amplitude (single-burst
       energy injection).
     - :func:`run_sweep` — one Rust process loops over ``z_injections``
       internally.
   * - Many injection redshifts at fixed ``x_inj`` (monochromatic
       photon injection).
     - :func:`run_photon_sweep` — same idea for the photon-sweep case.
   * - Many ``x_inj`` values, batched.
     - :func:`run_photon_sweep_batch`.

For parameter scans over scenario knobs (``gamma_x``, ``epsilon``,
``m_ev``, ``f_x``, …), call :func:`solve` in a Python loop — there is
no built-in batched entry point for arbitrary scenarios.


Injection scenarios
-------------------

The ``injection`` argument to :func:`solve` is a ``dict`` with a
``"type"`` key and scenario-specific parameter keys.
``delta_rho`` is always a **top-level** argument (not an injection key).
For the physics behind each scenario and full derivations, see the
`paper <https://github.com/bakerem/spectroxide>`_.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - ``"type"``
     - Required keys
   * - ``"single_burst"``
     - ``z_h``, ``sigma_z``
   * - ``"decaying_particle"``
     - ``f_x`` [eV], ``gamma_x`` [1/s]
   * - ``"annihilating_dm"``
     - ``f_ann`` [eV/s]
   * - ``"annihilating_dm_pwave"``
     - ``f_ann`` [eV/s]
   * - ``"monochromatic_photon"``
     - ``x_inj``, ``delta_n_over_n``, ``z_h``, ``sigma_z``, ``sigma_x``
   * - ``"decaying_particle_photon"``
     - ``x_inj_0``, ``f_inj``, ``gamma_x`` [1/s]
   * - ``"dark_photon_resonance"``
     - ``epsilon``, ``m_ev`` [eV]

Each parameter name is mapped to the corresponding Rust CLI flag
``--<kebab-case>`` (e.g. ``f_x → --f-x``, ``delta_n_over_n →
--delta-n-over-n``).


Custom heating and photon-source callables
------------------------------------------

For arbitrary heating histories or frequency-dependent photon sources,
pass a Python callable to :func:`solve` instead of using the
``injection`` dict.  Both are tabulated on a log-spaced grid and
dispatched to the corresponding Rust ``tabulated-*`` subcommand; values
outside the integration range are treated as zero.

``dq_dz`` — energy-injection history
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   solve(dq_dz=lambda z: dQ_dz(z), method="pde",
         z_min=1e3, z_max=3e6, n_z=5000)

* **Signature**: ``dq_dz(z) -> float`` (or array). The wrapper attempts a
  vectorised call ``dq_dz(z_arr)`` first and falls back to scalar
  evaluation. Vectorise where you can — the tabulation grid has 5000
  points by default.
* **Quantity**: :math:`d(\Delta\rho/\rho_\gamma)/dz`, the per-redshift
  derivative of the fractional energy perturbation. Dimensionless.
* **Sign**: positive for heating (energy added to the photon bath).
* **Tabulation grid**: log-spaced in :math:`(1+z)` from ``max(z_end,
  z_min)`` to ``z_max`` with ``n_z`` points. Defaults
  :math:`z_{\min}=10^3`, :math:`z_{\max}=3\times10^6`, :math:`n_z=5000`.
* **Mode**: with ``method="pde"`` the callable is tabulated and the Rust
  PDE solver integrates it. Without ``method="pde"``, ``solve`` falls
  back to the analytic Green's function (no Rust binary).

``photon_source`` — frequency-dependent photon injection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   solve(photon_source=lambda x, z: source(x, z),
         z_min=1e3, z_max=3e6, n_z=5000,
         x_min=0.01, x_max=30.0, n_x=500)

* **Signature**: ``photon_source(x, z) -> float``. Called scalar-by-scalar
  on the tabulation grid (no vectorisation), so keep it cheap.
* **Quantity**: :math:`d(\Delta n)/dz` at frequency :math:`x` and redshift
  :math:`z`, the per-redshift derivative of the photon-occupation
  perturbation.  Dimensionless.
* **Frequency**: :math:`x = h\nu/(k_{\rm B} T_z)` — the same dimensionless
  variable used everywhere in the solver.
* **Sign**: positive for photon injection at :math:`(x, z)`.
* **Tabulation grid**: 2-D, log-spaced in both ``z`` (capped at 500
  points) and ``x``. Pass an explicit ``x=`` array to use a custom
  frequency grid.
* **Mode**: PDE only (the analytic Green's function does not handle
  arbitrary frequency-resolved sources).

Both callables are spot-checked for non-finite output at five log-spaced
redshifts before tabulation.


Quick example
-------------

.. tab-set::

   .. tab-item:: Single PDE solve

      .. code-block:: python

         from spectroxide import solve

         # Decaying-particle injection (one of the built-in scenarios)
         result = solve(
             injection={
                 "type": "decaying_particle",
                 "f_x": 1e6,                  # eV
                 "gamma_x": 1e-15,            # 1/s
             },
         )
         print(result.mu, result.y)
         x, dn = result.x, result.delta_n     # numpy arrays

   .. tab-item:: Redshift sweep

      .. code-block:: python

         from spectroxide import run_sweep

         # Single-burst energy injection at multiple z_h, one Rust process
         result = run_sweep(
             delta_rho=1e-5,
             z_injections=[1e4, 1e5, 1e6],
         )
         for r in result["results"]:
             print(r["z_h"], r["pde_mu"], r["pde_y"])

   .. tab-item:: Photon injection

      .. code-block:: python

         from spectroxide import run_photon_sweep

         # Monochromatic photon injection at x_inj = 0.5, ΔN/N = 1e-6
         result = run_photon_sweep(
             x_inj=0.5,
             delta_n_over_n=1e-6,
             z_injections=[1e5, 5e5],
         )

   .. tab-item:: Tabulated heating

      .. code-block:: python

         from spectroxide import solve

         # Arbitrary dQ/dz callable; tabulated and handed to Rust
         dq_dz = lambda z: 1e-9 / (1.0 + z)
         result = solve(dq_dz=dq_dz, method="pde")

   .. tab-item:: Parameter scan

      .. code-block:: python

         import numpy as np
         from spectroxide import solve

         # Scan decaying-particle gamma_x — call solve() in a Python loop
         results = [
             solve(injection={"type": "decaying_particle",
                              "f_x": 1e6, "gamma_x": g})
             for g in np.logspace(-16, -14, 20)
         ]
         mu_arr = np.array([r.mu for r in results])

.. automodule:: spectroxide.solver
   :no-members:


For the typed cosmology container (``Cosmology`` dataclass) and the
flat ΛCDM background quantities, see :doc:`cosmology`. ``solve`` and
``run_sweep`` accept either a :class:`~spectroxide.cosmology.Cosmology`
instance or a plain dict via the ``cosmo=`` keyword.


Reference
---------

.. autosummary::
   :nosignatures:

   solve
   run_sweep
   run_photon_sweep
   run_photon_sweep_batch

.. autofunction:: spectroxide.solver.solve
.. autofunction:: spectroxide.solver.run_sweep
.. autofunction:: spectroxide.solver.run_photon_sweep
.. autofunction:: spectroxide.solver.run_photon_sweep_batch


Result container
----------------

Structured return value from :func:`solve`. Bundles the frequency grid,
distortion ``Δn(x)``, scalar μ/y/ΔT/T components, and a convenience
property converting to intensity units.

.. autosummary::
   :nosignatures:

   SolverResult
   SolverResult.delta_I

.. autoclass:: spectroxide.solver.SolverResult
   :no-members:

.. autoproperty:: spectroxide.solver.SolverResult.delta_I


Convenience wrapper
-------------------

``spectroxide.solver`` also exports :func:`run_single`, a thin wrapper
around the **analytic** Green's function in :mod:`spectroxide.greens`.
Despite living in the solver module, it does **not** invoke the Rust
PDE — it bundles single-burst and custom-heating calculations into a
single dict-returning call.

.. autofunction:: spectroxide.solver.run_single


Quality presets
---------------

Two preset dicts control grid resolution and timestep caps for
:func:`solve`, :func:`run_sweep`, and the photon-sweep entry points.
``PRODUCTION`` is the default; pass ``debug=True`` to switch to the
faster ``DEBUG`` preset for quick checks.

.. autosummary::
   :nosignatures:

   PRODUCTION
   DEBUG

.. autodata:: spectroxide.solver.PRODUCTION
   :no-value:
.. autodata:: spectroxide.solver.DEBUG
   :no-value:


Build provenance
----------------

.. autofunction:: spectroxide.solver.get_physics_hash
