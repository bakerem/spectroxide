API reference
=============

The Python package ``spectroxide`` wraps the Rust PDE solver and provides
a pure-Python analytic Green's-function implementation. Most users will
only need the top-level import:

.. code-block:: python

   import spectroxide

Plot styling lives in submodules and must be imported explicitly:

.. code-block:: python

   from spectroxide.plot_params import SINGLE_COL, DOUBLE_COL


Primary solver
--------------

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: PDE solver — :doc:`solver`
      :link: solver
      :link-type: doc

      ``spectroxide.solver``. The full photon-Boltzmann PDE
      (Kompaneets + double Compton + bremsstrahlung) with adaptive
      redshift stepping. Handles single-burst, custom-scenario,
      photon-injection, and tabulated-heating runs. **This is what
      you almost certainly want.**


Approximations and helpers
--------------------------

The remaining modules support cross-checks, fast estimates, and
publication-quality plotting. They are useful but secondary; the
science targets are computed by the PDE solver above.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Analytic Green's function
      :link: greens
      :link-type: doc

      ``spectroxide.greens`` — pure-Python implementation of the
      three-component analytic Green's function of Chluba (2013, MNRAS
      436, 2232). Spectral shapes, μ/y/T branching functions,
      energy-injection and photon-injection convolutions. **An
      approximation** — accuracy is documented on that page.

   .. grid-item-card:: PDE-based numerical Green's function
      :link: greens_table
      :link-type: doc

      ``spectroxide.greens_table`` — precomputed numerical Green's
      function from the Rust PDE, tabulated for fast convolution. More
      accurate than the analytic GF in the μ↔y transition region
      (3 × 10⁴ < z < 10⁵).

   .. grid-item-card:: FIRAS data
      :link: firas
      :link-type: doc

      ``spectroxide.firas`` — load the COBE/FIRAS monopole, residuals,
      and the full 43 × 43 covariance matrix from the LAMBDA archive.
      Includes χ² and upper-limit utilities for downstream constraints.

   .. grid-item-card:: Cosmology
      :link: cosmology
      :link-type: doc

      ``spectroxide.cosmology`` — flat ΛCDM background quantities
      (Hubble rate, densities, recombination history), the
      ``Cosmology`` dataclass, and the ``DEFAULT_COSMO`` /
      ``PLANCK2015_COSMO`` / ``PLANCK2018_COSMO`` presets that other
      modules pull from.

   .. grid-item-card:: Dark photon
      :link: dark_photon
      :link-type: doc

      ``spectroxide.dark_photon`` — NWA helpers (ω_pl, z_res, γ_con) for
      resonant γ↔A' conversion; the route to reproduce the dark-photon
      constraint numbers.

   .. grid-item-card:: Axion
      :link: axion
      :link-type: doc

      ``spectroxide.axion`` — NWA helpers for resonant γ↔a axion–photon
      conversion (Cyr, Chluba & Manoj 2024); Wien-tail depletion with
      κ = g_aγγ B_rms.

   .. grid-item-card:: CosmoTherm interface
      :link: cosmotherm
      :link-type: doc

      ``spectroxide.cosmotherm`` — loaders for CosmoTherm ``DI`` files
      and the Green's-function database, plus heating-rate models used
      for cross-validation.

   .. grid-item-card:: Plotting
      :link: style
      :link-type: doc

      ``spectroxide.style`` and ``spectroxide.plot_params`` — Matplotlib
      style and constants for publication-quality figures.


.. toctree::
   :maxdepth: 2
   :hidden:

   solver
   greens
   greens_table
   firas
   cosmology
   dark_photon
   axion
   cosmotherm
   style
