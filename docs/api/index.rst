API reference
=============

The Python package ``spectroxide`` wraps the Rust partial differential
equation (:term:`PDE`) solver and provides a pure-Python analytic
Green's-function implementation. You need only the
top-level import:

.. code-block:: python

   import spectroxide

Import plot styling explicitly from its submodule:

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
      (Kompaneets, double Compton, and bremsstrahlung) with adaptive
      redshift stepping. Handles single-burst, custom-scenario,
      photon-injection, and tabulated-heating runs. This is what
      you almost certainly want.


Approximations and helpers
--------------------------

The remaining modules support cross-checks, fast estimates,
publication-quality plotting, and Far Infrared Absolute
Spectrophotometer (FIRAS) data utilities. They are useful but
secondary; the PDE solver computes the science targets.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Analytic Green's function
      :link: greens
      :link-type: doc

      ``spectroxide.greens`` — pure-Python implementation of the
      three-component analytic Green's function of Chluba (2013, MNRAS
      436, 2232). Spectral shapes, μ/y/T branching functions,
      energy-injection and photon-injection convolutions. An
      approximation — accuracy is documented on that page.

   .. grid-item-card:: PDE-based numerical Green's function
      :link: greens_table
      :link-type: doc

      ``spectroxide.greens_table`` — precomputed numerical Green's
      function from the Rust PDE, tabulated for fast convolution. More
      accurate than the analytic Green's function in the μ↔y transition
      region (3 × 10⁴ < z < 10⁵).

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
      ``Cosmology`` dataclass, and the ``DEFAULT_COSMO``,
      ``PLANCK2015_COSMO``, and ``PLANCK2018_COSMO`` presets that other
      modules pull from.

   .. grid-item-card:: Dark photon
      :link: dark_photon
      :link-type: doc

      ``spectroxide.dark_photon`` — narrow-width-approximation helpers
      (ω_pl, z_res, γ_con) for resonant γ↔A' conversion; the route to
      reproduce the dark-photon constraint numbers.

   .. grid-item-card:: Axion helpers (experimental)
      :link: axion
      :link-type: doc

      ``spectroxide.axion`` — narrow-width-approximation helpers for
      resonant γ↔a conversion.
      Experimental; the PDE path needs a binary built with
      ``--features axion``.

   .. grid-item-card:: Plotting
      :link: style
      :link-type: doc

      ``spectroxide.style`` and ``spectroxide.plot_params`` — Matplotlib
      style and constants for publication-quality figures.


``spectroxide.cosmotherm`` (loaders for CosmoTherm reference data) is a
development-only cross-validation module: its loaders, conventions, and
file paths can change without notice, and it is not part of the
documented API. :func:`~spectroxide.greens.strip_gbb` used to live there;
it is now in ``spectroxide.greens``, re-exported at the top level, and
documented on the :doc:`greens` page. The old import path still works
with a deprecation warning.

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
   style
