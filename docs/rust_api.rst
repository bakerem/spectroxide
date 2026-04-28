Rust API
========

The Rust crate ``spectroxide`` is the PDE solver itself; the Python package is a
thin wrapper that calls the Rust CLI for heavy computations and provides a
pure-Python Green's function for quick estimates. Most users only touch Python,
but the Rust API is useful when:

* embedding the solver in a larger Rust program,
* writing a custom injection scenario,
* accessing intermediate solver state (electron temperature history,
  per-step diagnostics) that the Python wrapper does not expose.

Browsing the crate documentation
--------------------------------

The rustdoc output is the authoritative reference. Two copies are available:

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Embedded rustdoc
      :link: rust/spectroxide/index.html

      Built from the current source tree and served alongside this site.
      Rebuild locally with ``make -C docs html`` (requires ``cargo`` on
      ``PATH``).

   .. grid-item-card:: docs.rs
      :link: https://docs.rs/spectroxide

      Authoritative copy for the latest published release on crates.io.

The crate root (``spectroxide``) contains an overview, and ``spectroxide::prelude``
re-exports everything you typically need: ``Cosmology``,
``ThermalizationSolver``, ``SolverConfig``, ``GridConfig``, ``FrequencyGrid``,
``InjectionScenario``, ``RefinementZone``, and the ``SolverBuilder`` /
``SolverDiagnostics`` / output types.

Quick Rust example
------------------

.. code-block:: rust

   use spectroxide::prelude::*;

   let mut solver = ThermalizationSolver::builder(Cosmology::planck2018())
       .grid(GridConfig::production())
       .injection(InjectionScenario::SingleBurst {
           z_h: 2e5,
           delta_rho_over_rho: 1e-5,
           sigma_z: 100.0,
       })
       .z_range(5e5, 1e3)
       .build()
       .unwrap();

   let snapshots = solver.run_with_snapshots(&[1e3]);
   let mu = snapshots.last().unwrap().mu;

CLI
---

For one-off runs from the command line (rather than from Rust or Python code),
see :doc:`cli`.
