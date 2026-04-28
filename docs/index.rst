spectroxide
===========

A PDE solver for CMB :math:`\mu`- and :math:`y`-type spectral
distortions from energy and photon injection between :math:`z \sim 10^3`
and :math:`z \sim 5\times 10^6`.

The photon occupation number :math:`n(x, \tau)` evolves under
Kompaneets scattering, double Compton, bremsstrahlung, Hubble expansion,
and a user-specified injection source. The core solver is implemented
in Rust and integrated through Python; an analytic Green's-function
approximation (Chluba 2013) is also provided for fast estimates.

For physical background and derivations, see the
`paper <https://github.com/bakerem/spectroxide>`_.


Quick example
-------------

.. code-block:: python

   from spectroxide import solve

   # Full Rust PDE — single burst at z_h = 2e5, Δρ/ρ = 1e-5
   result = solve(injection={"type": "single_burst", "z_h": 2e5}, delta_rho=1e-5)
   print(f"mu = {result.mu:.3e}, y = {result.y:.3e}")

For full coverage of injection scenarios, photon injection, and tabulated
sources see :doc:`tutorials/index` or :doc:`cli`.


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   installation

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: User Guide

   tutorials/index
   cli

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference

   api/index
   rust_api
