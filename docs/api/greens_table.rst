PDE-based numerical Green's function (``spectroxide.greens_table``)
===================================================================

.. currentmodule:: spectroxide.greens_table


Precomputed numerical Green's function from the Rust PDE, tabulated for
fast convolution of arbitrary injection histories. Strictly more
accurate than the :doc:`analytic Green's function <greens>`, especially
in the μ↔y transition region (3 × 10⁴ < z < 10⁵) where the analytic
visibility fits break down.

Use this when you need fast convolution but want PDE-quality results
(e.g. parameter scans where running the full PDE per point is too
expensive). For one-off solves, run :doc:`the PDE directly <solver>`.

Quick example
-------------

.. code-block:: python

   from spectroxide.greens_table import load_or_build_greens_table

   # Loads from cache; builds via Rust PDE if missing
   table = load_or_build_greens_table()

   # Convolve over a heating history dQ/dz
   import numpy as np
   x  = np.logspace(-2, 1.5, 200)
   dq_dz = lambda z: 1e-9 / (1.0 + z)
   dn = table.convolve(x, dq_dz, z_min=1e3, z_max=1e7)

.. automodule:: spectroxide.greens_table
   :no-members:


Tables
------

Two table classes wrap the precomputed Rust output: ``GreensTable`` for
energy injection (single ``z_h`` axis) and ``PhotonGreensTable`` for
monochromatic photon injection (two-dimensional ``(x_inj, z_h)`` axis).

``GreensTable``
~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   GreensTable
   GreensTable.greens_function
   GreensTable.distortion_from_heating
   GreensTable.mu_y_from_heating
   GreensTable.save
   GreensTable.load

.. autoclass:: spectroxide.greens_table.GreensTable
   :no-members:

.. automethod:: spectroxide.greens_table.GreensTable.greens_function
.. automethod:: spectroxide.greens_table.GreensTable.distortion_from_heating
.. automethod:: spectroxide.greens_table.GreensTable.mu_y_from_heating
.. automethod:: spectroxide.greens_table.GreensTable.save
.. automethod:: spectroxide.greens_table.GreensTable.load


``PhotonGreensTable``
~~~~~~~~~~~~~~~~~~~~~

.. note::

   The photon Green's function depends on both the injection
   redshift ``z_h`` and the injection frequency ``x_inj``, so the
   table is two-dimensional and convolution against an injection
   history requires a 2-D integral. For dense scans this is often
   slower than just running the PDE directly.

.. autosummary::
   :nosignatures:

   PhotonGreensTable
   PhotonGreensTable.greens_function_photon
   PhotonGreensTable.distortion_from_photon_injection
   PhotonGreensTable.save
   PhotonGreensTable.load

.. autoclass:: spectroxide.greens_table.PhotonGreensTable
   :no-members:

.. automethod:: spectroxide.greens_table.PhotonGreensTable.greens_function_photon
.. automethod:: spectroxide.greens_table.PhotonGreensTable.distortion_from_photon_injection
.. automethod:: spectroxide.greens_table.PhotonGreensTable.save
.. automethod:: spectroxide.greens_table.PhotonGreensTable.load


Caching
-------

Tables are expensive to build (one PDE solve per ``z_h``), so the
``load_or_build_*`` entry points cache results on disk and reuse them
across sessions. Each cached table is tagged with a hash of the physics
configuration used to generate it; on load, the hash is checked against
the current code and a :class:`GreensTableHashMismatch` warning is
emitted if they disagree, so stale caches from earlier code versions do
not silently shadow updated physics. Pass ``rebuild=True`` to force a
fresh build, or delete the cache file to start over.

.. autoclass:: spectroxide.greens_table.GreensTableHashMismatch
   :no-members:


Builders / loaders
------------------

Cache-aware constructors that load from disk or trigger a Rust PDE
build if no cached table is present.

.. autosummary::
   :nosignatures:

   load_or_build_greens_table
   load_or_build_photon_greens_table

.. autofunction:: spectroxide.greens_table.load_or_build_greens_table
.. autofunction:: spectroxide.greens_table.load_or_build_photon_greens_table
