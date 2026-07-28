CosmoTherm interface (``spectroxide.cosmotherm``)
=================================================

.. currentmodule:: spectroxide.cosmotherm

Loaders and helpers for CosmoTherm reference data: distortion-intensity
(``DI``) files, the Green's-function database, and heating-rate models
used for cross-validation. Not re-exported at the top level (except
:func:`strip_gbb`, which is available as ``spectroxide.strip_gbb``) —
import explicitly:

.. code-block:: python

   from spectroxide.cosmotherm import load_di_file, di_to_delta_n

   x, di = load_di_file("path/to/DI_file.dat")
   dn = di_to_delta_n(x, di)

Data loaders
------------

.. autosummary::
   :nosignatures:

   load_di_file
   di_to_delta_n
   load_greens_database
   reconstruct_full_gf
   cosmotherm_gf_to_delta_n
   convolve_cosmotherm_gf
   cosmotherm_gf_distortion

.. autofunction:: spectroxide.cosmotherm.load_di_file
.. autofunction:: spectroxide.cosmotherm.di_to_delta_n
.. autofunction:: spectroxide.cosmotherm.load_greens_database
.. autofunction:: spectroxide.cosmotherm.reconstruct_full_gf
.. autofunction:: spectroxide.cosmotherm.cosmotherm_gf_to_delta_n
.. autofunction:: spectroxide.cosmotherm.convolve_cosmotherm_gf
.. autofunction:: spectroxide.cosmotherm.cosmotherm_gf_distortion

``strip_gbb`` (projecting out the temperature-shift mode) is documented
on the :doc:`greens` page.


Heating-rate models
-------------------

CosmoTherm-convention energy-injection histories (s-wave/p-wave
annihilation, decay) for building comparison heating rates.

.. autosummary::
   :nosignatures:

   ct_heating_rate_swave
   ct_heating_rate_pwave
   ct_heating_rate_decay

.. autofunction:: spectroxide.cosmotherm.ct_heating_rate_swave
.. autofunction:: spectroxide.cosmotherm.ct_heating_rate_pwave
.. autofunction:: spectroxide.cosmotherm.ct_heating_rate_decay
