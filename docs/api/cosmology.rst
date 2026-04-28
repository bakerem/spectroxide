Cosmology (``spectroxide.cosmology``)
=====================================

.. currentmodule:: spectroxide.cosmology


Flat ΛCDM background quantities, cosmology presets and the typed
:class:`Cosmology` dataclass. The Green's-function and PDE-table
modules pull from this module rather than redefining cosmology
themselves; users can either pass a :class:`Cosmology` instance or a
plain ``dict`` with the same keys to anything that takes a ``cosmo=``
argument.

Quick example
-------------

.. code-block:: python

   from spectroxide.cosmology import (
       Cosmology, hubble, cosmic_time, ionization_fraction,
       DEFAULT_COSMO, PLANCK2018_COSMO,
   )

   # Background rates with the default (Chluba 2013) parameters
   H = hubble(1e5)                      # 1/s
   t = cosmic_time(1e5)                 # s
   X_e = ionization_fraction(1100.0)    # free electrons / H atom

   # Use a typed Cosmology instance
   p18 = Cosmology.planck2018()
   H_p18 = hubble(1e5, p18.to_dict())

.. automodule:: spectroxide.cosmology
   :no-members:


Cosmology dataclass
-------------------

Typed parameter container with class-method presets and a ``to_dict``
helper for routines that expect a plain mapping.

.. autosummary::
   :nosignatures:

   Cosmology
   Cosmology.default
   Cosmology.planck2015
   Cosmology.planck2018
   Cosmology.to_dict

.. autoclass:: spectroxide.cosmology.Cosmology
   :no-members:

.. automethod:: spectroxide.cosmology.Cosmology.default
.. automethod:: spectroxide.cosmology.Cosmology.planck2015
.. automethod:: spectroxide.cosmology.Cosmology.planck2018
.. automethod:: spectroxide.cosmology.Cosmology.to_dict


Presets
-------

Plain-dict cosmology presets. Each contains the keys ``h``, ``omega_b``,
``omega_m``, ``y_p``, ``t_cmb``, ``n_eff``.

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

.. autodata:: spectroxide.cosmology.DEFAULT_COSMO
.. autodata:: spectroxide.cosmology.PLANCK2015_COSMO
.. autodata:: spectroxide.cosmology.PLANCK2018_COSMO


Background quantities
---------------------

Flat ΛCDM Hubble rate, densities, and integrated quantities.

.. autosummary::
   :nosignatures:

   hubble
   cosmic_time
   rho_gamma
   omega_gamma
   n_hydrogen
   n_electron
   baryon_photon_ratio

.. autofunction:: spectroxide.cosmology.hubble
.. autofunction:: spectroxide.cosmology.cosmic_time
.. autofunction:: spectroxide.cosmology.rho_gamma
.. autofunction:: spectroxide.cosmology.omega_gamma
.. autofunction:: spectroxide.cosmology.n_hydrogen
.. autofunction:: spectroxide.cosmology.n_electron
.. autofunction:: spectroxide.cosmology.baryon_photon_ratio


Recombination
-------------

Free-electron fraction ``X_e(z)``: Saha for helium, Peebles three-level
atom for hydrogen with fudge factor ``F = 1.125`` (Chluba & Thomas
2011). The ODE table is cached per cosmology.

.. autosummary::
   :nosignatures:

   ionization_fraction

.. autofunction:: spectroxide.cosmology.ionization_fraction
