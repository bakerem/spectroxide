Dark-photon helpers (``spectroxide.dark_photon``)
=================================================

.. currentmodule:: spectroxide.dark_photon

Pure-Python narrow-width-approximation (NWA) helpers for resonant
``γ ↔ A'`` conversion. Mirror the Rust ``src/dark_photon.rs`` routines
and are the documented route to reproduce the dark-photon constraint
numbers. Not re-exported at the top level — import explicitly:

.. code-block:: python

   from spectroxide.dark_photon import (
       plasma_frequency_ev,
       resonance_redshift,
       gamma_con,
   )

   # Plasma frequency today and the redshift where ω_pl(z) = m_A'
   omega_pl = plasma_frequency_ev(z=0.0)
   z_res = resonance_redshift(m_ev=1e-5)

   # Conversion probability factor γ_con for a given mixing ε
   gc = gamma_con(m_ev=1e-5, epsilon=1e-7)

.. autosummary::
   :nosignatures:

   plasma_frequency_ev
   resonance_redshift
   dln_omega_pl_sq_dlna
   gamma_con
   gc_per_epsilon_sq

.. autofunction:: spectroxide.dark_photon.plasma_frequency_ev
.. autofunction:: spectroxide.dark_photon.resonance_redshift
.. autofunction:: spectroxide.dark_photon.dln_omega_pl_sq_dlna
.. autofunction:: spectroxide.dark_photon.gamma_con
.. autofunction:: spectroxide.dark_photon.gc_per_epsilon_sq
