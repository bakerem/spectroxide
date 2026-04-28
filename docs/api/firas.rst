FIRAS data (``spectroxide.firas``)
==================================

.. currentmodule:: spectroxide.firas


Loads the COBE/FIRAS monopole spectrum, residuals, and the full 43 × 43
frequency-frequency covariance matrix from the LAMBDA archive into clean
numpy arrays. A :class:`FIRASData` instance is the primary handle: its
attributes give the data itself, and its methods provide χ²-based
constraint utilities for downstream analysis (μ/y upper limits, joint
fits over a free CMB temperature, etc.).

Quick example
-------------

.. code-block:: python

   from spectroxide.firas import FIRASData

   firas = FIRASData()

   # Loaded data
   nu = firas.freq_ghz                # frequency channels [GHz]
   r  = firas.residual_kJy            # monopole residuals [kJy/sr]
   C  = firas.cov                     # full 43 × 43 covariance [(kJy/sr)²]

   # χ² of an arbitrary intensity model against the residuals
   chi2 = firas.chi2(model_kJy=my_model)

   # 95 % CL upper limits on |μ| and y
   mu_lim = firas.upper_limit_mu()
   y_lim  = firas.upper_limit_y()

Module-level upper limits
-------------------------

Precomputed FIRAS constraints (Fixsen et al. 1996) exposed as constants
for quick use without constructing a :class:`FIRASData` object.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Constant
     - Value
     - Meaning
   * - ``MU_FIRAS_95``
     - ``9 × 10⁻⁵``
     - 95 % CL upper limit on ``|μ|``.
   * - ``Y_FIRAS_95``
     - ``1.5 × 10⁻⁵``
     - 95 % CL upper limit on ``y``.
   * - ``MU_FIRAS_68``
     - ``4.5 × 10⁻⁵``
     - 68 % CL (1σ) upper limit on ``|μ|``.
   * - ``Y_FIRAS_68``
     - ``7.5 × 10⁻⁶``
     - 68 % CL (1σ) upper limit on ``y``.

.. autodata:: spectroxide.firas.MU_FIRAS_95
   :no-value:
.. autodata:: spectroxide.firas.Y_FIRAS_95
   :no-value:
.. autodata:: spectroxide.firas.MU_FIRAS_68
   :no-value:
.. autodata:: spectroxide.firas.Y_FIRAS_68
   :no-value:


FIRASData
---------

.. automodule:: spectroxide.firas
   :no-members:

.. autoclass:: spectroxide.firas.FIRASData
   :no-members:

Loaded data
~~~~~~~~~~~

Attributes populated on construction. ``cov`` and ``cov_inv`` use the
full LAMBDA correlation matrix; ``sigma_kJy`` is the diagonal only.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Attribute
     - Shape
     - Description
   * - ``n_freq``
     - ``int``
     - Number of frequency channels (43).
   * - ``freq_cm``
     - ``(43,)``
     - Channel frequencies in cm⁻¹.
   * - ``freq_ghz``
     - ``(43,)``
     - Channel frequencies in GHz.
   * - ``x``
     - ``(43,)``
     - Dimensionless ``x = h ν / (k_B T_CMB)``.
   * - ``spectrum_MJy``
     - ``(43,)``
     - Monopole spectrum in MJy/sr.
   * - ``residual_kJy``
     - ``(43,)``
     - Monopole residuals in kJy/sr (data minus reference blackbody).
   * - ``sigma_kJy``
     - ``(43,)``
     - 1-σ diagonal uncertainties in kJy/sr.
   * - ``galaxy_kJy``
     - ``(43,)``
     - Modelled high-latitude galactic spectrum in kJy/sr.
   * - ``cov``
     - ``(43, 43)``
     - Full monopole covariance in (kJy/sr)².
   * - ``cov_inv``
     - ``(43, 43)``
     - Inverse covariance in (kJy/sr)⁻².
   * - ``corr``
     - ``(43, 43)``
     - Frequency-frequency correlation matrix (dimensionless).


Constraint utilities
~~~~~~~~~~~~~~~~~~~~

Full-covariance χ² primitive plus the headline upper-limit and
joint-fit routines used in production. Additional helpers
(template-builders, amplitude fitters, Fisher-matrix utilities) exist
on :class:`FIRASData` for ad-hoc analysis but are not part of the
documented surface.

.. autosummary::
   :nosignatures:

   FIRASData.chi2
   FIRASData.upper_limit_mu
   FIRASData.upper_limit_y
   FIRASData.profile_limit_floating_T
   FIRASData.fit_distortion

.. automethod:: spectroxide.firas.FIRASData.chi2
.. automethod:: spectroxide.firas.FIRASData.upper_limit_mu
.. automethod:: spectroxide.firas.FIRASData.upper_limit_y
.. automethod:: spectroxide.firas.FIRASData.profile_limit_floating_T
.. automethod:: spectroxide.firas.FIRASData.fit_distortion
