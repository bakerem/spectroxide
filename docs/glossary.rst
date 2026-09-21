Glossary
========

This page defines the abbreviations and the dimensionless variables that the
rest of the documentation uses. Every other page spells out an abbreviation at
its first use and links here.

Abbreviations
-------------

.. glossary::
   :sorted:

   BR
      Bremsstrahlung. Free-free emission and absorption of a photon when an
      electron scatters off an ion (:math:`e + \mathrm{ion} \to e +
      \mathrm{ion} + \gamma`). It changes the photon number, and it
      becomes the main source of low-frequency photons at lower redshifts.

   CL
      Confidence level, as in "95% CL upper limit".

   CLI
      Command-line interface. Here, the ``spectroxide`` binary and its
      subcommands; see :doc:`cli`.

   CMB
      Cosmic microwave background. The thermal radiation left over from the
      early Universe, with a present-day temperature of about 2.73 K.

   DC
      Double Compton scattering (:math:`\gamma + e \to \gamma + \gamma + e`).
      It changes the photon number, and it is the main source of photons in the
      thermalization era (:math:`z \gtrsim 10^6`).

   DI
      Distortion intensity, :math:`\Delta I_\nu`. The difference between the
      observed specific intensity and that of a blackbody at the reference
      temperature. The CosmoTherm reference files named ``DI_*.dat`` tabulate
      it in Jy/sr against frequency in GHz.

   DM
      Dark matter.

   FIRAS
      Far Infrared Absolute Spectrophotometer. The instrument on the Cosmic
      Background Explorer (COBE) satellite whose measurement of the CMB
      spectrum gives the present
      limits :math:`|\mu| < 9\times 10^{-5}` and :math:`|y| < 1.5\times
      10^{-5}` (95% CL; Fixsen et al. 1996).

   GF
      Green's function. The distortion produced by a unit energy release at
      one redshift. Convolving it with a heating history gives a fast
      approximation to the full PDE solution.

   IC
      Initial condition.

   IMEX
      Implicit-explicit. A time-stepping scheme that treats stiff terms
      implicitly and the remaining terms explicitly within one step.

   NWA
      Narrow-width approximation. It treats resonant photon conversion as
      instantaneous at the redshift where the photon plasma mass equals the
      mass of the new particle.

   ODE
      Ordinary differential equation.

   PDE
      Partial differential equation. Here, the photon Boltzmann equation for
      the occupation number :math:`n(x, z)`, which the Rust solver integrates.

   PIXIE
      Primordial Inflation Explorer. A proposed CMB spectrometer with a target
      sensitivity about a thousand times better than FIRAS.

Distortion types
----------------

.. glossary::

   μ distortion
      A Bose-Einstein spectrum with a nonzero chemical potential,
      :math:`n = 1/(e^{x+\mu} - 1)`. Energy released at :math:`z \gtrsim
      2\times 10^5`, when Compton scattering still redistributes photons
      efficiently, produces it, with :math:`\mu \approx 1.401\,
      \Delta\rho/\rho`.

   y distortion
      The spectrum produced when Compton scattering is too slow to reach
      equilibrium, at :math:`z \lesssim 10^4`. It has the shape of the
      thermal Sunyaev-Zeldovich effect, with :math:`y = \Delta\rho/(4\rho)`.

Dimensionless variables
-----------------------

.. glossary::

   x
      Dimensionless frequency, :math:`x = h\nu/(k T_z)`, where :math:`T_z =
      T_0 (1+z)` is the reference blackbody temperature. It does not change
      as the Universe expands.

   θ_e
      Dimensionless electron temperature, :math:`\theta_e = k T_e/(m_e c^2)`.

   ρ_e
      Ratio of the electron temperature to the reference photon temperature,
      :math:`\rho_e = T_e/T_z`.

   Δn
      The distortion of the photon occupation number, :math:`\Delta n = n -
      n_\mathrm{pl}`, where :math:`n_\mathrm{pl} = 1/(e^x - 1)` is the Planck
      spectrum. This is the quantity the solver evolves.
