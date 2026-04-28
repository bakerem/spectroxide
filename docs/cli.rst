CLI reference
=============

The ``spectroxide`` binary provides a command-line interface to the PDE solver.
Output is JSON by default and written to stdout.

.. code-block:: bash

   cargo run --release --bin spectroxide -- <subcommand> [options]


Subcommands
-----------

``solve``
~~~~~~~~~

Run the PDE solver for a specific injection scenario.

.. code-block:: bash

   spectroxide solve <injection-type> [options]

**Injection types:**

.. list-table::
   :widths: 30 50
   :header-rows: 1

   * - Type
     - Required flags
   * - ``single-burst``
     - ``--z-h``, ``--delta-rho`` [``--sigma-z``]
   * - ``decaying-particle``
     - ``--f-x``, ``--gamma-x``
   * - ``annihilating-dm``
     - ``--f-ann``
   * - ``annihilating-dm-pwave``
     - ``--f-ann``
   * - ``dark-photon-resonance``
     - ``--epsilon``, ``--m-ev``
   * - ``monochromatic-photon``
     - ``--x-inj``, ``--delta-n-over-n``, ``--z-h`` [``--sigma-x``]
   * - ``decaying-particle-photon``
     - ``--x-inj-0``, ``--f-inj``, ``--gamma-x``
   * - ``tabulated-heating``
     - ``--heating-table PATH`` (CSV: ``z,dq_dz``)
   * - ``tabulated-photon``
     - ``--photon-table PATH`` (CSV: ``z,x1,...,xN``)

``sweep``
~~~~~~~~~

Sweep over injection redshifts with single-burst heating.

.. code-block:: bash

   spectroxide sweep --delta-rho 1e-5 [--z-start 5e6] [--z-end 1e3]

``photon-sweep``
~~~~~~~~~~~~~~~~

Sweep over injection redshifts for monochromatic photon injection at a
fixed frequency.

.. code-block:: bash

   spectroxide photon-sweep --x-inj 1.0 [--delta-n-over-n 1e-5]

``photon-sweep-batch``
~~~~~~~~~~~~~~~~~~~~~~

Run photon sweeps for multiple injection frequencies in parallel.

.. code-block:: bash

   spectroxide photon-sweep-batch --x-inj-values 0.5,1.0,3.0,10.0

``greens``
~~~~~~~~~~

Evaluate the Green's function approximation (no PDE).

.. code-block:: bash

   spectroxide greens --z-h 2e5 [--delta-rho 1e-5]

``info``
~~~~~~~~

Print cosmological parameters and derived quantities.

.. code-block:: bash

   spectroxide info [--cosmology planck2018]


Solver options
--------------

These flags apply to ``solve``, ``sweep``, ``photon-sweep``, and
``photon-sweep-batch``:

.. list-table::
   :widths: 30 15 45
   :header-rows: 1

   * - Flag
     - Default
     - Description
   * - ``--z-start <z>``
     - (varies)
     - Starting redshift
   * - ``--z-end <z>``
     - 500
     - Final redshift
   * - ``--n-points <n>``
     - (preset)
     - Frequency-grid point count. Overrides the active fast/production preset.
   * - ``--production-grid``
     -
     - Use the high-resolution production grid preset (4000 points).
   * - ``--dy-max <val>``
     -
     - Cap on the adaptive ``y_C`` step.
   * - ``--dtau-max <val>``
     - 10
     - Cap on the dimensionless Compton optical-depth step (use 3 for ``<0.1%`` precision).
   * - ``--dtau-max-photon-source <val>``
     - 1.0
     - Cap on ``dτ`` while a photon source is active (tighter near a δ-line source).
   * - ``--no-dcbr``
     -
     - Disable double Compton and bremsstrahlung (diagnostic).
   * - ``--split-dcbr``
     -
     - Operator-split DC/BR instead of coupled Newton iteration.
   * - ``--no-number-conserving``
     -
     - Disable the number-conserving :math:`T`-shift subtraction (on by default).
   * - ``--nc-z-min <z>``
     - 5e4
     - Below this redshift the number-conserving correction is suppressed.
   * - ``--no-auto-refine``
     -
     - Disable automatic grid refinement near photon-injection features.
   * - ``--threads <n>``
     -
     - Threads for parallel sweep execution.

Cosmology options
-----------------

.. code-block:: bash

   --cosmology default|planck2015|planck2018
   --omega-b 0.044  --omega-m 0.26  --h 0.71  --y-p 0.24  --t-cmb 2.726  --n-eff 3.046

Individual parameters override the selected preset. The CLI uses the
**same convention** as the Python API: ``--omega-b`` is the fractional
baryon density :math:`\Omega_b` and ``--omega-m`` is fractional total
matter :math:`\Omega_m = \Omega_b + \Omega_\mathrm{cdm}`. The CLI
converts to physical densities :math:`\omega_b = \Omega_b h^2` and
:math:`\omega_\mathrm{cdm} = (\Omega_m - \Omega_b)\,h^2` before
constructing the internal cosmology.

``--omega-b`` and ``--omega-m`` must be supplied together — the CDM
density is derived from their difference.


Output options
--------------

.. list-table::
   :widths: 30 60
   :header-rows: 1

   * - Flag
     - Description
   * - ``--format json|csv|table``
     - Output format (default: ``json``)
   * - ``--output <path>``
     - Write to file instead of stdout


Examples
--------

.. code-block:: bash

   # Single burst in the mu-era
   spectroxide solve single-burst --z-h 2e5 --delta-rho 1e-5

   # Dark photon resonance
   spectroxide solve dark-photon-resonance --epsilon 1e-9 --m-ev 1e-7

   # Decaying particle with Planck 2018 cosmology
   spectroxide solve decaying-particle --f-x 7.8e5 --gamma-x 1.1e-10 \
       --cosmology planck2018

   # Sweep with production grid, save to file
   spectroxide sweep --delta-rho 1e-5 --production-grid \
       --output sweep_results.json

   # Green's function comparison
   spectroxide greens --z-h 2e5 --delta-rho 1e-5

   # Cosmology info
   spectroxide info --cosmology planck2018
