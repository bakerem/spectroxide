CLI reference
=============

The ``spectroxide`` binary provides a command-line interface (:term:`CLI`) to the
partial differential equation (:term:`PDE`) solver. The CLI outputs JSON by
default and writes it to stdout.

.. code-block:: bash

   cargo run --release --bin spectroxide -- SUBCOMMAND [OPTIONS]

Replace ``SUBCOMMAND`` with one of the subcommands in the next section, and ``OPTIONS`` with
any of the flags that the subcommand accepts.

The CLI rejects, with an error, any flag that the subcommand or injection type
does not read, a flag given twice, a value after a flag that takes none (such as
``--production-grid``), and any extra word that is not a flag's value.


Subcommands
-----------

``solve``
~~~~~~~~~

Run the PDE solver for a specific injection scenario.

.. code-block:: bash

   spectroxide solve INJECTION_TYPE [OPTIONS]

Replace ``INJECTION_TYPE`` with a name from the first column of the following
table, and ``OPTIONS`` with that scenario's flags and any solver, cosmology, or
output options.

**Injection types:**

.. list-table::
   :widths: 30 50
   :header-rows: 1

   * - Type
     - Required flags
   * - ``single-burst``
     - ``--z-h`` [``--delta-rho``, default 1e-5] [``--sigma-z``]
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
``photon-sweep-batch``. In the Flag column, an
uppercase word such as ``Z``, ``N``, or ``VALUE`` stands for the value that you
supply:

.. list-table::
   :widths: 30 15 45
   :header-rows: 1

   * - Flag
     - Default
     - Description
   * - ``--z-start Z``
     - (varies)
     - Starting redshift. For ``solve``: :math:`z_h + 7\sigma_z` for ``single-burst``
       and ``monochromatic-photon``, where :math:`\sigma_z` is ``--sigma-z`` or
       :math:`\max(0.04\,z_h, 100)`; the resonance redshift for
       ``dark-photon-resonance``; 5e6 for every other type. The sweeps start
       each point at :math:`z_h + 7\sigma_z`.
   * - ``--z-end Z``
     - 500
     - Final redshift. Must be greater than 0.
   * - ``--n-points N``
     - 2000
     - Frequency-grid point count; 4000 by default with ``--production-grid``.
       Overrides the point count of either grid.
       Below 1000 points the solver warns that the result is untested.
   * - ``--production-grid``
     - off
     - Use the high-resolution production grid preset (4000 points).
   * - ``--dy-max VALUE``
     - 0.02
     - Cap on the adaptive ``y_C`` step.
   * - ``--dtau-max VALUE``
     - 10
     - Cap on the dimensionless Compton optical-depth step (use 3 for ``<0.1%`` precision).
   * - ``--dtau-max-photon-source VALUE``
     - 1.0
     - Cap on ``dτ`` while a photon source is active (tighter near a δ-line source).
       Must be greater than 0.
   * - ``--no-dcbr``
     - off
     - Disable double Compton and bremsstrahlung (diagnostic).
   * - ``--split-dcbr``
     - off
     - Operator-split DC/BR instead of coupled Newton iteration.
   * - ``--no-number-conserving``
     - off
     - Disable the number-conserving :math:`T`-shift subtraction (on by default).
   * - ``--nc-z-min Z``
     - 5e4
     - Below this redshift the number-conserving correction is suppressed.
       Must be 0 or greater; 0 applies it at all redshifts.
   * - ``--no-auto-refine``
     - off
     - Disable automatic grid refinement near photon-injection features.
   * - ``--threads N``
     - all cores
     - Threads for parallel sweep execution. Sweep subcommands only; ``solve``
       rejects it.

Cosmology options
-----------------

These flags select a cosmology preset or override individual parameters:

.. code-block:: bash

   --cosmology default|planck2015|planck2018
   --omega-b 0.044  --omega-m 0.26  --h 0.71  --y-p 0.24  --t-cmb 2.726  --n-eff 3.046

Individual parameters override the selected preset. The CLI uses the
same convention as the Python API: ``--omega-b`` is the fractional
baryon density :math:`\Omega_b` and ``--omega-m`` is fractional total
matter :math:`\Omega_m = \Omega_b + \Omega_\mathrm{cdm}`. The CLI
converts to physical densities :math:`\omega_b = \Omega_b h^2` and
:math:`\omega_\mathrm{cdm} = (\Omega_m - \Omega_b)\,h^2` before
constructing the internal cosmology.

``--omega-b`` and ``--omega-m`` must be supplied together — the CLI derives the CDM
density from their difference.

With a preset, ``--h`` alone keeps :math:`\omega_b` and :math:`\omega_\mathrm{cdm}`
fixed. To fix the fractional densities, also pass ``--omega-b`` and ``--omega-m``.


Output options
--------------

These flags control the output format and destination. Replace ``PATH`` with the
file to write:

.. list-table::
   :widths: 30 60
   :header-rows: 1

   * - Flag
     - Description
   * - ``--format json|csv|table``
     - Output format (default: ``json``)
   * - ``--output PATH``
     - Write to file instead of stdout


Examples
--------

The following commands show common ways to run the solver from the command line:

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
