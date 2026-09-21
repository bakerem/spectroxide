Installation
============

This page covers two ways to install spectroxide: an automated script, and a manual
step-by-step setup of the Rust toolchain and the Python package.

Quick install (recommended)
---------------------------

The install script handles the Rust toolchain, compilation, and Python package:

.. code-block:: bash

   git clone https://github.com/bakerem/spectroxide.git
   cd spectroxide

   # Into a new conda environment (recommended)
   ./install.sh --conda spectroxide --extras notebook

   # Or into your current Python environment
   ./install.sh

``numpy`` and ``scipy`` are required by the Python package itself and are
always installed. The ``--extras`` flag selects optional add-ons on top:

.. list-table::
   :widths: 15 50 35
   :header-rows: 1

   * - Extra
     - Adds
     - Use case
   * - ``plot``
     - matplotlib
     - Scripts and plotting (default)
   * - ``notebook``
     - matplotlib, jupyter
     - Interactive notebooks
   * - ``dev``
     - matplotlib, jupyter, pytest, mutmut
     - Development and testing
   * - ``doc``
     - sphinx, pydata-sphinx-theme, nbsphinx, nbsphinx-link, sphinx-copybutton, ipython
     - Building documentation

Run ``./install.sh --help`` for all options, such as the flags that skip steps or print verbose output.


Manual installation
-------------------

Building from source needs the Rust toolchain, which compiles the
partial differential equation (:term:`PDE`) solver and its
command-line interface (:term:`CLI`), and Python for the wrapper package.

#. Optional: If you do not have Rust, install it with
   `rustup <https://rustup.rs/>`_:

   .. code-block:: bash

      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
      source "$HOME/.cargo/env"

#. Optional: Create and activate a conda environment for Python 3.9+:

   .. code-block:: bash

      conda create -n spectroxide python=3.11
      conda activate spectroxide

#. Clone the repository and enter it:

   .. code-block:: bash

      git clone https://github.com/bakerem/spectroxide.git
      cd spectroxide

#. Build the Rust PDE solver:

   .. code-block:: bash

      cargo build --release

#. Optional: Run the Rust tests. Use ``--release``, because some solver tests
   are slow in a debug build:

   .. code-block:: bash

      cargo test --release

#. Install the Python package. For plotting support (matplotlib), run:

   .. code-block:: bash

      pip install -e "python/.[plot]"

   Optional: To also install Jupyter for the notebooks, run this command
   instead:

   .. code-block:: bash

      pip install -e "python/.[notebook]"


Verifying the installation
--------------------------

After installation, verify both components work:

.. code-block:: bash

   # Rust binary
   cargo run --release --bin spectroxide -- info

   # Python package
   python -c "from spectroxide import run_single; print(run_single(z_h=2e5, delta_rho=1e-5))"
