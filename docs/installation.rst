Installation
============

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
     - matplotlib, jupyter
     - Development and testing
   * - ``doc``
     - sphinx, pydata-sphinx-theme, nbsphinx, nbsphinx-link, sphinx-copybutton, ipython
     - Building documentation

Run ``./install.sh --help`` for all options (skip steps, verbose output, etc.).


Manual installation
-------------------

**Rust** (required for the PDE solver and CLI):

If you don't have Rust installed, the easiest way is via `rustup <https://rustup.rs/>`_:

.. code-block:: bash

   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source "$HOME/.cargo/env"

**Python 3.9+** (required for the Python package and notebooks):

.. code-block:: bash

   conda create -n spectroxide python=3.11
   conda activate spectroxide

**Build and install:**

.. code-block:: bash

   cargo build --release                 # build Rust PDE solver
   cargo test                            # run all tests
   pip install -e "python/.[plot]"       # Python package with matplotlib
   pip install -e "python/.[notebook]"   # ... or with Jupyter too


Verifying the installation
--------------------------

After installation, verify both components work:

.. code-block:: bash

   # Rust binary
   cargo run --release --bin spectroxide -- info

   # Python package
   python -c "from spectroxide import run_single; print(run_single(z_h=2e5, delta_rho=1e-5))"
