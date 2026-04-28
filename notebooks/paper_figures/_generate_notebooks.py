#!/usr/bin/env python3
"""Generate all paper figure notebooks.

Each notebook is self-contained: run top-to-bottom to regenerate one paper figure.
Output PDFs go to ../figures/ (i.e. notebooks/figures/).
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
FIG_DIR = HERE.parent / "figures"


def make_notebook(cells):
    """Create a notebook dict from a list of (cell_type, source_string) tuples."""
    nb_cells = []
    for ctype, src in cells:
        if ctype == "markdown":
            nb_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": src.splitlines(True),
            })
        else:
            nb_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": src.splitlines(True),
            })
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(name, cells):
    path = HERE / (name + ".ipynb")
    nb = make_notebook(cells)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("  Created " + path.name)


# Common path setup for first cell
_SETUP = (
    "import os\n"
    "from pathlib import Path\n"
    "\n"
    "# Ensure cargo is on PATH\n"
    "cargo_bin = Path.home() / '.cargo' / 'bin'\n"
    "if cargo_bin.is_dir() and str(cargo_bin) not in os.environ.get('PATH', ''):\n"
    "    os.environ['PATH'] = str(cargo_bin) + os.pathsep + os.environ.get('PATH', '')\n"
    "\n"
    "PROJECT_ROOT = Path.cwd().parent.parent\n"
    "FIG_DIR = PROJECT_ROOT / 'notebooks' / 'figures'\n"
    "FIG_DIR.mkdir(exist_ok=True)\n"
)


# ================================================================
# Read original source files and extract the code we need
# ================================================================

def read_script(path):
    """Read a Python script and return its contents (minus shebang/docstring preamble)."""
    return Path(path).read_text()


def read_notebook_cells(path):
    """Read a .ipynb and return list of (cell_type, source_string) tuples."""
    with open(path) as f:
        nb = json.load(f)
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


# ================================================================
# Build all notebooks
# ================================================================

def build_all():
    root = HERE.parent.parent  # project root

    # ------------------------------------------------------------------
    # 1. Visibility Functions (Fig 1) — from dev/scripts/plot_visibility_comparison.py
    # ------------------------------------------------------------------
    vis_src = (root / "dev" / "scripts" / "plot_visibility_comparison.py").read_text()
    # Remove the shebang, docstring, and sys.path hack; replace with our setup
    # and fix the output path
    lines = vis_src.splitlines(True)
    # Find where the actual code starts (after imports)
    code_start = 0
    for i, line in enumerate(lines):
        if line.startswith("import sys"):
            code_start = i
            break

    # Build clean version
    vis_code = ""
    skip_syspath = True
    for line in lines[code_start:]:
        if "sys.path.insert" in line:
            continue
        if "pathlib.Path(__file__)" in line and "outpath" in line:
            vis_code += "outpath = FIG_DIR / 'pde_visibility_fit.pdf'\n"
            continue
        if "pathlib.Path(__file__)" in line and "datadir" in line:
            vis_code += "datadir = PROJECT_ROOT / 'dev' / 'data'\n"
            continue
        vis_code += line

    write_notebook("visibility_functions", [
        ("markdown",
         "# Visibility Functions: Literature vs PDE-Fitted Parameters\n\n"
         "Generates `pde_visibility_fit.pdf` (Figure 1 in paper).\n\n"
         "Compares Chluba (2013) parameterizations of J_bb*, J_mu, J_y "
         "with our PDE-derived fits. Bottom panel shows fractional residuals."),
        ("code", _SETUP + vis_code),
    ])

    # ------------------------------------------------------------------
    # 2. mu/y vs Injection Redshift (Fig 2) — from dev/notebooks/mu_y_vs_zh.ipynb
    # ------------------------------------------------------------------
    mu_y_cells = read_notebook_cells(root / "dev" / "notebooks" / "mu_y_vs_zh.ipynb")
    # Cells: 1=imports, 3=sweep, 4=extract, 5=compute, 6=analytic, 8=plot_left, 9=plot_era1, 10=plot_right+save
    # We need code cells 1,3,4,5,6,8,9,10 (indices of code cells)
    code_cells = [(i, src) for i, (ct, src) in enumerate(mu_y_cells) if ct == "code"]

    # Cell 0 (code): imports — fix path
    imports_src = code_cells[0][1]
    imports_src = imports_src.replace(
        "PROJECT_ROOT = Path.cwd().parent.parent",
        ""  # we set it in _SETUP
    ).replace(
        "FIG_DIR = PROJECT_ROOT / \"notebooks\" / \"figures\"\nFIG_DIR.mkdir(exist_ok=True)",
        ""  # we set it in _SETUP
    )

    # Merge all code cells into logical groups
    sweep_src = code_cells[1][1]  # run sweep
    extract_src = code_cells[2][1]  # extract spectra
    compute_src = code_cells[3][1]  # mu_over_drho etc
    analytic_src = code_cells[4][1]  # z_smooth etc

    # Plot cells: 5,6,7 are the three plot sub-cells
    plot_parts = []
    for idx in range(5, len(code_cells)):
        plot_parts.append(code_cells[idx][1])
    plot_src = "\n".join(plot_parts)
    # Fix output path
    plot_src = plot_src.replace(
        'outpath = FIG_DIR / "pde_mu_y_vs_zh.pdf"',
        "outpath = FIG_DIR / 'pde_mu_y_vs_zh.pdf'"
    )

    write_notebook("mu_y_vs_injection_redshift", [
        ("markdown",
         "# Distortion Amplitudes vs Injection Redshift\n\n"
         "Generates `pde_mu_y_vs_zh.pdf` (Figure 2 in paper).\n\n"
         "Left: mu/drho in the mu-era. Right: 4y/drho in the y-era."),
        ("code", _SETUP + "\n" + imports_src),
        ("markdown", "## Run PDE sweep"),
        ("code", sweep_src),
        ("markdown", "## Extract spectra via three-component GF fit"),
        ("code", extract_src + "\n" + compute_src + "\n" + analytic_src),
        ("markdown", "## Plot"),
        ("code", plot_src),
    ])

    # ------------------------------------------------------------------
    # 3. CosmoTherm Comparison (Fig 3) — from dev/notebooks/pde_validation.ipynb
    # ------------------------------------------------------------------
    pv_cells = read_notebook_cells(root / "dev" / "notebooks" / "pde_validation.ipynb")
    pv_code = [(i, src) for i, (ct, src) in enumerate(pv_cells) if ct == "code"]

    # Use explicit cell indices (from manual exploration):
    # Cell 18: CosmoTherm imports
    # Cell 19: CT load (try/except)
    # Cell 21: plot_params imports
    # Cell 22: CT setup (reconstruct_full_gf, target_zh, PDE runs)
    # Cell 23: Figure
    ct_imports = pv_cells[18][1]
    ct_load = pv_cells[19][1]
    pp_imports = pv_cells[21][1]
    ct_setup = pv_cells[22][1]
    ct_figure = pv_cells[23][1]

    # Fix figure save path
    ct_figure = ct_figure.replace(
        'plt.savefig("../../notebooks/figures/pde_cosmotherm_comparison.pdf", bbox_inches="tight")',
        "plt.savefig(FIG_DIR / 'pde_cosmotherm_comparison.pdf', bbox_inches='tight')"
    )

    # Build the _get helper
    get_helper = (
        "def _get(d, key, fallback_key=None):\n"
        "    if key in d:\n"
        "        return d[key]\n"
        "    if fallback_key and fallback_key in d:\n"
        "        return d[fallback_key]\n"
        "    return 0.0\n"
    )

    write_notebook("cosmotherm_comparison", [
        ("markdown",
         "# PDE vs CosmoTherm Spectral Comparison\n\n"
         "Generates `pde_cosmotherm_comparison.pdf` (Figure 3 in paper).\n\n"
         "Six panels (2x3) comparing intensity distortions from single-burst "
         "energy injection at representative redshifts."),
        ("code", _SETUP + "\n"
         "import numpy as np\n"
         "import matplotlib.pyplot as plt\n"
         "\n"
         "from spectroxide import (\n"
         "    run_sweep, run_single,\n"
         "    greens_function, mu_shape, y_shape, g_bb,\n"
         "    j_mu, j_y, j_bb_star,\n"
         "    KAPPA_C,\n"
         "    delta_n_to_delta_I,\n"
         "    apply_style, C, SINGLE_COL, DOUBLE_COL,\n"
         ")\n"
         + ct_imports + "\n"
         + pp_imports + "\n"
         "\napply_style()\n"
         "\ndelta_rho = 1e-5\n"
         "\n" + get_helper),
        ("markdown", "## Load CosmoTherm GF database and run PDE"),
        ("code", ct_load + "\n" + ct_setup),
        ("markdown", "## Figure"),
        ("code", ct_figure),
    ])

    # ------------------------------------------------------------------
    # 4. DM Scenario Comparison (Fig 4) — from dev/notebooks/pde_greens_function.ipynb
    # ------------------------------------------------------------------
    gf_cells = read_notebook_cells(root / "dev" / "notebooks" / "pde_greens_function.ipynb")
    gf_code = {i: src for i, (ct, src) in enumerate(gf_cells) if ct == "code"}

    def find_gf_cell(keyword):
        for i, src in gf_code.items():
            if keyword in src:
                return i, src
        return None, None

    # Use explicit cell index for imports (cell 1) and strip sys.path hack
    gf_imports = gf_cells[1][1]
    # Remove sys.path.insert line
    gf_imports = "\n".join(
        line for line in gf_imports.splitlines()
        if "sys.path.insert" not in line
    )
    _ = None  # unused
    _, gf_params = find_gf_cell("f_ann_CT_sw = 1e-22")      # cell 3
    _, gf_ct_load = find_gf_cell("z_h_ct, x_ct, g_th_ct = load_greens_database")  # cell 5
    _, gf_pde_table = find_gf_cell("Build PDE-derived Green's function table")  # cell 7
    _, gf_convolve = find_gf_cell("Convolve PDE GF table with DM")  # cell 8
    _, gf_dm_pde = find_gf_cell("dm_injections =")          # cell 12
    _, gf_strip = find_gf_cell("pde_no_gbb = {}")           # cell 14
    _, gf_figure = find_gf_cell("Paper figure: all DM scenarios overlaid")  # cell 20

    # Fix paths in cells
    gf_pde_table = gf_pde_table.replace(
        'CACHE_PATH = pathlib.Path("../data/pde_gf_table_cache.npz")',
        "CACHE_PATH = PROJECT_ROOT / 'dev' / 'data' / 'pde_gf_table_cache.npz'"
    ).replace(
        'CACHE_PATH = pathlib.Path("../data/pde_gf_table_cache.npz")',
        "CACHE_PATH = PROJECT_ROOT / 'dev' / 'data' / 'pde_gf_table_cache.npz'"
    )

    gf_dm_pde = gf_dm_pde.replace(
        'PDE_CACHE = pathlib.Path("../data/pde_dm_results_nc_cache.npz")',
        "PDE_CACHE = PROJECT_ROOT / 'dev' / 'data' / 'pde_dm_results_nc_cache.npz'"
    )

    gf_figure = gf_figure.replace(
        'plt.savefig("../../notebooks/figures/pde_gf_dm_comparison.pdf", dpi=SAVE_DPI)',
        "plt.savefig(FIG_DIR / 'pde_gf_dm_comparison.pdf', dpi=SAVE_DPI)"
    )

    write_notebook("dm_scenario_comparison", [
        ("markdown",
         "# Dark Matter Spectral Distortions: PDE vs GF\n\n"
         "Generates `pde_gf_dm_comparison.pdf` (Figure 4 in paper).\n\n"
         "Three DM scenarios (decay, s-wave, p-wave annihilation) compared "
         "across PDE solver, CosmoTherm GF convolution, and spectroxide GF table."),
        ("code", _SETUP + "\n" + gf_imports),
        ("markdown", "## DM scenario parameters"),
        ("code", gf_params),
        ("markdown", "## CosmoTherm GF convolution"),
        ("code", gf_ct_load),
        ("markdown", "## PDE GF table (build or load cache)"),
        ("code", gf_pde_table),
        ("markdown", "## Convolve PDE GF table with DM heating rates"),
        ("code", gf_convolve),
        ("markdown", "## Direct PDE runs"),
        ("code", gf_dm_pde),
        ("markdown", "## Strip G_bb from PDE"),
        ("code", gf_strip),
        ("markdown", "## Figure"),
        ("code", gf_figure),
    ])

    # ------------------------------------------------------------------
    # 5. Pathological Heating (Fig 5) — from dev/notebooks/remake_pathological_figure.ipynb
    # ------------------------------------------------------------------
    ph_cells = read_notebook_cells(root / "dev" / "notebooks" / "remake_pathological_figure.ipynb")
    ph_code = {i: src for i, (ct, src) in enumerate(ph_cells) if ct == "code"}

    def find_ph_cell(keyword):
        for i, src in ph_code.items():
            if keyword in src:
                return i, src
        return None, None

    _, ph_imports = find_ph_cell("from spectroxide import")
    _, ph_helpers = find_ph_cell("def strip_gbb")
    _, ph_scenarios = find_ph_cell("k_sin = 2 * np.pi")
    _, ph_gf_load = find_ph_cell("heat_table = load_or_build_greens_table")
    _, ph_compute = find_ph_cell("x_obs = np.linspace")
    _, ph_figure = find_ph_cell("fig, axes = plt.subplots")

    # Strip duplicate Path import (already in _SETUP)
    ph_imports = ph_imports.replace("from pathlib import Path\n\n", "")

    # Fix paths
    ph_figure = ph_figure.replace(
        'outpath = "../../notebooks/figures/pathological_heating_validation.pdf"\n'
        'fig.savefig(outpath, bbox_inches="tight")\n'
        'print(f"\\nSaved {outpath}")\n'
        'plt.close(fig)',
        "fig.savefig(FIG_DIR / 'pathological_heating_validation.pdf', bbox_inches='tight')\n"
        "plt.show()"
    )

    write_notebook("pathological_heating", [
        ("markdown",
         "# Pathological Heating Scenarios: PDE vs CosmoTherm GF\n\n"
         "Generates `pathological_heating_validation.pdf` (Figure 5 in paper).\n\n"
         "Three stress-test heating histories: sinusoidal, wide Gaussian, "
         "and double power-law. Compares PDE, PDE GF table, and CosmoTherm GF."),
        ("code", _SETUP + "\n" + ph_imports),
        ("markdown", "## Helpers"),
        ("code", ph_helpers),
        ("markdown", "## Define pathological heating functions"),
        ("code", ph_scenarios),
        ("markdown", "## Load Green's function tables"),
        ("code", ph_gf_load),
        ("markdown", "## Run PDE, GF table, and CosmoTherm GF for each scenario"),
        ("code", ph_compute),
        ("markdown", "## Figure"),
        ("code", ph_figure),
    ])

    # ------------------------------------------------------------------
    # 6. Photon Injection Spectra (Fig 6) — from notebooks/physics/photon_injection.ipynb
    # ------------------------------------------------------------------
    pi_cells = read_notebook_cells(root / "notebooks" / "physics" / "photon_injection.ipynb")
    pi_code = {i: src for i, (ct, src) in enumerate(pi_cells) if ct == "code"}

    def find_pi_cell(keyword):
        for i, src in pi_code.items():
            if keyword in src:
                return i, src
        return None, None

    _, pi_imports = find_pi_cell("from spectroxide import")  # cell 1
    _, pi_cache = find_pi_cell("pde_spectra = {}")          # cell 3
    _, pi_figure = find_pi_cell("photon_injection_spectra.pdf")  # cell 27

    # Fix paths
    pi_figure = pi_figure.replace(
        "plt.savefig('../figures/photon_injection_spectra.pdf', dpi=SAVE_DPI, bbox_inches='tight')",
        "plt.savefig(FIG_DIR / 'photon_injection_spectra.pdf', dpi=SAVE_DPI, bbox_inches='tight')"
    )

    write_notebook("photon_injection_spectra", [
        ("markdown",
         "# Photon Injection Spectral Distortions\n\n"
         "Generates `photon_injection_spectra.pdf` (Figure 6 in paper).\n\n"
         "Three panels (x_inj = 0.1, 1.0, 5.0) showing PDE (solid) and "
         "GF (dashed) spectra at four injection redshifts."),
        ("code", _SETUP + "\n" + pi_imports),
        ("code", pi_cache),
        ("markdown", "## Generate figure"),
        ("code", pi_figure),
    ])

    # ------------------------------------------------------------------
    # 7. FIRAS Photon Injection Limits (Fig 7) — from dev/scripts/remake_firas_photon_limits.py
    # ------------------------------------------------------------------
    fpl_src = (root / "dev" / "scripts" / "remake_firas_photon_limits.py").read_text()
    # Remove sys.path hack and fix output path
    fpl_lines = fpl_src.splitlines(True)
    fpl_code = ""
    for line in fpl_lines:
        if line.startswith('"""') or line.startswith("#!/"):
            continue
        if "sys.path.insert" in line:
            continue
        if "import sys" in line:
            continue
        if "Path(__file__)" in line and "FIG_DIR" in line:
            continue
        fpl_code += line

    write_notebook("firas_photon_limits", [
        ("markdown",
         "# FIRAS Photon Injection Limits\n\n"
         "Generates `firas_photon_limits_paper.pdf` (Figure 7 in paper).\n\n"
         "COBE/FIRAS 68% CL upper limits on Delta N_gamma / N_gamma "
         "as a function of injection frequency x_i at three redshifts."),
        ("code", _SETUP + "\nimport time\n" + fpl_code),
    ])

    # ------------------------------------------------------------------
    # 8. Dark Photon Constraints (Fig 8) — from notebooks/physics/dark_photon_validation.ipynb
    # ------------------------------------------------------------------
    dp_cells = read_notebook_cells(root / "notebooks" / "physics" / "dark_photon_validation.ipynb")
    dp_code = {i: src for i, (ct, src) in enumerate(dp_cells) if ct == "code"}

    def find_dp_cell(keyword):
        for i, src in dp_code.items():
            if keyword in src:
                return i, src
        return None, None

    _, dp_imports = find_dp_cell("from spectroxide import")          # cell 1
    _, dp_physics = find_dp_cell("def plasma_frequency_ev")         # cell 3
    _, dp_spectral = find_dp_cell("g1 = G1_PLANCK")                # cell 5
    _, dp_gamma = find_dp_cell("def dln_omega_pl2_dlna")            # cell 6
    _, dp_firas = find_dp_cell("firas = FIRASData()")               # cell 8
    _, dp_templates = find_dp_cell("def dp_gf_custom_template")     # cell 10
    _, dp_pde_worker = find_dp_cell("def _pde_worker")              # cell 11
    _, dp_pde_run = find_dp_cell("masses_low = np.geomspace")       # cell 12
    _, dp_ct_lims = find_dp_cell("cosmotherm_dp_lims")              # cell 16
    _, dp_figure = find_dp_cell("Publication figure: FIRAS dark photon")  # cell 17

    # Fix paths
    dp_ct_lims = dp_ct_lims.replace(
        "np.loadtxt('dev/data/cosmotherm_dp_lims.csv'",
        "np.loadtxt(PROJECT_ROOT / 'dev' / 'data' / 'cosmotherm_dp_lims.csv'"
    )
    dp_figure = dp_figure.replace(
        "fig.savefig('../figures/dp_firas_pde_constraints.pdf', bbox_inches='tight')",
        "fig.savefig(FIG_DIR / 'dp_firas_pde_constraints.pdf', bbox_inches='tight')"
    )
    # Remove the inline mass-vs-zres plot and validation tracking from dp_physics
    # Keep only the function definitions (up to the first fig/ax line)
    dp_physics_lines = dp_physics.splitlines(True)
    dp_physics_clean = []
    for line in dp_physics_lines:
        if line.startswith("fig, ax") or line.startswith("masses_ev ="):
            break
        dp_physics_clean.append(line)
    dp_physics = "".join(dp_physics_clean)
    dp_spectral = dp_spectral.split("\nresults.append")[0]
    dp_gamma = dp_gamma.split("\n# Recombination correction")[0]
    dp_pde_run = dp_pde_run.split("\n# Compare PDE vs GF")[0]

    # Remove 'results = []' from imports and unneeded plot code
    dp_imports = dp_imports.replace("results = []", "")

    write_notebook("dark_photon_constraints", [
        ("markdown",
         "# Dark Photon FIRAS Constraints\n\n"
         "Generates `dp_firas_pde_constraints.pdf` (Figure 8 in paper).\n\n"
         "FIRAS 95% CL upper limits on dark photon kinetic mixing epsilon "
         "as a function of mass, via PDE spectral fit with floating-T profile likelihood."),
        ("code", _SETUP + "\n" + dp_imports),
        ("markdown", "## Physics: plasma frequency, resonance redshift"),
        ("code", dp_physics),
        ("markdown", "## Spectral integrals"),
        ("code", dp_spectral),
        ("markdown", "## Conversion parameter gamma_con"),
        ("code", dp_gamma),
        ("markdown", "## FIRAS data and profile likelihood"),
        ("code", dp_firas),
        ("markdown", "## Dark photon templates and GF limits"),
        ("code", dp_templates),
        ("markdown", "## PDE worker and parallel execution"),
        ("code", dp_pde_worker),
        ("code", dp_pde_run),
        ("markdown", "## Load CosmoTherm reference and plot"),
        ("code", dp_ct_lims + "\n" + dp_figure),
    ])

    # ------------------------------------------------------------------
    # 9. Energy Conservation (Fig 10) — from dev/scripts/photon_energy_conservation.py
    # ------------------------------------------------------------------
    ec_src = (root / "dev" / "scripts" / "photon_energy_conservation.py").read_text()
    ec_lines = ec_src.splitlines(True)
    ec_code = ""
    for line in ec_lines:
        if line.startswith('"""') or line.startswith("#!/"):
            continue
        if "sys.path.insert" in line:
            continue
        if "import sys" in line:
            continue
        if "Path(__file__)" in line and "FIG_DIR" in line:
            continue
        ec_code += line

    # Split into heat injection and photon injection sections
    parts = ec_code.split("# ================================================================\n# Figure 2: Photon injection")
    heat_code = parts[0]
    photon_code = "# Photon injection" + parts[1] if len(parts) > 1 else ""

    write_notebook("energy_conservation", [
        ("markdown",
         "# Energy Conservation Diagnostics\n\n"
         "Generates `pde_energy_conservation.pdf` and "
         "`pde_energy_conservation_photon.pdf` (Figure 10 in paper).\n\n"
         "Left panel: heat injection. Right panel: photon injection at three x_inj."),
        ("code", _SETUP + "\n" + heat_code),
        ("markdown", "## Photon injection energy conservation"),
        ("code", photon_code),
    ])

    # ------------------------------------------------------------------
    # 10. Convergence Study (Fig 11) — from dev/scripts/convergence_figure.py
    # ------------------------------------------------------------------
    conv_src = (root / "dev" / "scripts" / "convergence_figure.py").read_text()
    conv_lines = conv_src.splitlines(True)
    conv_code = ""
    for line in conv_lines:
        if line.startswith('"""') or line.startswith("#!/"):
            continue
        if "sys.path.insert" in line:
            continue
        if "import sys" in line:
            continue
        if "Path(__file__)" in line and "FIG_DIR" in line:
            continue
        conv_code += line

    # Fix DATA_FILE and output paths
    conv_code = conv_code.replace(
        'DATA_FILE = Path("/tmp/convergence_data_full.txt")',
        "DATA_FILE = Path('/tmp/convergence_data_full.txt')"
    )
    conv_code = conv_code.replace(
        'outpath = FIG_DIR / "convergence_study.pdf"',
        "outpath = FIG_DIR / 'convergence_study.pdf'"
    )

    write_notebook("convergence_study", [
        ("markdown",
         "# Convergence Study\n\n"
         "Generates `convergence_study.pdf` (Figure 11 in paper).\n\n"
         "Two panels: (a) spatial convergence via spectral L2 error, "
         "(b) joint spatial+temporal convergence of mu and y.\n\n"
         "**Prerequisite**: Run the convergence test first:\n"
         "```bash\n"
         "cargo test --release convergence_study_full_data -- --nocapture "
         "> /tmp/convergence_data_full.txt 2>&1\n"
         "```"),
        ("code", _SETUP + "\n" + conv_code),
    ])


if __name__ == "__main__":
    print("Generating paper figure notebooks...")
    build_all()
    print("\nDone! 10 notebooks created in notebooks/paper_figures/")
