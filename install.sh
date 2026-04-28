#!/usr/bin/env bash
# ============================================================================
# spectroxide installer
#
# Usage:
#   ./install.sh                    # Install into current Python environment
#   ./install.sh --conda spectroxide  # Create/use conda env "spectroxide"
#   ./install.sh --extras notebook  # Install with notebook extras
#   ./install.sh --help             # Show all options
#
# What it does:
#   1. Installs Rust (via rustup) if not already present
#   2. Builds the Rust PDE solver in release mode
#   3. Installs the Python package (editable) with chosen extras
#   4. Runs a quick smoke test to verify everything works
# ============================================================================

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
CONDA_ENV=""
EXTRAS="plot"
SKIP_RUST_INSTALL=false
SKIP_BUILD=false
SKIP_PYTHON=false
SKIP_TEST=false
VERBOSE=false

# ── Colors (disabled if not a terminal) ─────────────────────────────────────
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    RED='\033[0;31m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    GREEN='' YELLOW='' RED='' BOLD='' NC=''
fi

info()  { echo -e "${GREEN}[spectroxide]${NC} $*"; }
warn()  { echo -e "${YELLOW}[spectroxide]${NC} $*"; }
error() { echo -e "${RED}[spectroxide]${NC} $*" >&2; }
die()   { error "$@"; exit 1; }

# ── Usage ───────────────────────────────────────────────────────────────────
usage() {
    cat <<'EOF'
spectroxide installer

USAGE:
    ./install.sh [OPTIONS]

OPTIONS:
    --conda ENV_NAME    Create (or activate) a conda/mamba environment
    --extras EXTRAS     Python extras to install (default: plot)
                        Options: plot, notebook, dev, doc
    --skip-rust         Skip Rust/rustup installation (use existing)
    --skip-build        Skip cargo build (binary already compiled)
    --skip-python       Skip Python package installation
    --skip-test         Skip the post-install smoke test
    --verbose           Show full build output
    -h, --help          Show this help message

EXAMPLES:
    # Minimal: just install into current env
    ./install.sh

    # Fresh conda env with Jupyter support
    ./install.sh --conda spectroxide --extras notebook

    # Developer setup
    ./install.sh --conda spectroxide-dev --extras dev

    # CI / Docker: Rust already present, skip smoke test
    ./install.sh --skip-rust --skip-test
EOF
    exit 0
}

# ── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --conda)    CONDA_ENV="$2"; shift 2 ;;
        --extras)   EXTRAS="$2"; shift 2 ;;
        --skip-rust)   SKIP_RUST_INSTALL=true; shift ;;
        --skip-build)  SKIP_BUILD=true; shift ;;
        --skip-python) SKIP_PYTHON=true; shift ;;
        --skip-test)   SKIP_TEST=true; shift ;;
        --verbose)     VERBOSE=true; shift ;;
        -h|--help)     usage ;;
        *) die "Unknown option: $1 (try --help)" ;;
    esac
done

# ── Locate repo root ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

if [[ ! -f "$REPO_ROOT/Cargo.toml" ]]; then
    die "Cannot find Cargo.toml. Run this script from the repository root."
fi

info "Repository root: ${BOLD}$REPO_ROOT${NC}"

# ── Step 1: Conda environment ──────────────────────────────────────────────
if [[ -n "$CONDA_ENV" ]]; then
    info "Setting up conda environment: ${BOLD}$CONDA_ENV${NC}"

    # Find conda/mamba (prefer conda; fall back to mamba/micromamba)
    CONDA_CMD=""
    if command -v conda &>/dev/null; then
        CONDA_CMD="conda"
    elif command -v mamba &>/dev/null; then
        CONDA_CMD="mamba"
    elif command -v micromamba &>/dev/null; then
        CONDA_CMD="micromamba"
    else
        die "No conda/mamba found. Install miniforge (https://github.com/conda-forge/miniforge) or pass --conda '' to skip."
    fi

    # Create env if it doesn't exist
    if ! $CONDA_CMD env list 2>/dev/null | grep -qw "$CONDA_ENV"; then
        info "Creating new environment '$CONDA_ENV' with Python 3.11..."
        if [[ "$VERBOSE" == true ]]; then
            $CONDA_CMD create -n "$CONDA_ENV" python=3.11 -y
        else
            $CONDA_CMD create -n "$CONDA_ENV" python=3.11 -y -q
        fi
    else
        info "Environment '$CONDA_ENV' already exists, reusing."
    fi

    # Activate the environment
    if [[ "$CONDA_CMD" == "micromamba" ]]; then
        eval "$(micromamba shell hook --shell bash)"
        micromamba activate "$CONDA_ENV"
    else
        # Source conda shell hook so `conda activate` works in scripts
        CONDA_BASE="$($CONDA_CMD info --base 2>/dev/null)" || die "Could not determine conda base path."
        if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
        else
            die "Cannot find conda.sh at $CONDA_BASE/etc/profile.d/conda.sh"
        fi
        conda activate "$CONDA_ENV" || die "Could not activate conda env '$CONDA_ENV'. Try: conda activate $CONDA_ENV && ./install.sh"
    fi

    # Verify we're actually in the right environment
    ACTIVE_ENV="$(basename "${CONDA_PREFIX:-}")"
    if [[ "$ACTIVE_ENV" != "$CONDA_ENV" ]]; then
        die "Activation failed: expected env '$CONDA_ENV' but got '$ACTIVE_ENV'. Try: conda activate $CONDA_ENV && ./install.sh"
    fi

    info "Using Python: $(which python) ($(python --version 2>&1))"
fi

# ── Step 2: Rust toolchain ─────────────────────────────────────────────────
ensure_rust() {
    # Add cargo to PATH if it exists but isn't on PATH
    if [[ -d "$HOME/.cargo/bin" ]]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    fi

    if command -v rustc &>/dev/null && command -v cargo &>/dev/null; then
        info "Rust already installed: $(rustc --version)"
        return 0
    fi
    return 1
}

if [[ "$SKIP_RUST_INSTALL" == false ]]; then
    if ! ensure_rust; then
        info "Installing Rust via rustup..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
        export PATH="$HOME/.cargo/bin:$PATH"

        if ! command -v cargo &>/dev/null; then
            die "Rust installation completed but cargo not found on PATH. Try restarting your shell."
        fi
        info "Rust installed: $(rustc --version)"
    fi
else
    if ! ensure_rust; then
        die "Rust not found and --skip-rust was specified. Install Rust first: https://rustup.rs"
    fi
fi

# ── Step 3: Build Rust binary ──────────────────────────────────────────────
if [[ "$SKIP_BUILD" == false ]]; then
    info "Building Rust PDE solver (release mode)..."
    if [[ "$VERBOSE" == true ]]; then
        cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml"
    else
        cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" 2>&1 | tail -5
    fi

    BINARY="$REPO_ROOT/target/release/spectroxide"
    if [[ ! -x "$BINARY" ]]; then
        die "Build succeeded but binary not found at $BINARY"
    fi
    info "Binary built: ${BOLD}$BINARY${NC}"
else
    info "Skipping Rust build (--skip-build)"
fi

# ── Step 4: Install Python package ─────────────────────────────────────────
if [[ "$SKIP_PYTHON" == false ]]; then
    # Check Python is available
    if ! command -v python &>/dev/null && ! command -v python3 &>/dev/null; then
        die "Python not found. Install Python 3.9+ or use --conda to create an environment."
    fi

    PYTHON_CMD="python"
    if ! command -v python &>/dev/null; then
        PYTHON_CMD="python3"
    fi

    # Check Python version
    PY_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PY_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
    PY_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")

    if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 9 ]]; }; then
        die "Python >= 3.9 required, found $PY_VERSION"
    fi

    info "Installing Python package (extras: ${BOLD}$EXTRAS${NC}) with Python $PY_VERSION..."
    if [[ "$VERBOSE" == true ]]; then
        $PYTHON_CMD -m pip install -e "$REPO_ROOT/python/.[$EXTRAS]"
    else
        $PYTHON_CMD -m pip install -q -e "$REPO_ROOT/python/.[$EXTRAS]"
    fi
    info "Python package installed."
else
    info "Skipping Python installation (--skip-python)"
fi

# ── Step 5: Smoke test ─────────────────────────────────────────────────────
if [[ "$SKIP_TEST" == false ]]; then
    info "Running smoke test..."

    # Test 1: Rust binary runs
    if "$REPO_ROOT/target/release/spectroxide" info &>/dev/null; then
        info "  Rust binary: OK"
    else
        warn "  Rust binary: FAILED (try: cargo build --release)"
    fi

    # Test 2: Python import works
    PYTHON_CMD="${PYTHON_CMD:-python}"
    if $PYTHON_CMD -c "
import spectroxide
import numpy as np
x = np.linspace(0.1, 20, 100)
result = spectroxide.run_single(z_h=5e4, delta_rho=1e-5, x=x)
assert result is not None and len(result['delta_n']) == 100
print('  Python import + Greens function: OK')
" 2>&1; then
        true  # printed inside Python
    else
        warn "  Python smoke test: FAILED"
    fi

    # Test 3: PDE solver runs (quick single burst)
    if $PYTHON_CMD -c "
from spectroxide import run_sweep
result = run_sweep(z_injections=[5e4], delta_rho=1e-5, n_points=200)
assert 'results' in result and len(result['results']) > 0
print('  PDE solver (via Python): OK')
" 2>&1; then
        true
    else
        warn "  PDE solver smoke test: FAILED (binary may not be built)"
    fi
fi

# ── Done ────────────────────────────────────────────────────────────────────
echo ""
info "${BOLD}Installation complete!${NC}"
echo ""
echo "  Quick start:"
echo "    python -c 'from spectroxide import run_sweep; print(\"ready!\")'"
echo ""
echo "  Run tests:"
echo "    cargo test --release        # Rust tests (416)"
echo "    cd python && pytest         # Python tests"
echo ""
if [[ -n "$CONDA_ENV" ]]; then
    echo "  Remember to activate your environment in new shells:"
    echo "    conda activate $CONDA_ENV"
    echo ""
fi
