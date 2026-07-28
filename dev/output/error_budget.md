# Numerical error budget (audit Phase 2, B3)

Relative discretization-error estimates for the production solver,
from Richardson extrapolation of the self-convergence suites and the
true-error MMS suite (`tests/mms_convergence.rs`). These set the
floor for benchmark-pack tolerances (plan Part A).

| Error source | Setting | Relative error | Method |
|---|---|---|---|
| Spatial (μ, full physics) | N=2000 | 1.77e-03 | Richardson p=1.97 over N=[500, 1000, 2000, 4000] |
| Spatial (μ, full physics) | N=4000 | 4.52e-04 | Richardson p=1.97 over N=[500, 1000, 2000, 4000] |
| Temporal (μ, full physics, production defaults) | dtau_max=10 | 2.86e-03 | direct dtau_max refinement, Richardson p=1.00 |
| Temporal (y, full physics, production defaults) | dtau_max=10 | 1.44e-03 | direct dtau_max refinement, Richardson p=1.00 |
| Spatial spectrum, MMS true error (Kompaneets, production grid) | N=2000 | 9.14e-05 | exact manufactured solution, rel. x³-weighted L2 |
| Spatial spectrum, MMS true error (Kompaneets, production grid) | N=4000 | 2.28e-05 | exact manufactured solution, rel. x³-weighted L2 |

Additional floors (measured elsewhere in the Phase 2 suites):

- Newton-tolerance conservation floor: ~2×10⁻⁸ relative photon-number
  drift over 10³ steps (`conservation_fuzz.rs`, pure Compton).
- Energy-ledger closure under fuzzing: ≤ 0.7% across randomized heat
  scenarios at N=800–1600 (dominated by spatial truncation).
- Kernel MMS orders: Kompaneets CN 2.0 (space and time), coupled
  DC/BR backward Euler 1.0 (time) / 2.0 (space) — design orders
  confirmed against exact solutions.

Regenerate: `python dev/scripts/error_budget.py` after refreshing the
logs (see module docstring).
