# Add a session default cosmology to the Python package

## Status

Proposed, 2026-09-25.

## Context

A user who wants Planck 2018 must pass `cosmo=` (or `cosmo_params=`) to every call. A script
that forgets one call silently mixes the Planck 2018 background with the Chluba (2013) default
(h = 0.71, Ω_b = 0.044), and nothing flags the mismatch.

Today `cosmo=None` means different things in different places:

| Function | `cosmo=None` means |
|---|---|
| `cosmology.py`, `dark_photon.py`, `axion.py` helpers | `DEFAULT_COSMO` |
| `solve(method="pde")`, `run_sweep()`, photon sweeps, table builders | no cosmology flags, so the Rust default (same values) |
| `photon_survival_probability`, `greens_function_photon` | the analytic `P_s = exp(−x_c/x)` (Chluba 2015, Eq. 24), a different *method*, not only different parameters |
| `solve(method="greens_function")` (heat) | not cosmology-aware; passing `cosmo=` raises |
| `solve(method="table")` | the table's build cosmology; passing `cosmo=` raises; the table does not check it |
| `cosmotherm.py` loaders | `COSMOTHERM_GF_COSMO` (n_eff = 3.04), chosen to match CosmoTherm's files |

A session default has to say what happens in each row, or it adds a seventh meaning.

Precedent: `astropy.cosmology.default_cosmology` is a module-level setting that also works as a
context manager.

## Decision

Add a session default to `spectroxide.cosmology`, re-exported at top level:

```python
spectroxide.set_default_cosmology(Cosmology.planck2018())   # whole script
with spectroxide.default_cosmology(Cosmology.planck2018()): # one block
    ...
spectroxide.get_default_cosmology()                          # current value
```

`set_default_cosmology(None)` restores the Chluba (2013) parameters. An explicit `cosmo=` always
wins. The setting is a plain module-level global, and the context manager swaps it and restores
the old value on exit. A `contextvars.ContextVar` was rejected: new threads start with an empty
context, so a `ThreadPoolExecutor` worker would silently fall back to the Chluba (2013) values
after a script-wide `set_default_cosmology`. `DEFAULT_COSMO` stays a constant and keeps naming
the Chluba (2013) parameters.

Row by row:

- **Background helpers** (`hubble`, `ionization_fraction`, dark-photon and axion helpers):
  `cosmo=None` resolves to the session default.
- **PDE paths** (`solve`, `run_sweep`, photon sweeps, table builders): if the session default
  differs from the Chluba (2013) parameters, the wrapper passes it to the binary as flags.
  Otherwise it passes nothing, as now, so default runs are byte-identical.
- **Photon Green's function and `P_s`**: `cosmo=None` keeps the analytic form while the session
  default is the Chluba (2013) set. If the user has set a different default, the functions use
  the cosmology-aware path with it. Setting a cosmology is a request to use it wherever the code
  can.
- **Heat Green's function**: unchanged. It stays cosmology-independent. If the session default is
  not the Chluba (2013) set, `solve(method="greens_function")` warns once per process that the
  heat Green's function ignores it.
- **Table mode**: tables built after this change always record their cosmology in the metadata.
  If the session default differs from a table's recorded cosmology, `solve(method="table")`
  raises. Tables without the record warn instead.
- **CosmoTherm loaders**: unchanged. They keep `COSMOTHERM_GF_COSMO`, because they exist to match
  an external code's parameters, not the user's.

## Consequences

- A script sets its cosmology once, and every cosmology-aware call uses it.
- Behavior of every existing call with no session default set is unchanged, and PDE output is
  byte-identical.
- Hidden state: a notebook cell that sets the default changes every later cell, including cells
  run out of order. The context manager limits this; the paper-figure notebooks should keep
  passing `cosmo=` explicitly so that a figure never depends on cell order.
- Worker processes started with the `spawn` method do not inherit the setting. The package
  starts none itself (it calls the Rust binary with `subprocess`), but user code with
  `multiprocessing` must set the default in each worker.
- `photon_survival_probability(x, z)` with no `cosmo=` can now return the cosmology-aware value.
  At z ≤ 5e4 and x ≲ x_c this differs from the analytic form by orders of magnitude, so the
  docstring must say that the session default switches the method.
- The Python default can now differ from the CLI and Rust defaults, which ADR 0010 aligned. This
  is a user choice, not a default drift, and the Rust side does not change.
- The context manager is not thread-safe: two threads that enter it at once overwrite each
  other's value. The docstring must say so.
- Tests: resolution order (explicit, context manager, session, Chluba 2013), the context manager
  restoring on exceptions, a thread-pool worker seeing the session value, CLI flags emitted only when the default differs,
  the heat-GF warning, and the table mismatch error.
