# Review: notebooks/tutorials/ against the Google developer documentation style guide

Scope: the six notebooks in `notebooks/tutorials/` (`01_getting_started.ipynb` through
`06_greens_table.ipynb`). Reviewed with `python dev/scripts/nb_md_replace.py --show NOTEBOOK`
and a line-numbered dump of Markdown-cell sources only (`.ipynb` files were never opened with a
text reader). Rust/Python doc comments are out of scope for this file. Line numbers are 1-indexed
within each Markdown cell's source, matching the numbering the review script prints; "cell N"
is the zero-based cell index reported by `nb_md_replace.py --show`.

Every code cell in all six notebooks is immediately preceded by a Markdown cell, so no notebook
has a bare code block dropped in with zero lead-in text. That check found no findings to report,
with one caveat noted under "Systematic patterns."

## Summary of counts by severity

| File | HIGH | MEDIUM | LOW | DOMAIN | Total |
|---|---|---|---|---|---|
| 01_getting_started.ipynb | 0 | 22 | 12 | 0 | 34 |
| 02_energy_injection.ipynb | 0 | 6 | 6 | 0 | 12 |
| 03_new_physics.ipynb | 0 | 5 | 3 | 0 | 8 |
| 04_custom_scenarios.ipynb | 0 | 10 | 2 | 0 | 12 |
| 05_observational_constraints.ipynb | 1 | 4 | 5 | 2 | 12 |
| 06_greens_table.ipynb | 0 | 7 | 0 | 0 | 7 |
| **Total** | **1** | **54** | **28** | **2** | **85** |

## Systematic patterns

- **Bold used for term introduction (rule F6, 17 instances across 5 files).** Every notebook
  except `04` opens a bulleted or numbered list with a bold term or figure label followed by an
  em dash and a description (for example `**Green's function** ... —`, `**Fig. 7** —`). The guide
  reserves bold for UI elements and sparingly for run-in headings/notice labels; term introduction
  belongs in italics. The `**Next:**` label that closes several notebooks is a defensible run-in
  heading and is not flagged.
- **Em dash set off with spaces (rule F11, 25 instances across 5 files).** All narrative asides
  use `word — word` instead of Google's unspaced `word—word`. `01_getting_started.ipynb` alone has
  11 instances; the first 10 are listed below and the count is noted.
- **"The table below ..." (rule T9, 6 instances across 5 of 6 files).** Every notebook's Summary
  section (and one mid-notebook table) refers to the table by document position instead of a
  sentence that stands alone or a cross-reference.
- **Numbered lists for non-sequential sets (rule L1, 3 instances in files 02–04).** Three
  notebooks number a list of scenario types or API entry points that has no required order; Google
  style reserves numbered lists for sequential steps. `05_observational_constraints.ipynb` shows
  the opposite failure once: a genuine 5-step procedure is written as roman numerals inside one
  prose sentence instead of as a numbered list (HIGH — wrong list type for a procedure).
- **Dropped-subject fragments styled as sentences (rule T14, 13 instances across files 01, 04, 06).**
  A second sentence in a paragraph or bullet frequently drops the subject ("Captures the...",
  "Useful for plotting...", "Internally converted to...", "Solver tabulates..."). List items that
  are legitimate parallel noun phrases (for example the three-item lists in `02` and the bulleted
  case list in `03`) are not flagged as fragments — only prose sentences missing a subject or verb.
- **Range punctuation is inconsistent (rule F8, 3 instances).** `01_getting_started.ipynb` writes
  "2-5%" and "5-20 s" with a plain hyphen in its opening cell, then "2–5%" and "5–20 s" with an en
  dash everywhere else in the same notebook and across the other five notebooks.
  `06_greens_table.ipynb`'s only range, "30-70%", also uses a plain hyphen, inconsistent with the
  en-dash convention the rest of the tutorial series follows.
- **`dict` used as a bare word instead of code font (rule F1, 3 instances in files 01–03).**
  "a dict", "injection dict", "injection dict or call" name the Python type without code font.
- **Import cells have no dedicated lead-in sentence.** Cell 1 of every notebook (`import numpy as
  np`, etc.) follows only the notebook's opening paragraph, which introduces the notebook's topic
  but not the import itself. This is a common, low-friction notebook convention rather than a bare
  code block after a heading, so it is not listed as a line-item finding, but it is noted here for
  completeness since the brief asked whether every code cell is introduced.

## 01_getting_started.ipynb

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| cell 0, L3 | T14 | MEDIUM | "Quick-start tour of the two solver modes:" | Give it a subject: "This notebook is a quick tour of the two solver modes:" |
| cell 0, L5 | F6 | MEDIUM | "**Green's function** (`method=\"greens_function\"`)" | Use italics for the term introduction: "*Green's function*" |
| cell 0, L5 | F8 | MEDIUM | "2-5% accurate in the deep" | Use an en dash for the range: "2–5%", matching cell 4/10/16 in this file |
| cell 0, L5 | F11 | LOW | "milliseconds. 2-5% accurate" — dash example: "`method=\"greens_function\"`) — pure-Python" | Remove the spaces around the em dash: "greens_function\")—pure-Python" (1 of 11 in this file; see systematic patterns) |
| cell 0, L6 | F6 | MEDIUM | "**Partial differential equation (PDE)** (`method=\"pde\"`)" | Use italics: "*Partial differential equation (PDE)*" |
| cell 0, L6 | F8 | MEDIUM | "takes about 5-20 s per solve" | Use an en dash: "5–20 s", matching cell 6 in this file |
| cell 0, L6 | F11 | LOW | "through the Rust binary. Reference accuracy" — dash example: "(`method=\"pde\"`) — full Kompaneets" | Remove spaces around the dash (2 of 11) |
| cell 0, L6 | T14 | MEDIUM | "Reference accuracy; takes about 5-20 s per solve." | Give it a subject and verb: "It gives reference accuracy and takes about 5–20 s per solve." |
| cell 0, L8 | T14 | MEDIUM | "Walks through distortion shapes, single-burst injection, parallel parameter sweeps, intensity conversion, and cosmology presets." | Add a subject: "This notebook walks through distortion shapes, ..." |
| cell 2, L5 | F6 | MEDIUM | "**$\mu$-distortion** $M(x)$" | Use italics for the term |
| cell 2, L5 | F11 | LOW | "$M(x)$ — Bose-Einstein with chemical potential" | Remove spaces around the dash (3 of 11) |
| cell 2, L6 | F6 | MEDIUM | "**$y$-distortion** $Y_{SZ}(x)$" | Use italics for the term |
| cell 2, L6 | F11 | LOW | "$Y_{SZ}(x)$ — frequency redistribution without thermalization" | Remove spaces around the dash (4 of 11) |
| cell 2, L6 | T14 | MEDIUM | "Late-time injection ($z \lesssim 5\times 10^4$)." | Give it a verb: "It is a late-time injection ($z \lesssim 5\times 10^4$)." |
| cell 2, L7 | F6 | MEDIUM | "**Temperature shift** $G_{bb}(x) = x\,n_{pl}(1+n_{pl})$" | Use italics for the term |
| cell 2, L7 | F11 | LOW | "$G_{bb}(x) = x\,n_{pl}(1+n_{pl})$ — degenerate with $T_{CMB}$" | Remove spaces around the dash (5 of 11) |
| cell 4, L3 | T14 | MEDIUM | "Pure Python, vectorized, evaluates in milliseconds." | Add a subject: "It is pure Python, vectorized, and evaluates in milliseconds." |
| cell 6, L3 | T14 | MEDIUM | "Captures the $\mu$–$y$ transition and small nonlinear corrections that the Green's function misses." | Add a subject: "It captures the $\mu$–$y$ transition..." |
| cell 8, L1 | H3 | MEDIUM | "## 4. Parallel sweeps with `run_sweep`" | Drop the code font from the heading: "## 4. Parallel sweeps with run_sweep" |
| cell 8, L3 | F3 | MEDIUM | "pass `n_threads=N` to cap it" | State what `N` stands for: "pass `n_threads=N`, where `N` is the number of worker threads, to cap it" |
| cell 8, L5 | F11 | LOW | "a `results` list — one entry per injection redshift" | Remove spaces around the dash (6 of 11) |
| cell 8, L5 | F1 | LOW | "The return is a dict with a `results` list" | Use code font: "a `dict` with a `results` list" |
| cell 16, L3 | T9 | MEDIUM | "The table below compares the three solve modes by speed, accuracy, and when to use each one." | Refer to the table without a position word: "This table compares the three solve modes..." |
| cell 16, L11 | L2 | MEDIUM | "Next:" | Introduce the list with a full sentence: "Continue with:" or "Next, read:" |
| cell 16, L12 | F6 | MEDIUM | "**02 Energy injection** — built-in scenarios (decaying particles, dark matter)" | Use italics or a description-list term instead of bold |
| cell 16, L12 | F11 | LOW | same line, "**02 Energy injection** — built-in scenarios" | Remove spaces around the dash (7 of 11) |
| cell 16, L13 | F6 | MEDIUM | "**03 New physics** — dark photon, monochromatic photon injection" | Use italics instead of bold |
| cell 16, L13 | F11 | LOW | same line | Remove spaces around the dash (8 of 11) |
| cell 16, L14 | F6 | MEDIUM | "**04 Custom scenarios** — user-defined `dq_dz` and photon sources" | Use italics instead of bold |
| cell 16, L14 | F11 | LOW | same line | Remove spaces around the dash (9 of 11) |
| cell 16, L15 | F6 | MEDIUM | "**05 Observational constraints** — Far Infrared Absolute Spectrophotometer limits and full-spectrum fitting" | Use italics instead of bold |
| cell 16, L15 | F11 | LOW | same line | Remove spaces around the dash (10 of 11; 1 further instance at cell 16, L16 not itemized — 11 total in this file) |
| cell 16, L16 | F6 | MEDIUM | "**06 Green's function tables** — PDE-accurate interpolation at GF speed" | Use italics instead of bold |
| cell 16, L16 | T12 | LOW | "PDE-accurate interpolation at GF speed" | Spell out on first use: "at Green's function (GF) speed," since "GF" is never expanded in this notebook |

## 02_energy_injection.ipynb

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| cell 0, L5 | F6 | MEDIUM | "**Decaying particles** — exponential decay of a long-lived relic" | Use italics instead of bold |
| cell 0, L5 | F11 | LOW | same line | Remove spaces around the dash (1 of 4) |
| cell 0, L6 | F6 | MEDIUM | "**Dark matter (DM) annihilation (s-wave)** — $\langle\sigma v\rangle$ constant" | Use italics instead of bold |
| cell 0, L6 | F11 | LOW | same line | Remove spaces around the dash (2 of 4) |
| cell 0, L7 | F6 | MEDIUM | "**DM annihilation (p-wave)** — $\langle\sigma v\rangle \propto v^2 \propto (1+z)$" | Use italics instead of bold |
| cell 0, L7 | F11 | LOW | same line | Remove spaces around the dash (3 of 4) |
| cell 0, L5-7 | L1 | MEDIUM | "1. **Decaying particles** ...\n2. **Dark matter (DM) annihilation (s-wave)** ...\n3. **DM annihilation (p-wave)** ..." | These three scenario types have no required order; use a bulleted list |
| cell 4, L3 | T9 | MEDIUM | "The table below compares how the two annihilation channels scale with redshift and which spectral distortion each one produces." | Drop the position reference: "This table compares..." |
| cell 4, L5 | L5 | LOW | "| | s-wave | p-wave |" | Label the blank corner cell, for example "Quantity" |
| cell 8, L3 | T9 | MEDIUM | "The table below lists the injection dictionary for each scenario." | Drop the position reference: "This table lists..." |
| cell 8, L5 | F1 | LOW | "| Scenario | injection dict |" | Use code font: "injection `dict`" |
| cell 8, L13 | F11 | LOW | "`03_new_physics.ipynb` — dark photon oscillation, monochromatic photon injection." | Remove spaces around the dash (4 of 4) |

## 03_new_physics.ipynb

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| cell 0, L3 | T14 | MEDIUM | "Two scenarios that go beyond heat injection with the partial differential equation (PDE) solver:" | Add a main verb: "Two scenarios go beyond heat injection with the PDE solver:" |
| cell 0, L5-6 | L1 | MEDIUM | "1. **Dark photon depletion** — ...\n2. **Monochromatic photon injection** — ..." | These two scenarios have no required order; use a bulleted list |
| cell 0, L5 | F6 | MEDIUM | "**Dark photon depletion** — $\gamma\to A'$ resonant conversion removes cosmic microwave background photons" | Use italics instead of bold |
| cell 0, L5 | F11 | LOW | same line | Remove spaces around the dash (1 of 2) |
| cell 0, L6 | F6 | MEDIUM | "**Monochromatic photon injection** — line emission at fixed $x_{\rm inj}$" | Use italics instead of bold |
| cell 0, L6 | F11 | LOW | same line | Remove spaces around the dash (2 of 2) |
| cell 8, L3 | T9 | MEDIUM | "The table below lists the injection dictionary or function call for each scenario." | Drop the position reference: "This table lists..." |
| cell 8, L5 | F1 | LOW | "| Scenario | injection dict or call |" | Use code font: "injection `dict` or call" |

## 04_custom_scenarios.ipynb

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| cell 0, L3 | T14 | MEDIUM | "Two custom-API hooks for user-defined injection physics:" | Add a verb: "Two custom-API hooks let you plug in injection physics:" |
| cell 0, L5-6 | L1 | MEDIUM | "1. `dq_dz(z)` — ...\n2. `photon_source(x, z)` — ..." | These two hooks have no required order; use a bulleted list |
| cell 0, L5 | F11 | LOW | "`dq_dz(z)` — heating rate $d(\Delta\rho/\rho)/dz$" | Remove spaces around the dash (1 of 2) |
| cell 0, L5 | T14 | MEDIUM | "Works in Green's function (GF) and partial differential equation (PDE) modes." | Add a subject: "It works in Green's function (GF) and PDE modes." |
| cell 0, L6 | F11 | LOW | "`photon_source(x, z)` — frequency-resolved $d(\Delta n)/dz$" | Remove spaces around the dash (2 of 2) |
| cell 0, L6 | T14 | MEDIUM | "PDE only." | Expand it: "It applies to the PDE solver only." |
| cell 6, L3 | T14 | MEDIUM | "Same Gaussian, full nonlinear PDE." | Add a verb: "This uses the same Gaussian, run through the full nonlinear PDE." |
| cell 6, L3 | T14 | MEDIUM | "Solver tabulates `dq_dz` on a redshift grid internally." | Restore the article: "The solver tabulates `dq_dz`..." |
| cell 10, L3 | T14 | MEDIUM | "Two Gaussians: 70% in $\mu$-era ($z=3\times10^5$), 30% in $y$-era ($z=5\times10^3$)." | Add a verb: "Two Gaussians contribute: 70% in the $\mu$-era..., 30% in the $y$-era..." |
| cell 12, L3 | T14 | MEDIUM | "Internally converted to the Boltzmann source $S = (d\Delta n/dz)\,|dz/d\tau|$." | Add a subject: "The solver converts it internally to the Boltzmann source..." |
| cell 16, L3 | T9 | MEDIUM | "The table below lists the call for each custom-injection mode." | Drop the position reference: "This table lists..." |
| cell 16, L11-14 | L3 | MEDIUM | "- `dq_dz(z)` returns ... (positive = heating)\n- `photon_source(x, z)` returns ...\n- Set `z_start` = ...\n- Always verify normalization with `np.trapezoid`" | Make all four items the same grammatical form; either state facts throughout or give instructions throughout |

## 05_observational_constraints.ipynb

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| cell 0, L1 | T12 | LOW | "# FIRAS limits on monochromatic photon injection and dark photon mixing" | Spell it out in the title too, or move the definition ahead of first use: "Far Infrared Absolute Spectrophotometer (FIRAS) limits on..." |
| cell 0, L5 | F6 | MEDIUM | "**Fig. 7** — FIRAS upper limit on monochromatic photon injection" | Use italics or a description-list term instead of bold |
| cell 0, L5 | F11 | LOW | same line | Remove spaces around the dash (1 of 4) |
| cell 0, L6 | F6 | MEDIUM | "**Fig. 8** — FIRAS upper limit on dark-photon kinetic mixing" | Use italics instead of bold |
| cell 0, L6 | F11 | LOW | same line | Remove spaces around the dash (2 of 4) |
| cell 0, L6 | T11 | DOMAIN | "(Chluba, Cyr & Johnson 2024)" | Standard citation shorthand; spelling out "and" is a style choice, not required |
| cell 0, L15 | F8 | LOW | "this tutorial runs 3 points each to demonstrate the workflow" | Spell out the single-digit count: "three points" |
| cell 2, L7 | F11 | LOW | "(Chluba 2015) — energy- and number-injection cancel — so the limit is weakest there" | Remove spaces around both dashes (3 and 4 of 4) |
| cell 6, L7 | T11 | DOMAIN | "(narrow-width approximation, Mirizzi et al. 2009; Chluba & Cyr 2024)" | Standard citation shorthand |
| cell 6, L9 | L1 | HIGH | "**Pipeline.** For each mass: (i) find $z_{\rm res}$ from the plasma frequency; (ii) compute $\gamma_{\rm con}/\epsilon^2$; (iii) run the PDE with `dark_photon_resonance` injection at a reference $\epsilon$; (iv) build a per-$\gamma_{\rm con}$ template $\Delta n(x)$ and fit it to FIRAS using `firas.profile_limit_floating_T`; (v) translate the bound on $\gamma_{\rm con}$ back to $\epsilon$..." | This is a 5-step sequential procedure; use a numbered list, one action per step, instead of roman numerals inline in one sentence |
| cell 6, L9 | F11 | MEDIUM | same sentence, four semicolons chaining five clauses | Break into a numbered list (see above) instead of a semicolon chain |
| cell 8, L1 | T9 | MEDIUM | "using the three points computed above" | Refer to it without a position word, or name the section: "using the three points from the previous step" |

## 06_greens_table.ipynb

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| cell 0, L3 | F8 | MEDIUM | "has 30-70% shape errors in the $\mu\to y$ transition" | Use an en dash: "30–70%", matching the en-dash convention the rest of the tutorial series uses for ranges |
| cell 0, L3 | T14 | MEDIUM | "CosmoTherm-style fix: precompute $G_{\rm th}(x, z_h)$ from partial differential equation (PDE) runs, interpolate." | Give it a subject: "The CosmoTherm-style fix precomputes $G_{\rm th}(x, z_h)$ from PDE runs and interpolates." |
| cell 2, L3 | T14 | MEDIUM | "Demo uses 30 redshifts; production tables use 150+." | Restore the article: "The demo uses 30 redshifts;..." |
| cell 2, L3 | T11 | MEDIUM | "production tables use 150+" | Spell it out: "150 or more" |
| cell 9, L3 | F2 | MEDIUM | "$d(\Delta\rho/\rho)/dz = f_X\Gamma_X e^{-\Gamma_X t(z)} / [H(z)(1+z)]$." (the only text before the following code cell) | Add a sentence describing what the code computes, not just the bare formula |
| cell 11, L1 | H3 | MEDIUM | "## 5. Caching and `solve(method=\"table\")`" | Drop the code font from the heading: "## 5. Caching and the table solve method" |
| cell 13, L3 | T9 | MEDIUM | "The table below lists the main table-building and lookup functions and what each one does." | Drop the position reference: "This table lists..." |
