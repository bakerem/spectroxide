# Documentation review: README.md and data/cosmotherm/README.md

Reviewed against the Google developer documentation style guide rubric (rule IDs T1–T15, H1–H5,
L1–L5, F1–F12, A1–A7). Read-only review; no files edited.

## Summary of counts by severity

| Severity | README.md | data/cosmotherm/README.md | Total |
|----------|-----------|----------------------------|-------|
| HIGH     | 1         | 0                          | 1     |
| MEDIUM   | 6         | 3                          | 9     |
| LOW      | 6         | 3                          | 9     |
| DOMAIN   | 1         | 1                          | 2     |
| **Total**| **14**    | **7**                      | **21**|

## Systematic patterns

- **Spaced em dash (F11, LOW).** README.md uses `" — "` (space–em dash–space) throughout,
  the Chicago-style convention, not Google's unspaced em dash. 11 instances:
  line 9 (×2), 25, 250, 295, 330, 331, 332, 333, 339, 349 (first 10 listed; 349 is the 11th).
  One representative row is included in the findings table below; the rest are not repeated
  individually to avoid padding.
- **Undefined abbreviations in terse architecture/inline annotations.** NWA (README, HIGH —
  used in the quick-start example, never expanded), IMEX, ΛCDM/LCDM, LLM (README); DI, CDM
  (data/cosmotherm/README.md). These are annotations in a file tree or a data-parameter aside,
  not full sentences, but the abbreviations themselves are still undefined anywhere in the page.
- **"vs" without a period in file-tree comments** (README.md lines 305, 314, 315, 317) — T9 flags
  "vs." in prose; these are terse comments, not full sentences, so severity is LOW rather than
  MEDIUM.
- Both files otherwise follow second person / active voice / present tense well in the main
  prose (no "we", no "will", no pre-announcing found in flowing prose).

## Findings: README.md

| Line | Rule | Severity | Quotation | Suggested fix |
|------|------|----------|-----------|----------------|
| 144 (also 278, 295, 296) | T12 | HIGH | `# Dark photon oscillation (NWA resonant conversion)` | Spell out "narrow-width approximation (NWA)" at first use; it is never expanded anywhere in the file even though it recurs 4 times. |
| 272 | T12 | LOW | `├── kompaneets.rs          # Compton scattering (IMEX Newton solver)` | Spell out "implicit–explicit (IMEX)" or drop the acronym; used once, not central to using the library. |
| 280, 290 | T12 | DOMAIN | `├── cosmology.rs           # Flat LCDM background` | Expand to "Lambda cold dark matter (ΛCDM)" at first use, or accept as standard field notation for a cosmology audience. |
| 326–327 | T5 | MEDIUM | `If you use spectroxide in your research, please cite the accompanying paper` | Drop "please": "Cite the accompanying paper..." (second instance at line 327, "please also cite Chluba (2013) and Chluba (2015)"). |
| 9, 25, 250, 295, 330–333, 339 | F11 | LOW | `These spectral distortions — μ-type (chemical potential) and y-type (Compton) — encode information` | Use unspaced em dashes consistently ("distortions—μ-type…—encode"); 11 instances total across the file (see Systematic patterns). |
| 339 | F11 | MEDIUM | `Contributions — new injection scenarios, improved physics, validation, bug fixes ---` | The same construction mixes a real em dash with a literal `---`. Use one em dash character throughout: "Contributions—new injection scenarios, improved physics, validation, and bug fixes—are welcome." |
| 13–15 | L2 | MEDIUM | `## Features` / `- **Full PDE solver** in Rust: implicit Kompaneets and coupled double Compton (DC) and bremsstrahlung (BR) with adaptive stepping` | Add an introductory sentence ending in a colon before the bullet list, for example "spectroxide provides:". |
| 17 | F8 | LOW | `**9 built-in injection scenarios**: single burst, decaying particles (heat or photon channel), dark matter (DM) annihilation (s-wave or p-wave), dark photon oscillation, monochromatic photon injection, and tabulated sources (plus custom heating through the Rust API)` | Spell out "Nine" so the item does not start with a numeral. |
| 252–254 | F11 | MEDIUM | `Units: $f_X$ [eV] is the energy released per baryon ($\Gamma_X$ [1/s] the decay rate); $f_{\rm ann}$ [eV/s] is energy per baryon per second; $\Delta N/N$ is the fractional photon-number injection.` | Break the semicolon chain into three sentences or a description list (one term per line). |
| 341 | T12 | MEDIUM | `codebase) are written with LLM assistance, and the workflow is built around that:` | Spell out "large language model (LLM)" at first use (also used at line 348 without definition). |
| 341 | T2 | MEDIUM | `Most contributions to spectroxide (including the bulk of the original codebase) are written with LLM assistance, and the workflow is built around that:` | Rewrite in active voice, naming who acts: "A human and an LLM write most contributions together..." |
| 151, 206, 208 | F2 | LOW | `sweep = run_sweep(z_injections=np.geomspace(2e3, 5e5, 15).tolist(), delta_rho=1e-5)` | Wrap or shorten the 3 code-sample lines that exceed 80 characters (151, 206, 208). |
| 305, 314, 315, 317 | T9 | LOW | `├── cosmotherm_comparison.rs # PDE vs CosmoTherm reference data` | Replace "vs" with "versus" or rephrase ("...compared with CosmoTherm reference data"); 4 instances total. |
| 132 | T11 | LOW | `# gamma_x = 1e-10 => lifetime ~1e10 s, decay peaks near z ~ 5e4)` | Replace the arrow "=>" with a word: "gamma_x = 1e-10 means lifetime ~1e10 s". |

## Findings: data/cosmotherm/README.md

| Line | Rule | Severity | Quotation | Suggested fix |
|------|------|----------|-----------|----------------|
| 6 | H3 | MEDIUM | `## DI files (included in repo)` | "DI" is undefined at the heading itself; the expansion ("distortion intensity") appears only two lines later in body text. Define it in or before the heading, for example "## Distortion-intensity (DI) files". |
| 22 | F2 | MEDIUM | ` ``` ` (opening fence of the cosmology-parameter block, no language) | Add a language hint (for example ` ```text `) or, better, present the parameters as a description list instead of a fenced block. |
| 41 | F4 | MEDIUM | `Source: https://www.jb.man.ac.uk/~jchluba/Science/CosmoTherm/Download.html` | Use a descriptive markdown link instead of a bare URL: "Source: [Chluba's CosmoTherm download page](https://www.jb.man.ac.uk/~jchluba/Science/CosmoTherm/Download.html)." |
| 25 | T12 | LOW | `but it is the CDM fraction —` | Spell out "cold dark matter (CDM)" at first use. |
| 43 | F1 | LOW | `The Green's function (GF) database uses spectroxide's default cosmology (h=0.71, Omega_b=0.044) matching` | Put parameter names/values in code font: `` h=0.71 ``, `` Omega_b=0.044 ``. |
| 25 | F11 | LOW | `the CDM fraction —` (line-final, followed by "it matches Planck-2015 Omega_cdm to 6 digits" on the next line) | Use an unspaced em dash consistently with Google style. |
| 22–30 | L1 | DOMAIN | `Y_p = 0.2467, T_CMB = 2.726 K` | This term–value parameter listing is formatted as a fenced code block rather than a description list; acceptable as a domain convention for citing a parameter file's header, but a description list would match L1 more closely. |
