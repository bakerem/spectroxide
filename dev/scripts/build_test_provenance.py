"""Assemble dev/audit/TEST_PROVENANCE.md from the census fragments (audit B0).

Inputs: the classification fragments produced by the census agents (JSON lists
of {file, test, provenance, target, source, flag, note}) plus the raw
inventory (dev/audit/test_assertions.json) for cross-checking coverage.

Usage: python dev/scripts/build_test_provenance.py <fragment.json> [...]
"""

import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CLASSES = [
    "analytic",
    "literature",
    "cross-method",
    "dimensional",
    "regression-pin",
    "structural",
]


def main():
    records = []
    for frag in sys.argv[1:]:
        records.extend(json.load(open(frag)))

    inv = json.load(open(REPO / "dev/audit/test_assertions.json"))
    inv_tests = {(r["file"], r["test"]) for r in inv["records"]}
    got_tests = {(r["file"], r["test"]) for r in records}
    missing = sorted(inv_tests - got_tests)

    counts = Counter(r["provenance"] for r in records)
    flagged = [r for r in records if r.get("flag")]
    pins = [r for r in records if r["provenance"] == "regression-pin"]

    out = REPO / "dev/audit/TEST_PROVENANCE.md"
    with open(out, "w") as f:
        f.write("# Test-Provenance Census (audit B0)\n\n")
        f.write(f"Generated {date.today().isoformat()} by "
                "`dev/scripts/build_test_provenance.py` from the assertion "
                "inventory (`dev/scripts/extract_test_assertions.py`, "
                f"{inv['n_assertions']} numeric assertions in "
                f"{inv['n_tests']} tests).\n\n")
        f.write(
            "Every test with a numeric assertion is classified by the origin "
            "of its target value:\n\n"
            "- **analytic** — closed-form result verifiable independently of the code\n"
            "- **literature** — number from a cited paper or public dataset\n"
            "- **cross-method** — agreement between two independent implementations\n"
            "- **dimensional** — sign/scaling/order-of-magnitude from a physical argument\n"
            "- **regression-pin** — the code's own historical output, explicitly labelled\n"
            "- **structural** — numeric literal is incidental (sizes, finiteness); no physics target\n\n"
        )
        f.write("## Summary\n\n| Class | Tests |\n|---|---|\n")
        for c in CLASSES:
            f.write(f"| {c} | {counts.get(c, 0)} |\n")
        f.write(f"| **total** | **{len(records)}** |\n\n")
        f.write(f"Flagged audit gaps (code-derived target without a pin label): "
                f"**{len(flagged)}**\n\n")
        if missing:
            f.write(f"Inventory tests not yet classified: {len(missing)}\n\n")
            for m in missing[:20]:
                f.write(f"- {m[0]}::{m[1]}\n")
            f.write("\n")

        if flagged:
            f.write("## Flagged gaps\n\n")
            for r in flagged:
                f.write(f"- `{r['file']}::{r['test']}` — {r['target']} "
                        f"({r.get('note', '')})\n")
            f.write("\n")

        if pins:
            f.write("## Declared regression pins\n\n")
            for r in pins:
                f.write(f"- `{r['file']}::{r['test']}` — {r['target']}"
                        f"{' — ' + r['note'] if r.get('note') else ''}\n")
            f.write("\n")

        f.write("## Full table\n\n")
        f.write("| File | Test | Class | Target | Source |\n|---|---|---|---|---|\n")
        for r in sorted(records, key=lambda r: (r["file"], r["test"])):
            tgt = r.get("target", "").replace("|", "\\|")
            src = r.get("source", "").replace("|", "\\|")
            flag = " ⚠" if r.get("flag") else ""
            f.write(f"| {r['file']} | `{r['test']}`{flag} | {r['provenance']} "
                    f"| {tgt} | {src} |\n")

    print(f"{len(records)} classified, {len(missing)} missing, "
          f"{len(flagged)} flagged -> {out}")
    for c in CLASSES:
        print(f"  {c:15s} {counts.get(c, 0)}")


if __name__ == "__main__":
    main()
