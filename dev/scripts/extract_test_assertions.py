"""Extract every numeric assertion from the test suites (audit B0).

Walks tests/*.rs, #[cfg(test)] modules in src/*.rs, and python/tests/*.py,
and emits a JSON inventory of (file, test, line, assertion) records for every
assertion that contains a numeric literal. This is the raw input for the
test-provenance census (dev/audit/TEST_PROVENANCE.md): every record must be
classified by provenance {analytic, literature, dimensional, cross-method,
regression-pin, structural}.

Usage: python dev/scripts/extract_test_assertions.py [-o output.json]
"""

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

RUST_TEST_FN = re.compile(r"^\s*(?:async\s+)?fn\s+(\w+)\s*\(")
RUST_TEST_ATTR = re.compile(r"^\s*#\[(?:test|ignore|should_panic)")
RUST_ASSERT = re.compile(
    r"^\s*(assert!|assert_eq!|assert_ne!|assert_relative_eq!|assert_abs_diff_eq!|"
    r"debug_assert!|panic!)"
)
PY_TEST_FN = re.compile(r"^\s*def\s+(test_\w+)\s*\(")
PY_ASSERT = re.compile(r"^\s*(assert\b|pytest\.approx|np\.testing\.|assert_allclose)")

# A numeric literal that looks like a physics target (not a bare index/0/1
# count); floats, scientific notation, or integers >= 2 digits.
NUMERIC = re.compile(r"(?<![\w.])(\d+\.\d*(e[+-]?\d+)?|\d+e[+-]?\d+|\.\d+|\d{2,})", re.I)


def _collect_statement(lines, i):
    """Join a multi-line assertion statement (until bracket balance)."""
    stmt = lines[i]
    depth = stmt.count("(") - stmt.count(")")
    j = i
    while depth > 0 and j + 1 < len(lines):
        j += 1
        stmt += " " + lines[j].strip()
        depth += lines[j].count("(") - lines[j].count(")")
    return stmt.strip(), j


def extract_rust(path):
    lines = path.read_text().splitlines()
    records = []
    current_test = None
    in_test_attr = False
    for i, line in enumerate(lines):
        if RUST_TEST_ATTR.match(line):
            in_test_attr = True
            continue
        m = RUST_TEST_FN.match(line)
        if m:
            current_test = m.group(1) if in_test_attr or "tests/" in str(path) else m.group(1)
            in_test_attr = False
            continue
        if RUST_ASSERT.match(line) and current_test:
            stmt, _ = _collect_statement(lines, i)
            if NUMERIC.search(stmt):
                records.append(
                    {
                        "file": str(path.relative_to(REPO)),
                        "test": current_test,
                        "line": i + 1,
                        "assertion": stmt[:400],
                    }
                )
    return records


def extract_python(path):
    lines = path.read_text().splitlines()
    records = []
    current_test = None
    for i, line in enumerate(lines):
        m = PY_TEST_FN.match(line)
        if m:
            current_test = m.group(1)
            continue
        if PY_ASSERT.match(line) and current_test:
            stmt, _ = _collect_statement(lines, i)
            if NUMERIC.search(stmt):
                records.append(
                    {
                        "file": str(path.relative_to(REPO)),
                        "test": current_test,
                        "line": i + 1,
                        "assertion": stmt[:400],
                    }
                )
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default=str(REPO / "dev/audit/test_assertions.json"))
    args = ap.parse_args()

    records = []
    for p in sorted((REPO / "tests").glob("*.rs")):
        records.extend(extract_rust(p))
    for p in sorted((REPO / "src").rglob("*.rs")):
        if "#[cfg(test)]" in p.read_text():
            records.extend(extract_rust(p))
    for p in sorted((REPO / "python/tests").glob("test_*.py")):
        records.extend(extract_python(p))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    by_file = {}
    for r in records:
        by_file.setdefault(r["file"], 0)
        by_file[r["file"]] += 1
    with open(out, "w") as f:
        json.dump(
            {
                "n_assertions": len(records),
                "n_tests": len({(r["file"], r["test"]) for r in records}),
                "by_file": by_file,
                "records": records,
            },
            f,
            indent=1,
        )
    print(f"{len(records)} numeric assertions in "
          f"{len({(r['file'], r['test']) for r in records})} tests -> {out}")
    for fname, n in sorted(by_file.items()):
        print(f"  {n:5d}  {fname}")


if __name__ == "__main__":
    main()
