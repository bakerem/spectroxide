#!/usr/bin/env python3
"""
Verify refs.bib DOIs against the CrossRef API.

For each BibTeX entry:
  - If it has a DOI, fetch metadata from https://api.crossref.org/works/{doi}
    and compare title, authors, year, journal/container, volume, pages.
  - If no DOI, flag as "no DOI - manual check needed".

Also detects fabricated DOIs (404) and major metadata mismatches that indicate
the DOI points to the wrong paper.
"""

import re
import sys
import time
import urllib.parse
from pathlib import Path
from difflib import SequenceMatcher

import requests

# ── BibTeX parser (minimal, sufficient for this file) ──────────────────────

def parse_bib(path: str) -> list[dict]:
    """Parse a .bib file into a list of dicts with keys lowercased."""
    text = Path(path).read_text()
    entries = []
    # Match entry type and cite key, then the brace-delimited body
    pattern = re.compile(
        r'@(\w+)\s*\{([^,]+),\s*(.*?)\n\}', re.DOTALL
    )
    for m in pattern.finditer(text):
        entry_type = m.group(1).lower()
        cite_key = m.group(2).strip()
        body = m.group(3)
        fields = {"_type": entry_type, "_key": cite_key}
        # Parse fields by finding 'key = ' then extracting the balanced-brace value
        for fm in re.finditer(r'(\w+)\s*=\s*', body):
            key = fm.group(1).lower()
            rest = body[fm.end():]
            val = ""
            if rest.startswith('{'):
                # Extract balanced braces (handles arbitrary nesting)
                depth = 0
                for i, ch in enumerate(rest):
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            val = rest[1:i]  # strip outer braces
                            break
            elif rest.startswith('"'):
                m2 = re.match(r'"([^"]*)"', rest)
                if m2:
                    val = m2.group(1)
            else:
                m2 = re.match(r'(\d+)', rest)
                if m2:
                    val = m2.group(1)
            val = " ".join(val.split())
            fields[key] = val
        entries.append(fields)
    return entries


# ── Comparison helpers ─────────────────────────────────────────────────────

def normalize(s: str) -> str:
    """Lowercase, strip braces/backslashes/accents/HTML/MathML, collapse whitespace."""
    s = s.lower()
    # Strip HTML/XML tags (CrossRef often returns <i>, <mml:math> etc.)
    s = re.sub(r'<[^>]+>', ' ', s)
    s = re.sub(r'[{}\\\'"~`^]', '', s)
    s = re.sub(r'\$[^$]*\$', '', s)  # strip TeX math
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def title_similar(a: str, b: str, threshold: float = 0.60) -> bool:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio() >= threshold


def strip_accents(s: str) -> str:
    """Remove unicode accents for fuzzy comparison."""
    import unicodedata
    nfkd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def authors_match(bib_author: str, cr_authors: list[dict]) -> tuple[bool, str]:
    """Check whether at least the first author surname appears in CrossRef data."""
    if not cr_authors:
        return False, "no authors in CrossRef"
    # Extract bib first-author surname (before first comma or 'and')
    bib_norm = strip_accents(normalize(bib_author))
    bib_first = bib_norm.split(" and ")[0].split(",")[0].strip()
    # CrossRef author surnames (also strip accents)
    cr_surnames = [strip_accents(normalize(a.get("family", ""))) for a in cr_authors]
    if any(bib_first and bib_first in s for s in cr_surnames):
        return True, ""
    # Also try matching against given names for consortium entries (empty family)
    cr_given = [strip_accents(normalize(a.get("given", ""))) for a in cr_authors]
    if any(bib_first and bib_first in g for g in cr_given):
        return True, ""
    return False, f"bib first-author '{bib_first}' not in CR {cr_surnames[:5]}"


def normalize_pages(p: str) -> str:
    """Normalize dashes and leading zeros in page strings."""
    p = re.sub(r'\s', '', p)
    p = p.replace('--', '-').replace('–', '-').replace('—', '-')
    # Remove leading zeros in each part
    parts = p.split('-')
    parts = [str(int(x)) if x.isdigit() else x for x in parts]
    return '-'.join(parts)


# ── CrossRef query ─────────────────────────────────────────────────────────

CROSSREF_BASE = "https://api.crossref.org/works/"
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "spectroxide-ref-checker/1.0 (mailto:bakerem@example.com)"
})


def fetch_crossref(doi: str) -> dict | None:
    """Fetch CrossRef metadata for a DOI. Returns None on 404/error."""
    # Ensure DOI is properly encoded but preserve slashes
    encoded = urllib.parse.quote(doi, safe="/():-.")
    url = CROSSREF_BASE + encoded
    try:
        r = SESSION.get(url, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json().get("message", {})
    except requests.RequestException as e:
        return {"_error": str(e)}


# ── Main checking logic ───────────────────────────────────────────────────

def check_entry(entry: dict) -> dict:
    """Check one bib entry against CrossRef. Returns a result dict."""
    key = entry["_key"]
    doi = entry.get("doi", "").strip()
    result = {
        "key": key,
        "type": entry["_type"],
        "doi": doi,
        "checks": {},
        "status": "PASS",
        "notes": [],
    }

    if not doi:
        result["status"] = "NO DOI"
        result["notes"].append("no DOI - manual check needed")
        return result

    cr = fetch_crossref(doi)

    if cr is None:
        result["status"] = "FAIL"
        result["notes"].append(f"DOI {doi} returned 404 - POSSIBLY FABRICATED")
        return result

    if "_error" in cr:
        result["status"] = "ERROR"
        result["notes"].append(f"CrossRef error: {cr['_error']}")
        return result

    # ── Title ──
    cr_titles = cr.get("title", [])
    cr_title = cr_titles[0] if cr_titles else ""
    bib_title = entry.get("title", "")
    t_match = title_similar(bib_title, cr_title)
    result["checks"]["title"] = {
        "match": t_match,
        "bib": bib_title[:80],
        "crossref": cr_title[:80],
    }
    if not t_match:
        result["status"] = "FAIL"
        result["notes"].append("TITLE MISMATCH - possible wrong DOI")

    # ── Authors ──
    cr_authors = cr.get("author", [])
    bib_author = entry.get("author", "")
    a_match, a_note = authors_match(bib_author, cr_authors)
    result["checks"]["authors"] = {
        "match": a_match,
        "note": a_note,
    }
    if not a_match and cr_authors:
        result["status"] = "FAIL" if result["status"] != "FAIL" else "FAIL"
        result["notes"].append(f"AUTHOR MISMATCH: {a_note}")

    # ── Year ──
    cr_year = ""
    if "published-print" in cr:
        cr_year = str(cr["published-print"].get("date-parts", [[""]])[0][0])
    elif "published-online" in cr:
        cr_year = str(cr["published-online"].get("date-parts", [[""]])[0][0])
    elif "published" in cr:
        cr_year = str(cr["published"].get("date-parts", [[""]])[0][0])
    elif "issued" in cr:
        cr_year = str(cr["issued"].get("date-parts", [[""]])[0][0])
    bib_year = entry.get("year", "")
    y_match = bib_year == cr_year
    result["checks"]["year"] = {
        "match": y_match,
        "bib": bib_year,
        "crossref": cr_year,
    }
    if not y_match:
        # Off by one year is common (published vs accepted)
        try:
            diff = abs(int(bib_year) - int(cr_year))
            if diff <= 1:
                result["notes"].append(f"YEAR OFF BY 1: bib={bib_year} vs CR={cr_year}")
            else:
                result["status"] = "FAIL"
                result["notes"].append(f"YEAR MISMATCH: bib={bib_year} vs CR={cr_year}")
        except ValueError:
            result["notes"].append(f"YEAR MISMATCH: bib={bib_year} vs CR={cr_year}")

    # ── Journal / container ──
    cr_journal = ""
    for jkey in ["container-title", "short-container-title"]:
        jvals = cr.get(jkey, [])
        if jvals:
            cr_journal = jvals[0]
            break
    bib_journal = entry.get("journal", entry.get("booktitle", ""))
    if bib_journal and cr_journal:
        j_match = title_similar(bib_journal, cr_journal, threshold=0.45)
        result["checks"]["journal"] = {
            "match": j_match,
            "bib": bib_journal,
            "crossref": cr_journal,
        }
        if not j_match:
            result["notes"].append(f"JOURNAL MISMATCH: bib='{bib_journal}' vs CR='{cr_journal}'")

    # ── Volume ──
    cr_volume = cr.get("volume", "")
    bib_volume = entry.get("volume", "")
    if bib_volume and cr_volume:
        v_match = bib_volume == cr_volume
        result["checks"]["volume"] = {"match": v_match, "bib": bib_volume, "crossref": cr_volume}
        if not v_match:
            result["notes"].append(f"VOLUME MISMATCH: bib={bib_volume} vs CR={cr_volume}")

    # ── Pages ──
    cr_pages = cr.get("page", cr.get("article-number", ""))
    bib_pages = entry.get("pages", "")
    if bib_pages and cr_pages:
        p_match = normalize_pages(bib_pages) == normalize_pages(cr_pages)
        result["checks"]["pages"] = {"match": p_match, "bib": bib_pages, "crossref": cr_pages}
        if not p_match:
            result["notes"].append(f"PAGES MISMATCH: bib='{bib_pages}' vs CR='{cr_pages}'")

    # If all checks passed and status is still PASS, keep it
    if result["status"] == "PASS" and not any(
        "MISMATCH" in n or "FABRICATED" in n for n in result["notes"]
    ):
        result["status"] = "PASS"

    return result


def main():
    bib_path = Path(__file__).resolve().parent.parent.parent / "paper" / "refs.bib"
    if not bib_path.exists():
        print(f"ERROR: {bib_path} not found")
        sys.exit(1)

    entries = parse_bib(str(bib_path))
    print(f"Parsed {len(entries)} BibTeX entries from {bib_path.name}\n")
    print("=" * 90)

    results = []
    for i, entry in enumerate(entries):
        key = entry["_key"]
        doi = entry.get("doi", "")
        print(f"[{i+1}/{len(entries)}] {key}", end="")
        if doi:
            print(f"  doi:{doi}", end="")
        print(" ... ", end="", flush=True)

        result = check_entry(entry)
        results.append(result)

        status = result["status"]
        if status == "PASS":
            print("PASS")
        elif status == "NO DOI":
            print("NO DOI")
        else:
            print(status)
            for note in result["notes"]:
                print(f"    -> {note}")

        # Be polite to CrossRef API
        if doi:
            time.sleep(0.3)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] == "FAIL"]
    no_doi = [r for r in results if r["status"] == "NO DOI"]
    errors = [r for r in results if r["status"] == "ERROR"]
    warnings = [r for r in results if r["status"] == "PASS" and r["notes"]]

    print(f"\nTotal entries:  {len(results)}")
    print(f"  PASS:         {len(passed)}")
    print(f"  FAIL:         {len(failed)}")
    print(f"  NO DOI:       {len(no_doi)}")
    print(f"  ERROR:        {len(errors)}")

    if failed:
        print(f"\n{'─'*90}")
        print("FAILURES (need attention):")
        print(f"{'─'*90}")
        for r in failed:
            print(f"\n  [{r['key']}]  doi:{r['doi']}")
            for note in r["notes"]:
                print(f"    * {note}")
            for field, info in r["checks"].items():
                if not info.get("match", True):
                    bib_val = info.get("bib", "")
                    cr_val = info.get("crossref", "")
                    if bib_val or cr_val:
                        print(f"      {field}: bib='{bib_val}'")
                        print(f"      {field}: cr ='{cr_val}'")

    # Print warnings (year off by 1, minor issues)
    warns = [r for r in results if r["status"] == "PASS" and r["notes"]]
    if warns:
        print(f"\n{'─'*90}")
        print("WARNINGS (minor issues, likely OK):")
        print(f"{'─'*90}")
        for r in warns:
            print(f"\n  [{r['key']}]")
            for note in r["notes"]:
                print(f"    * {note}")

    if no_doi:
        print(f"\n{'─'*90}")
        print("NO DOI (manual check needed):")
        print(f"{'─'*90}")
        for r in no_doi:
            print(f"  [{r['key']}] {r['type']}")

    if errors:
        print(f"\n{'─'*90}")
        print("ERRORS (API issues):")
        print(f"{'─'*90}")
        for r in errors:
            print(f"  [{r['key']}] {r['notes']}")

    print()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
