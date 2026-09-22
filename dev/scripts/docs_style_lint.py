#!/usr/bin/env python3
"""Grep-based checker for the mechanical Google-style rules in the documentation.

Plan: dev/PLAN_DOCS_GOOGLE_STYLE_2026-09-21.md, phase 0. The script is the measure
of progress and the regression check for that plan. It imports nothing from
spectroxide and uses only the standard library.

It extracts prose from each reviewed file (Markdown and reStructuredText with code
removed, Markdown cells of the tutorial notebooks, Python docstrings and comments,
Rust `///` and `//!` doc comments) and counts hits per rule per file.

Usage:
    python dev/scripts/docs_style_lint.py                 # count table
    python dev/scripts/docs_style_lint.py --list RULE     # file:line: text per hit
    python dev/scripts/docs_style_lint.py --json OUT.json # machine-readable counts
    python dev/scripts/docs_style_lint.py PATH [PATH...]  # restrict to these files
    python dev/scripts/docs_style_lint.py --check         # exit 1 on any hit

The rules are heuristics. A hit is a lead, not a verdict: "via", "above", and the
abbreviation check all have legitimate exceptions, which the status file records.
"""

import argparse
import ast
import io
import json
import re
import sys
import tokenize
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Same file set as dev/audit/DOCS_GOOGLE_STYLE_REVIEW_2026-09-20.md, "Scope and method".
FILE_GLOBS = [
    "README.md",
    "data/cosmotherm/README.md",
    "CONTRIBUTING.md",
    "CONTRIBUTING_CLAUDE.md",
    "docs/*.rst",
    "docs/tutorials/*.rst",
    "docs/api/*.rst",
    "notebooks/tutorials/*.ipynb",
    "python/spectroxide/*.py",
    "src/*.rs",
    "src/bin/check_adiabatic.rs",
    "examples/*.rs",
]

# Abbreviation -> regex for its expansion. The first use in a file passes if the
# expansion occurs within EXPANSION_WINDOW characters of it, or if it is a :term: role.
ABBREVIATIONS = {
    "PDE": r"partial[-\s]+differential[-\s]+equation",
    "CMB": r"cosmic\s+microwave\s+background",
    "DC": r"double[-\s]+Compton",
    "BR": r"bremsstrahlung",
    "GF": r"Green'?s[-\s]+function",
    "DM": r"dark[-\s]+matter",
    "NWA": r"narrow[-\s]+width[-\s]+approximation",
    "IC": r"initial[-\s]+condition",
    "CLI": r"command[-\s]+line\s+interface",
    "FIRAS": r"Far[-\s]+Infrared\s+Absolute\s+Spectrophotometer",
    "PIXIE": r"Primordial\s+Inflation\s+Explorer",
    "IMEX": r"implicit[-\s]+explicit",
    "ODE": r"ordinary[-\s]+differential[-\s]+equation",
    "DI": r"distortion\s+intensity|intensity\s+distortion",
    "CL": r"confidence\s+level",
}
EXPANSION_WINDOW = 120

# British forms. The generic -ise pattern catches unlisted verbs; ISE_OK holds the
# words that end in -ise in American English too.
ISE_OK = {
    "raise", "rise", "arise", "noise", "denoise", "precise", "imprecise", "concise",
    "exercise", "promise", "premise", "comprise", "compromise", "expertise", "advise",
    "revise", "supervise", "surprise", "disguise", "devise", "praise", "cruise", "poise",
    "franchise", "demise", "excise", "advertise", "improvise", "enterprise", "treatise",
    "reprise", "guise", "sunrise", "crises", "bruise", "despise", "chastise", "paradise",
    "merchandise", "televise", "apprise", "incise", "anise", "valise", "mayonnaise",
}  # fmt: skip
ISE_RE = re.compile(r"\b([A-Za-z]+?)is(e|ed|es|ing|er|ers|ation|ations)\b")
BRITISH_FIXED_RE = re.compile(
    r"\b(?:(?:analy|cataly|paraly)s(?:e|ed|ing)"
    r"|(?:colour|behaviour|favour|neighbour|honour|flavour|vapour|labour|rigour|vigour|odour)\w*"
    r"|(?:cent|fib|calib|lit|met)re(?:s|d)?"
    r"|(?:modell|labell|travell|cancell|channell|signall|levell|fuell)(?:ed|ing|er|ers)"
    r"|catalogue\w*|artefact\w*|programme\w*|grey\w*|whilst|amongst|towards|afterwards"
    r"|defence|judgement|acknowledgement|fulfil|fulfilment|ageing|sceptic\w*|focuss\w+)\b",
    re.IGNORECASE,
)

# First words that mark an imperative Rust summary (rule A2).
IMPERATIVE_VERBS = set("""
    accumulate add advance allocate append apply assemble assert attach bisect build cache
    compare deduplicate factorize reject
    calculate call check choose clamp clear clip collect combine compute configure
    construct convert copy count create decode decompose define derive detect determine
    disable dispatch drop dump emit enable encode ensure estimate evaluate evolve execute
    expand extract fill find fit flag flatten format gather generate get guard handle
    implement initialize inject insert install integrate interpolate invert iterate join
    linearize load locate log look make map mark match measure merge mirror modify move
    normalize open output override pack pad parse patch perform pick pop precompute
    predict prepare print probe process produce project propagate push query raise read
    rebuild recompute record reduce refine register remove render replace report rescale
    reset resolve return round run sample save scale scan search seed select send
    serialize set shift show skip smooth snap solve sort split start step stop store
    strip subdivide subtract suggest sum swap tabulate tag take test toggle trace track transform
    trim truncate try unpack update use validate verify warn weight wrap write zero
    """.split())

# Capitalized words that are legitimate inside a sentence-case heading.
PROPER = set("""
    Green Green's Compton Kompaneets Python Rust Planck Jupyter Sphinx GitHub Claude Bose
    Einstein Thomson Hubble Saha Peebles Newton Crank Nicolson Chluba CosmoTherm
    Richardson Wien Gaunt Born Sunyaev Zeldovich Zel'dovich Landau Zener NumPy SciPy
    Matplotlib Cargo Clippy Markdown Linux Windows American British Google Code Cyr
    Manoj Fisher Hessian Gaussian Bayesian Monte Carlo Docker Conda Miniforge Black
    Anthropic Optional Fokker Rayleigh Jeans Lyman Balmer Doppler Thomas Simpson
    Lagrange Hermite Chebyshev Euler
    """.split())

SIMPLE_RULES = {
    "eg": re.compile(r"\be\.g\.", re.IGNORECASE),
    "ie": re.compile(r"\bi\.e\.", re.IGNORECASE),
    "etc": re.compile(r"\betc\b\.?", re.IGNORECASE),
    "via": re.compile(r"\bvia\b", re.IGNORECASE),
    "and-or": re.compile(r"\band/or\b", re.IGNORECASE),
    "vs": re.compile(r"\bvs\b\.?", re.IGNORECASE),
    "paren-s": re.compile(r"\w\(s\)"),
    # Rule T9: "above" and "below" as a position on the page. A comparison
    # ("below z = 1e4", "above the threshold") is fine, so the word must close a
    # clause or follow a pointer word. Inline code is blanked to spaces, so
    # "above `1e7`." must not match: no whitespace is allowed before the punctuation.
    "position-word": re.compile(
        r"\b(?:(?:see|as|shown|listed|described|noted|given|defined|mentioned|discussed)"
        r"\s+(?:above|below)\b|(?:above|below)[.,:;)]"
        r"|the\s+(?:above|below)\b"
        r"|(?:table|figure|list|example|section|equation|formula|snippet|block|code|cell"
        r"|options?|steps?|notes?)\s+(?:above|below)\b)",
        re.IGNORECASE,
    ),
}
RULES = [
    "abbrev-first-use", *SIMPLE_RULES, "british",
    "rs-imperative-summary", "rs-noun-summary", "rs-oneline-no-period",
    "heading-then-code", "nb-code-no-intro", "title-case-heading",
]  # fmt: skip


# --------------------------------------------------------------------------- prose
# Each extractor returns a list of (line_number, text) with code blanked out, so that
# line numbers survive. Notebook line numbers are "cell index * 1000 + line in cell".


# Prefix on heading lines. The first-use rule skips headings, because Google style
# puts the expansion in the first body sentence, not in the heading.
HEADING_MARK = "\x02"
MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
RST_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+])\1{2,}\s*$")


def _blank_inline(text, kind):
    if kind == "rst":
        # Keep :term: roles; check_abbreviations treats them as a valid first use.
        text = re.sub(r":term:`([^`<]*)`", "\x00\\1\x01", text)
        text = re.sub(
            r":(?:math|code|func|class|mod|meth|attr|data|obj|ref|doc):`[^`]*`",
            " ",
            text,
        )
        text = re.sub(r"``[^`]*``", " ", text)
    text = re.sub(r"`[^`\n]*`", " ", text)
    text = re.sub(r"\$[^$\n]*\$", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    return text.replace("\x00", ":term:`").replace("\x01", "`")


def prose_markdown(lines, first=1):
    out, fence = [], False
    for i, line in enumerate(lines, first):
        if re.match(r"\s*(```|~~~)", line):
            fence = not fence
            out.append((i, ""))
        elif fence:
            out.append((i, ""))
        elif MD_HEADING.match(line):
            out.append((i, HEADING_MARK + _blank_inline(line, "md")))
        else:
            out.append((i, _blank_inline(line, "md")))
    return out


RST_LITERAL_DIRECTIVE = re.compile(
    r"\s*\.\. (?:code-block|code|sourcecode|math|literalinclude|ipython|parsed-literal|toctree"
    r"|autofunction|autoclass|automodule|autosummary|currentmodule|module)::"
)


def prose_rst(lines, first=1):
    out, skip_indent, pending = [], None, False
    for i, line in enumerate(lines, first):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if skip_indent is not None:
            if not stripped or indent > skip_indent:
                out.append((i, ""))
                continue
            skip_indent = None
        if pending:
            pending = False
        if RST_LITERAL_DIRECTIVE.match(line):
            skip_indent = indent
            out.append((i, ""))
            continue
        if stripped.endswith("::") and not stripped.startswith(".."):
            skip_indent = indent
            out.append((i, _blank_inline(line[: line.rstrip().rfind("::")], "rst")))
            continue
        if re.match(r"\s*\.\. _[^:]+:", line) or re.match(r"\s*:[a-z-]+:", line):
            out.append((i, ""))
            continue
        if RST_UNDERLINE.match(line) and out and out[-1][1].strip():
            out[-1] = (out[-1][0], HEADING_MARK + out[-1][1])
            out.append((i, ""))
            continue
        out.append((i, _blank_inline(line, "rst")))
    return out


def notebook_cells(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for c in nb["cells"]:
        src = c["source"]
        src = src if isinstance(src, str) else "".join(src)
        cells.append((c["cell_type"], src.split("\n")))
    return cells


def prose_notebook(path):
    out = []
    for k, (kind, lines) in enumerate(notebook_cells(path)):
        if kind == "markdown":
            out.extend(prose_markdown(lines, first=k * 1000 + 1))
    return out


def prose_python(path):
    source = path.read_text(encoding="utf-8")
    out = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
            ):
                if isinstance(body[0].value.value, str):
                    start = body[0].value.lineno
                    doc_lines = body[0].value.value.split("\n")
                    # Docstrings are NumPy-style reST; reuse the reST code stripping.
                    out.extend(prose_rst(doc_lines, first=start))
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            text = tok.string.lstrip("#").strip()
            if not text.startswith(("fmt:", "noqa", "type:", "!", "pragma")):
                out.append((tok.start[0], _blank_inline(text, "md")))
    out.sort(key=lambda t: t[0])
    return out


RS_DOC = re.compile(r"^\s*//[/!] ?(.*)$")


def rust_doc_blocks(lines):
    """Yield (first_line, [doc text lines], item_line) for each `///` or `//!` block."""
    i, n = 0, len(lines)
    while i < n:
        if RS_DOC.match(lines[i]) and not lines[i].lstrip().startswith("////"):
            start, block = i, []
            while i < n and RS_DOC.match(lines[i]):
                block.append(RS_DOC.match(lines[i]).group(1))
                i += 1
            j = i
            while j < n and (
                not lines[j].strip() or lines[j].lstrip().startswith("#[")
            ):
                j += 1
            yield start + 1, block, (lines[j] if j < n else "")
        else:
            i += 1


def prose_rust(path):
    lines = path.read_text(encoding="utf-8").split("\n")
    out = []
    for start, block, _ in rust_doc_blocks(lines):
        out.extend(prose_markdown(block, first=start))
    return out


def extract(path):
    suffix = path.suffix
    if suffix == ".md":
        return prose_markdown(path.read_text(encoding="utf-8").split("\n"))
    if suffix == ".rst":
        return prose_rst(path.read_text(encoding="utf-8").split("\n"))
    if suffix == ".ipynb":
        return prose_notebook(path)
    if suffix == ".py":
        return prose_python(path)
    if suffix == ".rs":
        return prose_rust(path)
    raise ValueError(f"unsupported file type: {path}")


# --------------------------------------------------------------------------- rules


def check_abbreviations(prose):
    text, starts = "", []
    for lineno, line in prose:
        starts.append((len(text), lineno))
        text += ("" if line.startswith(HEADING_MARK) else line) + "\n"

    def line_of(pos):
        lo = 0
        for offset, lineno in starts:
            if offset > pos:
                break
            lo = lineno
        return lo

    hits = []
    for abbr, expansion in ABBREVIATIONS.items():
        m = re.search(rf"(?<![A-Za-z0-9_/-]){abbr}s?(?![A-Za-z0-9_])", text)
        if not m:
            continue
        window = text[max(0, m.start() - EXPANSION_WINDOW) : m.end() + EXPANSION_WINDOW]
        if re.search(expansion, window, re.IGNORECASE):
            continue
        if text[max(0, m.start() - 7) : m.start()].endswith(":term:`"):
            continue
        hits.append((line_of(m.start()), f"{abbr} used before it is spelled out"))
    return hits


def check_british(prose):
    hits = []
    for lineno, line in prose:
        for m in ISE_RE.finditer(line):
            word = m.group(0).lower()
            stem = m.group(1).lower() + "ise"
            if stem in ISE_OK or stem.endswith("wise") or word in ISE_OK:
                continue
            hits.append((lineno, m.group(0)))
        for m in BRITISH_FIXED_RE.finditer(line):
            hits.append((lineno, m.group(0)))
    return hits


def _third_person(word):
    """True if `word` is the third-person singular of a verb in IMPERATIVE_VERBS."""
    if not word.endswith("s"):
        return False
    stems = {word[:-1], word[:-2] if word.endswith("es") else ""}
    if word.endswith("ies"):
        stems.add(word[:-3] + "y")
    return bool(stems & IMPERATIVE_VERBS)


def check_rust_summaries(path):
    lines = path.read_text(encoding="utf-8").split("\n")
    imperative, noun, no_period = [], [], []
    for start, block, item in rust_doc_blocks(lines):
        if not block or lines[start - 1].lstrip().startswith("//!"):
            continue
        summary = block[0].strip()
        if re.search(r"\bfn\b", item):
            first = re.match(r"[A-Za-z]+", summary)
            if first and first.group(0).lower() in IMPERATIVE_VERBS:
                imperative.append((start, summary))
            elif not (first and _third_person(first.group(0).lower())):
                # Rule A2, second half: a function summary opens with a verb
                # ("Returns the ..."), not with a noun phrase.
                noun.append((start, summary))
        if (
            len(block) == 1
            and summary
            and not summary.startswith(("#", "```", "-", "|"))
        ):
            if summary[-1] not in ".:?!":
                no_period.append((start, summary))
    return imperative, noun, no_period


def headings(path):
    """Return [(line_number, heading text, kind of the next non-blank line)]."""
    result = []
    if path.suffix == ".ipynb":
        for k, (kind, lines) in enumerate(notebook_cells(path)):
            if kind == "markdown":
                result.extend(_md_headings(lines, first=k * 1000 + 1))
        return result
    lines = path.read_text(encoding="utf-8").split("\n")
    if path.suffix == ".md":
        return _md_headings(lines)
    for i in range(1, len(lines)):
        if (
            RST_UNDERLINE.match(lines[i])
            and lines[i - 1].strip()
            and not RST_UNDERLINE.match(lines[i - 1])
        ):
            if len(lines[i].rstrip()) >= len(lines[i - 1].rstrip()):
                nxt = next((l for l in lines[i + 1 :] if l.strip()), "")
                is_code = bool(
                    re.match(
                        r"\s*\.\. (code-block|code|sourcecode|list-table|csv-table)::",
                        nxt,
                    )
                )
                is_code = (
                    is_code
                    or nxt.strip() == "::"
                    or bool(re.match(r"^[=+][=+-]{3,}", nxt))
                )
                result.append((i, lines[i - 1].strip(), "code" if is_code else "text"))
    return result


def _md_headings(lines, first=1):
    result, fence = [], False
    for i, line in enumerate(lines):
        if re.match(r"\s*(```|~~~)", line):
            fence = not fence
            continue
        m = None if fence else MD_HEADING.match(line)
        if m:
            nxt = next((l for l in lines[i + 1 :] if l.strip()), "")
            is_code = bool(re.match(r"\s*(```|~~~)", nxt)) or nxt.lstrip().startswith(
                "|"
            )
            result.append((first + i, m.group(2), "code" if is_code else "text"))
    return result


def check_headings(path):
    then_code, title_case = [], []
    for lineno, text, nxt in headings(path):
        if nxt == "code":
            then_code.append((lineno, text))
        plain = re.sub(r"`[^`]*`|\$[^$]*\$", " ", text)
        words = re.findall(r"[A-Za-z][A-Za-z'’-]*", plain)
        after_colon = {
            w
            for part in re.split(r"[:.?!]\s+", plain)[1:]
            for w in re.findall(r"[A-Za-z'’-]+", part)[:1]
        }
        bad = [
            w
            for w in words[1:]
            if re.fullmatch(r"[A-Z][a-z'’-]+", w)
            and w not in PROPER
            and w not in after_colon
        ]
        if bad:
            title_case.append((lineno, text))
    return then_code, title_case


def check_notebook_intros(path):
    hits, prev = [], None
    for k, (kind, lines) in enumerate(notebook_cells(path)):
        body = [l for l in lines if l.strip()]
        if kind == "code" and body:
            if (
                prev is None
                or prev[0] == "code"
                or not prev[1]
                or MD_HEADING.match(prev[1][-1])
            ):
                hits.append((k * 1000 + 1, body[0][:70]))
        if body or kind == "markdown":
            prev = (kind, body)
    return hits


def lint(path):
    hits = defaultdict(list)
    prose = extract(path)
    hits["abbrev-first-use"] = check_abbreviations(prose)
    for rule, pattern in SIMPLE_RULES.items():
        for lineno, line in prose:
            for m in pattern.finditer(line):
                hits[rule].append((lineno, line.strip()[:100]))
    hits["british"] = check_british(prose)
    if path.suffix == ".rs":
        (
            hits["rs-imperative-summary"],
            hits["rs-noun-summary"],
            hits["rs-oneline-no-period"],
        ) = check_rust_summaries(path)
    if path.suffix in (".md", ".rst", ".ipynb"):
        hits["heading-then-code"], hits["title-case-heading"] = check_headings(path)
    if path.suffix == ".ipynb":
        hits["nb-code-no-intro"] = check_notebook_intros(path)
    return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "paths", nargs="*", help="files to check (default: the reviewed file set)"
    )
    parser.add_argument(
        "--list", metavar="RULE", help="print every hit for RULE ('all' for every rule)"
    )
    parser.add_argument("--json", metavar="OUT", help="write per-file counts to OUT")
    parser.add_argument(
        "--check", action="store_true", help="exit with status 1 if any rule has a hit"
    )
    args = parser.parse_args()

    if args.paths:
        files = [Path(p).resolve() for p in args.paths]
    else:
        # The glossary defines the abbreviations, so the first-use rule does not apply to it.
        files = sorted(
            {f for g in FILE_GLOBS for f in ROOT.glob(g)} - {ROOT / "docs/glossary.rst"}
        )

    counts, totals = {}, defaultdict(int)
    for f in files:
        name = str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else str(f)
        hits = lint(f)
        counts[name] = {r: len(hits[r]) for r in RULES if hits[r]}
        for r in RULES:
            totals[r] += len(hits[r])
            if args.list in (r, "all"):
                for lineno, text in sorted(hits[r]):
                    where = (
                        f"cell {lineno // 1000}, line {lineno % 1000}"
                        if f.suffix == ".ipynb"
                        else lineno
                    )
                    print(f"{name}:{where}: [{r}] {text}")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"totals": totals, "files": counts}, indent=1) + "\n"
        )
    if not args.list:
        width = max(len(n) for n in counts)
        short = [r[:9] for r in RULES]
        print(f"{'file':<{width}} " + " ".join(f"{s:>9}" for s in short))
        for name, c in counts.items():
            if c:
                print(
                    f"{name:<{width}} "
                    + " ".join(f"{c.get(r, 0) or '':>9}" for r in RULES)
                )
        print(f"{'TOTAL':<{width}} " + " ".join(f"{totals[r]:>9}" for r in RULES))
    return 1 if args.check and any(totals.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
