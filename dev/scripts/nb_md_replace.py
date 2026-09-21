#!/usr/bin/env python3
"""Replace text in one Markdown cell of a notebook without touching anything else.

Plan: dev/PLAN_DOCS_GOOGLE_STYLE_2026-09-21.md, hazard H-4. The tutorial notebooks
carry executed outputs, and at least one mixes escaped and raw non-ASCII characters,
so a JSON load-and-dump rewrites output lines. This script edits the raw file text
instead, then proves that only the target cell's source changed.

Usage:
    python dev/scripts/nb_md_replace.py --show NOTEBOOK           # print Markdown cells
    python dev/scripts/nb_md_replace.py NOTEBOOK CELL OLD NEW     # replace OLD with NEW
    python dev/scripts/nb_md_replace.py --insert NOTEBOOK CELL TEXT  # new Markdown cell before CELL

CELL is the zero-based cell index. OLD must lie within one source line of that cell
and must occur exactly once in the raw file (add context until it does). NEW may
contain newlines; each becomes a new source line.
"""

import json
import sys
import uuid
from pathlib import Path


def cell_source(cell):
    src = cell["source"]
    return src if isinstance(src, str) else "".join(src)


def show(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    for k, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown":
            print(f"--- cell {k} (markdown)")
            print(cell_source(cell))
        else:
            first = cell_source(cell).strip().split("\n")[0][:80]
            print(f"--- cell {k} ({cell['cell_type']}): {first}")


def encode(text, ascii_only):
    return json.dumps(text, ensure_ascii=ascii_only)[1:-1]


def replace(path, index, old, new):
    raw = path.read_text(encoding="utf-8")
    before = json.loads(raw)
    cell = before["cells"][index]
    if cell["cell_type"] != "markdown":
        sys.exit(f"cell {index} is a {cell['cell_type']} cell, not Markdown")
    if "\n" in old:
        sys.exit("OLD must lie within one source line")
    if cell_source(cell).count(old) != 1:
        sys.exit(
            f"OLD occurs {cell_source(cell).count(old)} times in cell {index}; need exactly 1"
        )

    for ascii_only in (False, True):
        needle = encode(old, ascii_only)
        if raw.count(needle) == 1:
            break
    else:
        sys.exit("encoded OLD does not occur exactly once in the raw file; add context")

    pos = raw.index(needle)
    line_start = raw.rfind("\n", 0, pos) + 1
    indent = raw[line_start:pos]
    indent = indent[: len(indent) - len(indent.lstrip())]
    joiner = '\\n",\n' + indent + '"'
    replacement = joiner.join(encode(part, ascii_only) for part in new.split("\n"))
    raw_new = raw[:pos] + replacement + raw[pos + len(needle) :]

    after = json.loads(raw_new)
    expected = cell_source(cell).replace(old, new)
    if cell_source(after["cells"][index]) != expected:
        sys.exit("internal error: edited cell does not match the expected text")
    for k, (a, b) in enumerate(zip(before["cells"], after["cells"])):
        if k != index and a != b:
            sys.exit(f"internal error: cell {k} changed")
    stripped = lambda nb: {key: val for key, val in nb.items() if key != "cells"}
    if len(before["cells"]) != len(after["cells"]) or stripped(before) != stripped(
        after
    ):
        sys.exit("internal error: notebook structure changed")
    path.write_text(raw_new, encoding="utf-8")
    print(f"{path.name}: cell {index} updated")


def insert(path, index, text):
    """Insert a new Markdown cell holding TEXT before cell INDEX."""
    raw = path.read_text(encoding="utf-8")
    before = json.loads(raw)
    decoder = json.JSONDecoder()
    pos = raw.index("[", raw.index('"cells"')) + 1
    for _ in range(index):
        _, pos = decoder.raw_decode(raw, pos + len(raw[pos:]) - len(raw[pos:].lstrip()))
        pos = raw.index(",", pos) + 1
    pos += len(raw[pos:]) - len(raw[pos:].lstrip())
    line_start = raw.rfind("\n", 0, pos) + 1
    indent = raw[line_start:pos]

    lines = text.split("\n")
    cell = {"cell_type": "markdown"}
    if any("id" in c for c in before["cells"]):
        cell["id"] = uuid.uuid4().hex[:8]
    cell["metadata"] = {}
    cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    ascii_only = "\\u" in raw and not any(ord(ch) > 127 for ch in raw)
    body = json.dumps(cell, indent=1, ensure_ascii=ascii_only)
    body = body.replace("\n", "\n" + indent)
    raw_new = raw[:pos] + body + ",\n" + indent + raw[pos:]

    after = json.loads(raw_new)
    if after["cells"][:index] + after["cells"][index + 1 :] != before["cells"]:
        sys.exit("internal error: an existing cell changed")
    if after["cells"][index]["cell_type"] != "markdown":
        sys.exit("internal error: inserted cell is not Markdown")
    if {k: v for k, v in after.items() if k != "cells"} != {
        k: v for k, v in before.items() if k != "cells"
    }:
        sys.exit("internal error: notebook structure changed")
    path.write_text(raw_new, encoding="utf-8")
    print(f"{path.name}: Markdown cell inserted at index {index}")


def main():
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == "--show":
        show(Path(args[1]))
    elif len(args) == 4 and args[0] == "--insert":
        insert(Path(args[1]), int(args[2]), args[3])
    elif len(args) == 4:
        replace(Path(args[0]), int(args[1]), args[2], args[3])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
