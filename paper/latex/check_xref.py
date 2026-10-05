#!/usr/bin/env python3
"""Build check for the cross-references between the two submission PDFs.

paper.tex prints fixed fallback numbers for \\appref (main paper -> appendix) and \\mainref
(appendix -> main paper) when the other PDF's .aux file is absent, as on Overleaf.  This script
compares every fallback with the label numbers that the local build wrote into the .aux files and
fails (exit status 1) if a fallback disagrees, if a referenced label has no fallback or no
definition, or if a LaTeX log reports an undefined reference.  Run it after `make` has built
submission.pdf, submission_appendix.pdf, and arxiv.pdf (the Makefile does this).
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = [HERE / "paper.tex", HERE / "appendix.tex"]


def strip_comments(text):
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())


def first_group(text, start):
    """Return the contents of the brace group that opens at text[start]."""
    if text[start] != "{":
        raise ValueError(f"expected '{{' at offset {start}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
    raise ValueError("unbalanced braces")


def aux_labels(path):
    """Map label -> printed number for every \\newlabel in one .aux file."""
    labels = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for match in re.finditer(r"\\newlabel\{([^}]*)\}", text):
        data = first_group(text, match.end())
        labels[match.group(1)] = first_group(data, 0).strip()
    return labels


def main():
    sources = {path.name: strip_comments(path.read_text(encoding="utf-8")) for path in SOURCES}
    paper = sources["paper.tex"]
    fallbacks = {
        "appref": dict(re.findall(r"\\apprefFallback\{([^}]*)\}\{([^}]*)\}", paper)),
        "mainref": dict(re.findall(r"\\mainrefFallback\{([^}]*)\}\{([^}]*)\}", paper)),
    }
    used = {kind: set() for kind in fallbacks}
    for text in sources.values():
        for kind in used:
            used[kind].update(re.findall(r"\\" + kind + r"\{([^}]*)\}", text))
    # \appref labels live in the appendix PDF, \mainref labels in the main PDF; arxiv.pdf has both.
    aux_files = {"appref": ["submission_appendix.aux", "arxiv.aux"],
                 "mainref": ["submission.aux", "arxiv.aux"]}
    errors, warnings, checked = [], [], 0
    for kind, table in fallbacks.items():
        for label in sorted(used[kind] - set(table)):
            errors.append(f"\\{kind}{{{label}}} is used but has no \\{kind}Fallback entry in paper.tex")
        for label in sorted(set(table) - used[kind]):
            warnings.append(f"\\{kind}Fallback{{{label}}} is not used")
        for name in aux_files[kind]:
            path = HERE / name
            if not path.exists():
                errors.append(f"{name} is missing; build the PDFs before running this check")
                continue
            labels = aux_labels(path)
            for label, number in sorted(table.items()):
                if label not in labels:
                    errors.append(f"{name}: label {label} is not defined (fallback {number})")
                elif labels[label] != number:
                    errors.append(f"{name}: {label} is {labels[label]}, but its \\{kind}Fallback says {number}")
                else:
                    checked += 1
    for name in ("submission.log", "submission_appendix.log", "arxiv.log"):
        path = HERE / name
        if path.exists():
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if re.search(r"Reference `[^']*' on page \d+ undefined|There were undefined references", line):
                    errors.append(f"{name}: {line.strip()}")
    for message in warnings:
        print(f"check_xref: warning: {message}")
    for message in errors:
        print(f"check_xref: ERROR: {message}", file=sys.stderr)
    if errors:
        print(f"check_xref: FAILED ({len(errors)} errors)", file=sys.stderr)
        return 1
    print(f"check_xref: OK ({len(fallbacks['appref'])} \\apprefFallback and "
          f"{len(fallbacks['mainref'])} \\mainrefFallback entries; {checked} label comparisons)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
