#!/usr/bin/env python3
"""Build the arXiv source package and the PDF that arXiv will produce from it.

arXiv compiles the one file that contains \\documentclass and does not run BibTeX.  The package
therefore holds a single main.tex (the arXiv switch, arxiv-identity.tex, and paper.tex, in that
order), main.bbl, the appendix, the style files, and the figures.  The package is then compiled
again from a clean copy with pdfLaTeX only, as arXiv does, and the build fails if the result has an
undefined reference or citation, a "??", an overfull box, a placeholder or template value, or a
missing appendix.

Usage (paper/latex, TeX on PATH):  python3 make_arxiv_src.py --tarball OUT.tar.gz --pdf OUT.pdf
"""
import argparse
import io
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUPPORT = ["appendix.tex", "mlsys2025.sty", "fancyhdr.sty", "algorithm.sty", "algorithmic.sty",
           "figures/fig1_samework_ceiling.pdf", "figures/fig2_graph_ab_ratio.pdf",
           "figures/fig3_heldout_predicted_vs_measured.pdf"]
# Text that must not reach the arXiv PDF: the anonymous placeholders and the template values.
FORBIDDEN = ["[Author Name]", "[Affiliation", "[email]", "[repository URL]", "[anonymized",
             "Anonymous Author", "Anonymous Institution", "First Author", "Second Author",
             "example.org", "<owner>", "<repository>", "Under review"]
APPENDICES = ["A REPRODUCIBILITY", "B PROTOCOL TIMELINE", "C TRAFFIC MODEL",
              "D PROFILER DIAGNOSTICS", "E PREDICTIVE MODELING", "F KVQUANT QUALITY"]


def run(command, cwd):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    return result.returncode


def latex(cwd, passes):
    for _ in range(passes):
        run(["pdflatex", "-interaction=nonstopmode", "main"], cwd)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--identity", type=Path, default=HERE / "arxiv-identity.tex")
    parser.add_argument("--tarball", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    args = parser.parse_args()
    if not args.identity.exists():
        sys.exit(f"{args.identity.name} is missing: copy arxiv-identity.example.tex and fill it in")
    identity = args.identity.read_text(encoding="utf-8")
    main_tex = ("% Bytes Are Not Latency -- arXiv source: the arXiv switch, the author block, and the\n"
                "% shared paper source in one file (arXiv compiles the file with \\documentclass).\n"
                "\\def\\ARXIV{}\n" + identity.rstrip("\n") + "\n" + (HERE / "paper.tex").read_text(encoding="utf-8"))
    errors = []
    with tempfile.TemporaryDirectory() as tmp:
        build, clean = Path(tmp) / "build", Path(tmp) / "clean"
        build.mkdir()
        (build / "main.tex").write_text(main_tex, encoding="utf-8")
        for name in SUPPORT + ["references.bib", "mlsys2025.bst"]:
            (build / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(HERE / name, build / name)
        latex(build, 1)
        if run(["bibtex", "main"], build) != 0 or not (build / "main.bbl").exists():
            sys.exit("BibTeX failed; see the build log")
        latex(build, 3)

        files = ["main.tex", "main.bbl"] + SUPPORT
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for name in files:
                info = tar.gettarinfo(str(build / name), arcname=name)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                with open(build / name, "rb") as handle:
                    tar.addfile(info, handle)
        args.tarball.write_bytes(buffer.getvalue())

        # Compile the package as arXiv does: a clean copy, pdfLaTeX only.
        clean.mkdir()
        with tarfile.open(args.tarball) as tar:
            tar.extractall(clean)
        latex(clean, 3)
        log = (clean / "main.log").read_text(encoding="utf-8", errors="replace")
        pdf = clean / "main.pdf"
        if not pdf.exists():
            sys.exit("the arXiv package did not produce a PDF")
        text = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True).stdout
        flat = re.sub(r"\s+", " ", text)
        layout = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout
        squeezed = re.sub(r"\s+", "", layout)  # small caps split headings, as in "A R EPRODUCIBILITY"
        for pattern, message in ((r"Citation `[^']*' on page \d+ undefined", "undefined citation"),
                                 (r"Reference `[^']*' on page \d+ undefined", "undefined reference"),
                                 (r"There were undefined references", "undefined references"),
                                 (r"Overfull \\hbox", "overfull box")):
            if re.search(pattern, log):
                errors.append(message)
        if "??" in text:
            errors.append('"??" in the PDF')
        for value in FORBIDDEN:
            if value in flat:
                errors.append(f"placeholder or template value in the PDF: {value!r}")
        for heading in APPENDICES:
            if heading.replace(" ", "") not in squeezed:
                errors.append(f"appendix heading missing: {heading}")
        pages = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
        page_count = re.search(r"Pages:\s+(\d+)", pages).group(1)
        shutil.copy2(pdf, args.pdf)
    if errors:
        for message in errors:
            print(f"make_arxiv_src: ERROR: {message}", file=sys.stderr)
        sys.exit(1)
    print(f"make_arxiv_src: OK: {args.tarball.name} ({len(files)} files), {args.pdf.name} ({page_count} pages)")


if __name__ == "__main__":
    main()
