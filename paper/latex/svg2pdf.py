"""Convert the paper's SVG figures to vector PDF with outlined text.

Usage (from paper/latex/, with `pip install cairosvg` and Ghostscript on PATH):
    python3 svg2pdf.py ../figures/fig1_same_work_ratios.svg figures/fig1_same_work_ratios.pdf

CairoSVG draws the SVG at its own size (96 px per inch). Ghostscript then converts the
embedded TrueType text to outlines, so the figure PDF contains vector paths and no fonts
(MLSys asks for Type-1 fonts only). LaTeX scales the PDF to the text width.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import cairosvg

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    source, target = sys.argv[1], sys.argv[2]
    with tempfile.TemporaryDirectory() as scratch:
        drawn = Path(scratch) / "drawn.pdf"
        cairosvg.svg2pdf(url=source, write_to=str(drawn))
        subprocess.run(
            ["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-dNoOutputFonts",
             "-sDEVICE=pdfwrite", f"-sOutputFile={target}", str(drawn)],
            check=True,
        )
