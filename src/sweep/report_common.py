"""Shared by the report builders: the colour palette, the plot style, and a small renderer that turns a list of
blocks into a PDF with pdflatex.

A block is a tuple:
    ("h1", text)   ("h2", text)   ("p", text)   ("meta", text)   ("note", text)
    ("ul", [items])
    ("fig", file_name, caption)                      file_name is a key of the `figures` dict given to build_pdf
    ("table", header, rows[, column_layout])
Inside text, **bold** and `code` are understood, and common symbols (±, η, σ, Δ, ×, ≤ ...) are converted.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# palette: first three categorical slots (validated all-pairs on the light surface), blue sequential ramp, blue-grey-red diverging
C_TVB, C_F32, C_F64 = "#2a78d6", "#eb6834", "#1baf7a"
SURF, INK, INK2, GRID, HOLE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
DIV = LinearSegmentedColormap.from_list(
    "div", ["#0d366b", "#3987e5", "#f0efec", "#e34948", "#8f1f1f"]
)
STYLE = dict(
    {
        "figure.facecolor": SURF,
        "axes.facecolor": SURF,
        "savefig.facecolor": SURF,
        "font.size": 9,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "legend.frameon": False,
        "font.family": "DejaVu Sans",
    }
)
plt.rcParams.update(STYLE)


SYM = {
    "±": r"$\pm$",
    "−": r"$-$",
    "×": r"$\times$",
    "≈": r"$\approx$",
    "≤": r"$\le$",
    "≥": r"$\ge$",
    "→": r"$\to$",
    "η": r"$\eta$",
    "σ": r"$\sigma$",
    "Δ": r"$\Delta$",
    "μ": r"$\mu$",
    "·": r"$\cdot$",
    "–": "--",
    "|": r"\textbar{}",
    ">": r"\textgreater{}",
    "<": r"\textless{}",
}


def tex(t):
    t = str(t)
    t = re.sub(r"10\^\{(-?\d+)\}", lambda k: "10⟪%s⟫" % k.group(1), t)
    t = "".join("\\" + c if c in "&%#_{}$" else c for c in t.replace("\\", r"\textbackslash{}"))
    for a, b in SYM.items():
        t = t.replace(a, b)
    t = re.sub(r"10⟪(-?\d+)⟫", r"$10^{\1}$", t)
    t = re.sub(r'"([^"]*)"', r"``\1''", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t)
    t = re.sub(r"`(.+?)`", r"\\texttt{\1}", t)
    t = re.sub(r"(?<![\w$.{-])-(?=\d)", "$-$", t)  # a leading hyphen on a number is a minus sign
    return re.sub(r"(?<=\d),(?=\d{3})", "{,}", t)


def faded_map(ax, values, resolved, extent, cmap, vmin, vmax, fade=0.4):
    """Draw a map without throwing estimates away: every cell that has a value is coloured; cells that are not
    `resolved` (the estimate is within its noise) are drawn faded; cells with no value at all are grey.
    Returns the image of the resolved cells, for the colour bar."""
    common = dict(origin="lower", extent=extent, aspect="auto")
    ax.set_facecolor(SURF)
    no_value = ~np.isfinite(values)
    ax.imshow(
        np.where(no_value, 1.0, np.nan), cmap=ListedColormap([HOLE]), vmin=0, vmax=1, **common
    )
    ax.imshow(
        np.where(resolved, np.nan, values), cmap=cmap, vmin=vmin, vmax=vmax, alpha=fade, **common
    )
    return ax.imshow(np.where(resolved, values, np.nan), cmap=cmap, vmin=vmin, vmax=vmax, **common)


def build_pdf(blocks, figures, out_path):
    """Render the blocks to a PDF at out_path. `figures` maps a figure's file name to its path."""
    out = [
        r"\documentclass[10pt,letterpaper]{article}",
        r"\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}\usepackage[scaled=0.95]{helvet}\renewcommand{\familydefault}{\sfdefault}",
        r"\usepackage[margin=0.85in]{geometry}\usepackage{graphicx,booktabs,array,microtype,xcolor,parskip,enumitem}\usepackage[font={small,color=ink2},labelfont=bf,skip=4pt,justification=raggedright,singlelinecheck=false]{caption}",
        r"\definecolor{ink2}{HTML}{52514E}\setlist[itemize]{leftmargin=1.3em,itemsep=2pt,topsep=3pt}\pagestyle{plain}\raggedbottom\setlength{\emergencystretch}{2em}",
        r"\begin{document}",
    ]
    for b in blocks:
        k = b[0]
        if k == "h1":
            out.append(r"{\LARGE\bfseries %s\par}\vspace{4pt}" % tex(b[1]))
        elif k == "meta":
            out.append(r"{\small\color{ink2} %s\par}" % tex(b[1]))
        elif k == "h2":
            out.append(r"\vspace{8pt}\pagebreak[3]{\large\bfseries %s\par}\nopagebreak" % tex(b[1]))
        elif k == "p":
            out.append(tex(b[1]) + r"\par")
        elif k == "note":
            out.append(r"\begin{quote}\small\color{ink2} %s\end{quote}" % tex(b[1]))
        elif k == "ul":
            out.append(
                r"\begin{itemize}"
                + "".join(r"\item " + tex(i) + "\n" for i in b[1])
                + r"\end{itemize}"
            )
        elif k == "fig":
            out.append(
                r"\begin{minipage}{\linewidth}\centering\includegraphics[width=\linewidth]{%s}\captionof{figure}{%s}\end{minipage}\par\vspace{4pt}"
                % (os.path.abspath(figures[b[1]]), tex(b[2]))
            )
        elif k == "table":
            hd, rows = b[1], b[2]
            last_text = any(re.search(r"[a-z]{4}", str(r[-1])) for r in rows)
            spec = "l" + "r" * (len(hd) - 2) + ("l" if last_text else "r")
            if len(b) > 3:  # an explicit column layout, for tables of text
                spec = b[3]
            out.append(
                r"\begin{minipage}{\linewidth}\centering\small\begin{tabular}{@{}%s@{}}\toprule "
                % spec
                + " & ".join(r"\textbf{%s}" % tex(h) for h in hd)
                + r" \\ \midrule "
                + " ".join(" & ".join(tex(c) for c in r) + r" \\" for r in rows)
                + r" \bottomrule\end{tabular}\end{minipage}\par\vspace{6pt}"
            )
    out.append(r"\end{document}")
    with tempfile.TemporaryDirectory() as td:
        open(os.path.join(td, "writeup.tex"), "w").write("\n".join(out))
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "writeup.tex"],
            cwd=td,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if r.returncode != 0 or not os.path.exists(os.path.join(td, "writeup.pdf")):
            shutil.copy(os.path.join(td, "writeup.tex"), out_path + ".failed.tex")
            sys.exit(
                "pdflatex failed; source kept at " + out_path + ".failed.tex\n" + r.stdout[-1500:]
            )
        warnings = [line for line in r.stdout.splitlines() if "Overfull" in line]
        shutil.copy(os.path.join(td, "writeup.pdf"), out_path)
    print(
        "wrote",
        out_path,
        f"{os.path.getsize(out_path) / 1e6:.2f} MB;",
        len(warnings),
        "overfull boxes",
    )
    for line in warnings[:8]:
        print("  ", line)
