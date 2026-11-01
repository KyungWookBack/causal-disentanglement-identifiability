# Course report

`report.pdf` compares three causal disentanglement results and presents synthetic probes and a Norman Perturb-seq analysis. `report.tex` and `references.bib` contain the source and references; `course_report.sty` supplies the layout.

## Build

With a TeX distribution containing pdfLaTeX, BibTeX, `natbib`, and the packages listed in `report.tex`:

```bash
cd report
pdflatex -interaction=nonstopmode -halt-on-error report.tex
bibtex report
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

Alternatively, run `latexmk -pdf report.tex`.

The report and root README share the figures in `../figures/`. To rebuild those figures from the included tables, run `python run_reproduce.py` from the repository root and copy the two PDF figures from `outputs/figures/` into `figures/` before compiling.
