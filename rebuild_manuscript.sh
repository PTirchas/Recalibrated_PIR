#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

python3 -m analytics.run_all --verify-only

cd manuscript
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex

pdflatex -interaction=nonstopmode -halt-on-error supplementary.tex
bibtex supplementary
pdflatex -interaction=nonstopmode -halt-on-error supplementary.tex
pdflatex -interaction=nonstopmode -halt-on-error supplementary.tex
