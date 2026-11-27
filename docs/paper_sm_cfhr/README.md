# Manuscrito Científico SM-CFHR (`docs/paper_sm_cfhr/`)

Fuentes LaTeX, figuras vectoriales PDF y tablas del manuscrito principal (`main.tex` $\to$ `main.pdf`, 26 páginas) y del material suplementario (`suplemento.tex` $\to$ `suplemento.pdf`, 11 páginas).

## Compilación reproducible (requiere `pdflatex` y `bibtex`, p. ej. TeX Live 2026)

### Manuscrito principal
```bash
cd docs/paper_sm_cfhr
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
# O con latexmk:
latexmk -pdf -interaction=nonstopmode main.tex
```

### Material suplementario
```bash
cd docs/paper_sm_cfhr
pdflatex -interaction=nonstopmode suplemento.tex
bibtex suplemento
pdflatex -interaction=nonstopmode suplemento.tex
pdflatex -interaction=nonstopmode suplemento.tex
# O con latexmk:
latexmk -pdf -interaction=nonstopmode suplemento.tex
```

## Trazabilidad de fuentes auxiliares en `appendices/`
- `appendices/B_escenarios_benchmarks.tex`: **compilado** activamente mediante `\input` dentro de `suplemento.tex` (Tablas S2–S5).
- `appendices/A_demostraciones.tex` y `appendices/C_verificacion_lean4.tex`: **borradores históricos no compilados** (su contenido actualizado se encuentra integrado directamente en las Secciones S1 y S2 de `suplemento.tex`).

## Regeneración de figuras desde OOF congelado (sin reentrenar)
```bash
python docs/paper_sm_cfhr/scripts/generate_oof_figures.py
```
