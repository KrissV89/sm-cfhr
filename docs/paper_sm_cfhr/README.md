# Manuscrito Científico SM-CFHR (`docs/paper_sm_cfhr/`)

Manuscrito LaTeX completo correspondiente a la distribución oficial **`v1.0.3-polyform`** (protocolo científico **`v1.0.3`**, núcleo `src/adaptive_structured_huber.py` con SHA-256 `27a9fda4fa8b7ef0d95faf9e925e47ade9ed8f25e9ea5d97d6d7cf1c00ddde83`).

## Compilación en modo placeholder (sin resultados OOF integrados)
```bash
cd docs/paper_sm_cfhr
python scripts/integrate_results.py
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Integración automática cuando llegue el ZIP de resultados OOF
```bash
python docs/paper_sm_cfhr/scripts/integrate_results.py --results-zip /ruta/al/resultados_oof_v1.0.2.zip
cd docs/paper_sm_cfhr
pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```
El script verifica el `manifest_ashr_oof.json` y el SHA-256 del núcleo (`27a9fda4fa8b7ef0d95faf9e925e47ade9ed8f25e9ea5d97d6d7cf1c00ddde83`) sin reentrenar ningún modelo.
