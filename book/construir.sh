#!/usr/bin/env bash
# Construye el informe en HTML estático con Jupyter Book 2 y lo comprime en _build/informe_html.zip.
#   cd book && ./construir.sh
# Pasos:
#  1. Reúne las figuras en figuras/ (scripts/preparar_figuras.sh).
#  2. Borra la caché de ejecución: las celdas leen archivos de resultados/ que pueden cambiar
#     (p. ej. cuando aparezcan resultados/optimizadores/*.csv); con caché se reutilizarían salidas viejas.
#  3. jupyter-book build --html --execute con el kernel python3 del .venv del repositorio, 2 procesos como máximo.
#  4. Traduce las etiquetas de los pies de figura y tabla que mystmd escribe en inglés («Figure 1:»).
#  5. Comprime el HTML.
set -euo pipefail
LIBRO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAIZ="$(cd "$LIBRO/.." && pwd)"
export PATH="$RAIZ/.venv/bin:$PATH"          # jupyter-book, jupyter server y el kernel python3 del .venv
cd "$LIBRO"
bash scripts/preparar_figuras.sh
rm -rf _build/execute _build/html _build/site _build/informe_html.zip
jupyter-book build --html --execute --execute-parallel 2 "$@" 2>&1 | tee _build_log.txt
python scripts/localizar_html.py _build/html
( cd _build && rm -f informe_html.zip && python -m zipfile -c informe_html.zip html/ )
mv _build_log.txt _build/registro_construccion.txt
echo "HTML: $LIBRO/_build/html/index.html"
echo "ZIP:  $LIBRO/_build/informe_html.zip ($(du -h _build/informe_html.zip | cut -f1))"
