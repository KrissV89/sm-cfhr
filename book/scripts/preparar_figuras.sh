#!/usr/bin/env bash
# Copia a book/figuras/ las figuras que usa el libro.
#  - Las del manuscrito (paper/figuras/*.pdf) se convierten a PNG con pdftoppm (poppler), 150 ppp.
#  - Las de resultados/ y las capturas del dashboard ya son PNG y se copian tal cual.
# Las figuras se generan en otros sitios (scripts/generar_figuras_paper.py y los cuadernos 18, 22 y 23);
# este script solo las reúne. Ejecutar desde la raíz del repositorio o desde book/.
set -euo pipefail
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$RAIZ/book/figuras"
mkdir -p "$DEST"

for pdf in "$RAIZ"/paper/figuras/eda_*.pdf "$RAIZ"/paper/figuras/met_*.pdf "$RAIZ"/paper/figuras/res_*.pdf; do
  [ -e "$pdf" ] || continue
  base="$(basename "$pdf" .pdf)"
  pdftoppm -png -r 150 -singlefile "$pdf" "$DEST/$base"
done

copiar () {  # copiar <origen> <nombre destino>
  if [ -e "$1" ]; then cp "$1" "$DEST/$2"; else echo "falta: $1" >&2; fi
}
copiar "$RAIZ/resultados/robustez/figuras/1_sensibilidad_w1_k1.png"            rob_sensibilidad_w1_k1.png
copiar "$RAIZ/resultados/robustez/figuras/2_estabilidad_semillas.png"          rob_semillas.png
copiar "$RAIZ/resultados/robustez/figuras/3_ruido_y_atipicos.png"              rob_ruido_atipicos.png
copiar "$RAIZ/resultados/robustez/figuras/4_curva_aprendizaje.png"             rob_curva_aprendizaje.png
copiar "$RAIZ/resultados/robustez/figuras/4b_tiempos_n_p.png"                  rob_tiempos.png
copiar "$RAIZ/resultados/robustez/figuras/5_conformal_cobertura_y_ancho.png"   rob_conformal.png
copiar "$RAIZ/resultados/datasets_adicionales/figuras/mae_relativo.png"        dat_mae_relativo.png
copiar "$RAIZ/resultados/datasets_adicionales/figuras/rangos.png"              dat_rangos.png
copiar "$RAIZ/resultados/datasets_adicionales/figuras/mae_por_bloque.png"      dat_mae_por_bloque.png
for f in contexto contexto_top eda modelos modelos_movil; do
  copiar "$RAIZ/experiments/salidas/capturas/$f.png" "dash_$f.png"
done
echo "Figuras en $DEST: $(ls "$DEST" | wc -l)"
