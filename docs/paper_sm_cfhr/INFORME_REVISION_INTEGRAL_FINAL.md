# Informe Final de Revisión Científica, Gráfica y Editorial — SM-CFHR

**Fecha:** 2026-11-24  
**Repositorio:** `https://github.com/KrissV89/sm-cfhr` (`v1.0.3-polyform`, protocolo `v1.0.3`)

## 1. Cambios realizados en esta revisión integral

### A. Estandarización gráfica de las 14 figuras del manuscrito (`scripts/generate_conceptual_figures.py` y `scripts/generate_eda_figures.py`)
- **5 figuras conceptuales/matemáticas regeneradas:** tipografía sans-serif (`DejaVu Sans`) a escala de referencia $12\text{--}13\text{ pt}$, PDF vectorial con fuentes TrueType incrustadas (`pdf.fonttype = 42`, cero fuentes Type 3), etiquetas/unidades/leyendas en inglés y estilos distinguibles en escala de grises. Eliminadas cadenas obsoletas de `fig_pipeline_reproduccion.pdf` (`Veredicto R3`, `v1.0.2`, `615a5af`, nombres de scripts y archivos).
- **9 figuras EDA regeneradas desde datos reales (`InvML/base_generacion_clima/base_planta_dia.csv`):** mismo estándar vectorial TrueType sans-serif a $12\text{--}13\text{ pt}$ en inglés, sin alterar ningún CSV original. Verificado con `pdffonts` (`CID TrueType`, `emb=yes`).

### B. Rigor metodológico, complejidad y redacción científica natural
- **Eliminación de afirmaciones de prerregistro no verificadas:** sustituido «pre-registrado» por «preespecificado» en `main.tex`, `01_introduccion.tex`, `07_protocolo_inferencia.tex`, `08_resultados.tex` y `10_conclusiones_disponibilidad.tex`.
- **Prosa científica sobria:** eliminados anuncios mecánicos de sección («Esta sección analiza...», «Antes de formular...») y unificadas las transiciones argumentativas.
- **Complejidad computacional (`06_propiedades_y_lean4.tex`):** diferenciación explícita entre coste por evaluación $O(np)$, entrenamiento individual de $K$ iteraciones L-BFGS-B $O(np + Knp)$ (memoria $O(np + mp)$), búsqueda anidada TPE $O(B \cdot K_{\mathrm{cv}} \cdot Knp)$ y despliegue recursivo $O(Hp)$ por origen.
- **Ablaciones con hipótesis identificable (`07_protocolo_inferencia.tex`):** cada una de las seis ablaciones se presenta con su hipótesis específica, el componente modificado y su contraste frente al modelo completo.
- **Separación de licencias (`10_conclusiones_disponibilidad.tex`):** delimitación explícita entre PolyForm Noncommercial~1.0.0 para el código original (`edmolinares@soulspacers.com`), términos de uso de datos externos (XM, Open-Meteo, GEM), licencias de dependencias de terceros y derechos académicos del manuscrito.

### C. Nota de transparencia sobre recursos de referencia
- El archivo `PaperMethodologyExample.pdf` citado como guía metodológica no se encontraba presente en el sistema de archivos local durante la auditoría, por lo que los estándares gráficos y editoriales se aplicaron directamente según los criterios explícitos especificados en la revisión.

## 2. Cuestiones científicas que permanecen abiertas
- **Evaluación empírica fuera de muestra (OOF):** la ejecución definitiva de `SM-CFHR-Fixed` ($B=0$) y `SM-CFHR-HPO` ($B=25$) sobre las $28$ configuraciones y los $22$ modelos de referencia sigue en curso en equipo independiente; las tablas de resultados, pruebas DM--HLN--HAC / MCS / SPA, ablaciones empíricas y veredicto de las hipótesis H1--H4 permanecen marcados con `[PENDIENTE: ...]` sin cifras simuladas.
- **Validación externa independiente:** no existen aún conjuntos de datos fotovoltaicos externos a la cohorte colombiana (XM/Open-Meteo/GEFSv12), por lo que la generalización entre datasets y entre dominios permanece delimitada como trabajo futuro.
