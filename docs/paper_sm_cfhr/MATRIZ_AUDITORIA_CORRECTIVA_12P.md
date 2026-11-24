# Matriz de Auditoría Científica Correctiva (12 Puntos) — SM-CFHR

**Fecha:** 2026-11-24  
**Repositorio:** `https://github.com/KrissV89/sm-cfhr` (`v1.0.3-polyform`, protocolo `v1.0.3`)

## 1. Correcciones editoriales y matemáticas ejecutadas en el manuscrito

| # | Problema identificado | Evidencia | Modificación realizada | Verificación final |
|---|---|---|---|---|
| 1 | MCS y SPA usados en H2 como prueba directa de superioridad | `sections/07_protocolo_inferencia.tex` y `tab_res_hipotesis.tex` | Aclarado que supervivencia en MCS ($\alpha=0{,}10$) indica pertenencia al conjunto de mejores modelos indistinguibles (superioridad solo frente a los eliminados) y que SPA requiere contraste bidireccional | Revisado en `07_protocolo_inferencia.tex` y `tab_res_hipotesis.tex` |
| 2 | Sesgo de selección *post hoc* al elegir la «mejor referencia de biblioteca» en el mismo conjunto de evaluación | `sections/07`, `08`, `09` | Explicitado el sesgo *ex post* y la necesidad de complementar con referencias fijadas *a priori* (`MM30`, `Ridge`, `XGBoost`) y pruebas simultáneas (SPA, MCS) | Revisado en `07`, `08` y `09` |
| 3 | Criterio $T_{\mathrm{eff}} \ge 2H$ presentado sin justificación suficiente | `sections/07_protocolo_inferencia.tex` | Justificado como filtro operativo prudencial del protocolo congelado (al menos dos ciclos de horizonte efectivos antes de inferencia HAC), no como umbral asintótico universal | Revisado en `07_protocolo_inferencia.tex` |
| 4 | Afirmación incorrecta de que $\mathrm{MAE}_{\mathrm{macro}}$ en kWh elimina la dominancia por capacidad de plantas grandes | `sections/03_datos_eda.tex`, `07`, `09` | Corregido: macro-promediar iguala la **longitud de serie**, mientras que en $\text{kWh/día}$ los errores siguen ponderados por capacidad instalada; se requiere reportar $\mathrm{MAE}_{\mathrm{CF}}$ y $\mathrm{Skill}$ | Revisado en `03`, `07` y `09` |
| 5 | Punto de ruptura del $50\%$ atribuido sin matiz cuando $\mathrm{MAD}(\mathbf{y})=0$ | `sections/05_modelo_sm_cfhr.tex` | Precisado que el $50\%$ aplica en $\mathrm{MAD}(\mathbf{y}) > 0$; el respaldo $\operatorname{std}(\mathbf{y})$ en $\mathrm{MAD}(\mathbf{y})=0$ reduce el punto de ruptura a $0\%$ | Verificado en `05_modelo_sm_cfhr.tex` (pág. 16 del PDF) |
| 6 | Recuento de ablaciones («seis configuraciones frente al modelo completo») | `sections/07_protocolo_inferencia.tex`, `08_resultados.tex` | Corregido a **cinco modificaciones estructurales** frente al modelo completo ($6$ variantes en total) | Verificado en `07`, `08` y Tabla 4 |
| 7 | Demostración de coercividad con dos casos incompletos para sucesiones oscilantes | `appendices/A_demostraciones.tex` | Reemplazada por prueba directa en complemento de caja $(R_w, R_b)$ que cubre toda sucesión divergente $\|(b_m, \mathbf{w}_m)\|_2 \to \infty$ | Verificado en `appendices/A_demostraciones.tex` |
| 8 | Discrepancia en Lean 4: `gradB_lipschitz` solo prueba $\partial J/\partial b$, no $\nabla J$ completo | `SMCFHR/Gradient.lean:187-221` vs. Teorema 1(2) | Documentado explícitamente en Sección 6.4 y Tabla C5 que Lean verifica $\partial J/\partial b$, mientras la cota espectral $L_J$ de $\nabla J$ permanece en el Apéndice A | Verificado en `06_propiedades_y_lean4.tex` y `C_verificacion_lean4.tex` |
| 9 | Conclusiones de calibración fuerte basadas en $120$ simulaciones Monte Carlo | `sections/07_protocolo_inferencia.tex` | Reencuadrado como diagnóstico exploratorio de precisión limitada ($\text{SE} \approx 2{,}0$ pp al $5\%$), suficiente para detectar la distorsión grave de GW ($0{,}2083$) pero no para certificar calibración fina | Revisado en `07_protocolo_inferencia.tex` |
| 10 | Inventario `tab:flota17` en el cuerpo principal y formato de autores en `main.tex` | `sections/03_datos_eda.tex` y `main.tex:53` | Movida `tab:flota17` al Apéndice B (`Tabla B1`) y separados los dos `\author` en `main.tex` | Verificado visualmente en págs. 1 y 34 del PDF |
| 11 | Metadatos incompletos en `jensen2024solarqc` y verificación integral `.bib`/`.tex` | `bibliography/bibliography.bib` | Completados autores y libro de NREL (`NREL/TP-5D00-88300`, Cap. 4); verificado `BIB_TOTAL=75, TEX_CITES=75, UNCITED=0, MISSING=0` | Verificado con `bibtex` y script de auditoría |
| 12 | Inspección visual de figuras y maquetación del PDF | `pdftoppm` + inspección directa | Corregido solapamiento de $\pm\delta$ con leyenda en `fig_huber_estructura.pdf` y ancho de Tablas 1 y 2 | Verificado en PDF compilado y PNGs |

## 2. Cuestiones que requieren cambios en el protocolo experimental (fuera de autorización)
Las siguientes decisiones pertenecen al protocolo experimental congelado (`v1.0.3`) y **no fueron modificadas**:
1. **Métrica primaria oficial:** cambiar la métrica primaria de $\mathrm{MAE}_{\mathrm{macro}}$ en $\text{kWh/día}$ a $\mathrm{MAE}_{\mathrm{CF}}$ o $\mathrm{Skill}$ requeriría redefinir la regla de evaluación del pipeline OOF.
2. **Umbral $T_{\mathrm{eff}} \ge 2H$ y diseño de comparaciones:** modificar el filtro de tamaño muestral efectivo o reemplazar la selección del mejor modelo de biblioteca en `evaluate_r3_superiority` requeriría alterar el código de inferencia post-OOF.
3. **Número de réplicas Monte Carlo ($120$):** ampliar el estudio de calibración de tamaño de prueba requeriría ejecutar nuevas simulaciones fuera del protocolo congelado.
