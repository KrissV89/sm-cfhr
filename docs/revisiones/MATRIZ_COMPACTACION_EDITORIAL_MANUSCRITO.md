# Matriz de Compactación Editorial del Manuscrito SM-CFHR

## 1. Resumen cuantitativo de compactación

| Documento | Extensión previa | Extensión compactada | Reducción / Cambio |
| --- | --- | --- | --- |
| Manuscrito principal (`main.pdf`) | 38 páginas (`1 426 853` bytes) | 27 páginas (`1 060 926` bytes) | $-11$ páginas ($-28{,}9\%$), incluyendo referencias |
| Material Suplementario (`suplemento.pdf`) | 6 páginas (`433 514` bytes) | 12 páginas (`827 007` bytes) | $+6$ páginas (absorbe demostraciones auxiliares, parámetros operativos, Algoritmo S1, fórmulas HAC/$T_{\mathrm{eff}}$, EDA complementario y diagramas) |

### Distribución aproximada por sección en `main.pdf` (27 páginas)

| Sección | Inicio (`main.aux`) | Rol editorial |
| --- | --- | --- |
| Título, autores y resumen | p. 1 | Portada y síntesis ejecutiva |
| 1. Introducción | p. 2 | Motivación física, diseño y contribuciones |
| 2. Fundamentos y estado del arte | p. 3 | Contexto bibliográfico y delimitación de aportes |
| 3. Datos, depuración y EDA | p. 4 | Fuentes, filtros causales y 2 figuras clave (pp. 4–6) |
| 4. Conjuntos experimentales | p. 5 / 7 | Definiciones físicas, objetivos, predictores e IP/ET |
| 5. Modelo SM-CFHR | p. 8 | Preprocesamiento MAD, objetivo, $D_\alpha$, $\mathcal{C}$ y reconstrucción |
| 6. Propiedades matemáticas y Lean 4 | p. 10 | Gradiente exacto, Teoremas 1–2, KKT, influencia y Lean 4 |
| 7. Protocolo e inferencia estadística | p. 12 | Métricas, alineación macro ($h_{\mathrm{serie}}=1$), hipótesis y diagnósticos |
| 8. Resultados | p. 14 | **Íntegra:** 6 figuras OOF, 3 tablas, derrotas, ablaciones e hipótesis H1–H4 |
| 9. Discusión | p. 18 | Interpretación empírica y 8 limitaciones completas |
| 10. Conclusiones y disponibilidad | p. 21 | Síntesis teórica/empírica y disponibilidad académica |
| Bibliografía | pp. 22–27 | Referencias completas sin `???` |

---

## 2. Matriz de contenido conservado, resumido, trasladado y eliminado

| Elemento / Sección | Origen | Acción | Destino / Estado | Justificación científica y editorial |
| --- | --- | --- | --- | --- |
| Resumen (`\abstract`) | `main.tex` | **Resumido** | `main.tex` (p. 1) | Reduce ~20% de densidad narrativa manteniendo todas las cifras OOF verificadas, Lean 4 y la advertencia sobre referencias en energía bruta. |
| Introducción y advertencia de reutilización adaptativa | `01_introduccion.tex` | **Resumido** | `main.tex`, §1 (p. 2) | Elimina repetición de porcentajes ya cubiertos en Resumen y Resultados; conserva motivación de $39{,}6\times$, colas asimétricas, advertencia \cite{cawley2010overfit} y 3 contribuciones. |
| Estado del arte y atribución metodológica | `02_fundamentos_estado_arte.tex` | **Resumido** | `main.tex`, §2 (pp. 3–4) | Condensa la enumeración de las 12 adaptaciones residuales y las pruebas estadísticas manteniendo intactas todas las citas bibliográficas y la atribución de aportes. |
| Fuentes de datos, correlación $0{,}976$ vs. $0{,}615$, filtros $30\,964 \to 12\,756 \to 12\,725$ y sesgos de selección | `03_datos_eda.tex` | **Conservado** | `main.tex`, §3.1 (p. 4) | Evidencia indispensable de calidad causal y validez externa. |
| Figuras EDA clave: `fig:eda_univ` (distribución FC + rendimiento específico) y `fig:eda_dep` (autocorrelación y dependencia espacial) | `03_datos_eda.tex` | **Conservado** | `main.tex`, §3.2, Figuras 1–2 (p. 6) | Motivan directamente la pérdida de Huber, MAD, restricciones en $\mathcal{M}_+$, medias móviles, validación *leave-one-group-out* e inferencia HAC. |
| Narrativa EDA de 4 preguntas y 5 bloques de figuras apiladas | `03_datos_eda.tex` | **Resumido / Trasladado al suplemento** | Texto en 3 puntos (§3.2); figuras `eda_flujo_cohorte`, `eda_mapa_plantas`, `eda_cobertura_temporal`, `eda_horas_y_anomalos`, `eda_estacionalidad` y `eda_clima` en `suplemento.tex` §S5 (Figuras S2–S4) | Elimina 3,5 páginas de figuras exploratorias secundarias sin perder ninguna evidencia empírica ni sesgo documentado. |
| Ecuaciones esenciales de conjuntos (`E^{\max}`, `c_{i,t+h\mid t}`, `Y_{i,t}^{(H)}`, `\mathrm{cf}_{30/7/1}`, `x_5`–`x_7`, $\gamma_h$) y definición de IP / ET | `04_conjuntos_experimentales.tex` | **Conservado** | `main.tex`, §4.1–4.3 (pp. 5–8) | Formulación matemática mínima indispensable para reproducibilidad y comprensión sin ambigüedad (incluyendo que ET usa historial local causal). |
| Caja `Algoritmo 1` de despliegue recursivo y resguardos de arranque (`cf30 <- 0.15`, `n_tr >= 15`, `n_va >= 5`) | `04_conjuntos_experimentales.tex` | **Trasladado al suplemento** | `suplemento.tex` §S3.2 (`Algoritmo S1` y parámetros operativos) | El texto principal conserva una síntesis causal de 4 líneas; el pseudocódigo completo y los umbrales operativos secundarios pasan al suplemento. |
| Diagrama `fig:sm_cfhr_arq`, escalado MAD (`eq:mad_scale` con nota de punto de ruptura), objetivo $J(b,\mathbf{w})$, Huber $\rho_\delta$, $D_\alpha$, $\mathcal{C}$, `Fixed` vs. `HPO` y reconstrucción física (`eq:pred_fisica`) | `05_modelo_sm_cfhr.tex` | **Conservado** | `main.tex`, §5 (pp. 8–10) | Núcleo matemático completo del estimador propuesto (Criterio 1). |
| Pisos numéricos secundarios (`s_j > 10^{-8}`, imputación `0`) y diagrama `fig:despliegue_mh` | `05_modelo_sm_cfhr.tex` | **Trasladado al suplemento** | `suplemento.tex` §S3.2 y §S5 (`Figura S5`) | Evita redundancia visual con la Figura 3 manteniendo en §5.3 la advertencia de no convexidad del despliegue recursivo. |
| Gradiente analítico exacto (`eq:gradiente_smcfhr`), Teoremas 1–2, alcance de influencia/monotonía y síntesis Lean 4 | `06_propiedades_y_lean4.tex` | **Conservado / Resumido** | `main.tex`, §6 (pp. 10–12) | Preserva los enunciados matemáticos exactos, la ausencia de cota de apalancamiento en $Z$, el límite de monotonía parcial y el alcance real de Lean 4. |
| Hessiana por tramos (`eq:hessiana_regiones`), bloque explícito KKT, desglose Big-O y `fig:huber_estructura` | `06_propiedades_y_lean4.tex` | **Trasladado al suplemento** | `suplemento.tex` §S1.1 y §S5 (`Figura S6`) | Las demostraciones completas, Hessiana y KKT ya residen formalmente en S1; se conserva referencia cruzada directa. |
| Métrica `MAE_macro`, escalas (`MAE_CF,macro`, `nMAE_rel,gen`, `Skill_RMSE`), transparencia `Fixed` vs. `HPO` y tiempos de pared (`59x`) | `07_protocolo_inferencia.tex` | **Conservado** | `main.tex`, §7.1 (pp. 12–13) | Esencial para interpretar correctamente la escala y el presupuesto experimental. |
| Contradicción residual del filtro `T_eff >= 2H` en §7.2 y en H2 | `07_protocolo_inferencia.tex` | **Eliminado / Corregido** | Unificado en §7.2 bajo $h_{\mathrm{serie}}=1$ y $z_t^{\mathrm{macro}}$ sobre los $28$ escenarios completos | Resuelve la contradicción entre el filtro inicial erróneo ($T_{\mathrm{eff}} \ge 2H$ sobre bloques no solapados) y la inferencia definitiva macro-alineada (Criterio 6). |
| Fórmulas HAC (`\hat{\Omega}_{\mathrm{HAC}}`, `T_{\mathrm{eff}}`, `z_t^{\mathrm{macro}}`), política de varianza nula, diagnóstico GW AR(1) y `fig:validacion_anidada` | `07_protocolo_inferencia.tex` | **Trasladado al suplemento** | `suplemento.tex` §S4.1 y §S5 (`Figura S7`) | Fórmulas estadísticas auxiliares y diagrama de CV anidada pasan al suplemento preservando la lógica completa en §7.2. |
| Sección completa de Resultados: 6 figuras `fig_res_oof_*`, `tab:res_familias`, `tab:res_ablaciones`, `Fixed` vs. `HPO`, derrotas (`+24,24%`), métricas CF, ablación y conformal | `08_resultados.tex` | **Conservado íntegramente** | `main.tex`, §8 (pp. 14–19) | Cumple estrictamente el Criterio 4: cero recortes en resultados empíricos, figuras OOF ni derrotas. |
| Estado de **H1** en `tab:res_hipotesis` | `08_resultados.tex` (Tabla 3) | **Corregido** | `main.tex`, §8.4, Tabla 3 (p. 19) | Pasa de *«Confirmada frente a referencias en kWh»* a *«Respaldada solo frente a referencias en kWh (no aislada)»*, explicitando que la evidencia cerrada no separa normalización y estimador (Criterio 7). |
| Oración introductoria duplicada en Discusión | `09_discusion.tex` | **Eliminado** | `main.tex`, §9.1 (p. 18) | Suprime duplicación literal consecutiva (Criterio 5). |
| Atribución causal de peor desempeño de HPO a varianza de selección | `09_discusion.tex` (§9.1, ítem 3) | **Corregido** | `main.tex`, §9.1, ítem 3 (p. 18) | Reformulado como hipótesis compatible con sobreajuste/inestabilidad interna sin atribuir causalidad no aislada experimentalmente (Criterio 8). |
| Ocho limitaciones y amenazas a la validez | `09_discusion.tex` (§9.2) | **Conservado** | `main.tex`, §9.2 (pp. 19–20) | Mantiene íntegras todas las limitaciones metodológicas (Criterio 10). |
| Síntesis de Conclusiones y Disponibilidad | `10_conclusiones_disponibilidad.tex` | **Resumido / Depurado** | `main.tex`, §10 (p. 21) | Evita repetir verbatim Discusión y Resumen; elimina correo comercial personal y detalles operativos internos de importación local (Criterios 5 y 9). |
| Encabezado duplicado `\section` en apéndice B y detalles operativos internos de registro en S4 | `B_escenarios_benchmarks.tex`, `suplemento.tex` | **Corregido / Depurado** | `suplemento.tex` §S3–S4 | Corrige numeración vacía de S3 (`\section` $\to$ `\subsection`) y limpia menciones operativas internas (`228 reciclajes de proceso`, `entre máquinas independientes`). |

---

## 3. Auditoría de referencias cruzadas (`main.tex` $\leftrightarrow$ `suplemento.tex`)

Todas las referencias cruzadas fueron verificadas tras la compilación completa (`pdflatex` + `bibtex` $\times 2$) con `0` advertencias de referencias indefinidas y `0` `???` en `.bbl`:

- **Sección~S1** (`sup:demos`), **Proposición~S1** (`sup:prop_kkt`): demostraciones de convexidad, existencia, unicidad, KKT, Hessiana por tramos (`eq:sup_hessiana`) y complejidad.
- **Sección~S2** (`sup:lean4`), **Tabla~S1** (`tab:sup_lean4`): correspondencia de las $26$ declaraciones verificadas en Lean~4 (`v4.36.0`).
- **Sección~S3** (`sup:tablas`, `sup:operativos`), **Tablas~S2--S5** (`tab:flota17`, `tab:config_fixed_hpo`, `tab:escenarios28`, `tab:benchmarks22`), **Algoritmo~S1** (`alg:sup_rollout`): inventarios, parámetros operativos secundarios y despliegue recursivo causal.
- **Sección~S4** (`sup:repro`, `sup:stats`, `sup:env`), **Figura~S1** (`fig:sup_pipeline`): fórmulas HAC / $T_{\mathrm{eff}}$ / $z_t^{\mathrm{macro}}$, política de varianza nula, diagnóstico GW, entorno computacional y verificación SPA/MCS (`arch 8.0.0`).
- **Sección~S5** (`sup:figuras`), **Figuras~S2--S7**: figuras exploratorias complementarias (`eda_flujo_cohorte`, `eda_mapa_plantas`, `eda_cobertura_temporal`, `eda_horas_y_anomalos`, `eda_estacionalidad`, `eda_clima`) y diagramas metodológicos (`fig_despliegue_multihorizonte`, `fig_huber_estructura`, `fig_validacion_anidada`).
