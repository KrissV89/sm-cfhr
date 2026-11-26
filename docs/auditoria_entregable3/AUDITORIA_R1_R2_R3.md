# Auditoría de Requisitos R1, R2 y R3 — `SM-CFHR` / `ASHR-Fixed`

**Fecha:** 2026-11-23  
**Estado global:** Preparación completa para evaluación confirmatoria (`OOF definitivo NO ejecutado`).

---

## 1. R1 — Novedad y Protocolo Formal de Revisión de Literatura

### 1.1. Protocolo de Búsqueda Verificable en Disco
- **Bases de datos verificadas con exports en el repositorio:**
  - **Web of Science (WoS) Core Collection:** 24 archivos de exportación en `fuentes/bibliografia/wos/` (`WoS_Q1_modelos_fv_variables_meteorologicas_orden_citas.txt` a `WoS_Q24_kalman_sesgo_NWP_hidrologia.txt`) y resumen cribado en `fuentes/bibliografia/wos/HALLAZGOS_WOS.md`.
  - **Otras bases (Scopus, IEEE Xplore, Google Scholar):** `PENDIENTE DE EXPORTACIÓN ADICIONAL EN DISCO` (no se fabrican conteos ni cadenas no archivadas).
- **Cadenas temáticas principales documentadas en WoS:**
  1. `Q1`: Modelos de generación FV con variables meteorológicas (1,000 registros exportados, ordenados por citas, 2006–2026).
  2. `Q2`: Modelos de generación FV en Colombia a nivel de planta (39 registros con DOI único).
  3. `Q3`: Contexto latinoamericano de generación/recurso solar (894 registros).
  4. `Q4` & `Q9`: Validación temporal/espacial entre plantas y normalización inter-planta.
  5. `Q12`, `Q13`, `Q15`, `Q16`: Respuesta fraccionaria, restricciones monótonas y factor de capacidad FV.
  6. `Q19`, `Q20`, `Q21`, `Q22`, `Q23`: Nivel adaptativo/Kalman, MOS/EMOS, modelos jerárquicos multi-planta, combinación de pronósticos y regresión cuantílica/mediana.
- **Criterios de inclusión / exclusión:**
  - *Inclusión:* Pronóstico de generación eléctrica FV (D+1 o multihorizonte) a escala de planta o multi-planta, con variables meteorológicas o historial causal.
  - *Exclusión:* Estimación puramente espacial de potencial sin pronóstico temporal, predicción exclusiva de irradiancia sin conversión a potencia/factor de capacidad, o estudios sin validación temporal/espacial clara.
- **Snowballing (revisión de referencias citadas y artículos semilla):**
  - Verificado en `articulos_pdf/` y `book/references.bib` (incluyendo revisiones seminales de Yang et al. 2020, Sobri et al. 2018, Raza et al. 2016, y literatura metodológica de Huber 1964, Owen 2007, Diebold & Mariano 1995, Harvey et al. 1997, Giacomini & White 2006, Clark & West 2007, Hansen et al. 2011, Cawley & Talbot 2010, Cerqueira et al. 2020).

### 1.2. Matriz Comparativa de 8 Trabajos Cercanos Verificados

| Referencia verificada | Dominio / Datos | Formulación matemática | Restricción física / Estructura | Validación | Diferencia frente a `SM-CFHR` |
|---|---|---|---|---|---|
| **Pico, Ríos & Giraldo (2026)** (`DOI: 10.1088/2515-7620/ae603d`) | 7 plantas en Colombia (potencia **simulada** SAM + NSRDB) | LR, FFN, LSTM (10 min, 24 h) | Sin pérdida robusta Huber ni L2 estructurada | Cronológica | Usa potencia simulada (SAM), no generación medida XM ni factor de capacidad robusto multi-horizonte. |
| **Trigo-González et al. (2019)** (`DOI: 10.1016/j.renene.2018.12.014`) | Plantas en Atacama, Chile | MLR exportable sobre Performance Ratio | Pérdida L2 estándar, sin cotas monótonas | Multi-planta | No emplea pérdida Huber, regularización diferenciada ni monotonicidad en irradiancia/nivel. |
| **Bessa / Almeida et al. (2015)** (`DOI: 10.1016/j.solener.2015.03.006`) | 5 plantas FV medidas | Quantile Regression Forests (no paramétrico) | Árboles cuantílicos sin formulación convexa paramétrica | Multi-planta | Enfoque de bosques no paramétricos; `SM-CFHR` es paramétrico convexo interpretable $O(np)$. |
| **Bracale / Yang et al. (2020)** (`DOI: 10.1016/j.rser.2020.109792`) | Revisión SOTA FV (956 citas en Q1) | Síntesis de modelos ML y NWP | Recomienda estandarización general | Revisión | No propone una formulación unificada Huber + L2 estructurada + monotonía en CF. |
| **Huber (1964) / Owen (2007)** (`articulos_pdf/`) | Estadística robusta general | Pérdida Huber y M-estimadores | Penalización uniforme clásica (cuando aplica) | Teórica | Base matemática; carece de L2 estructurada por familias FV, normalización por CF y cotas físicas. |
| **K1 (`ModeloAnomalias`, repo)** | 17 plantas XM Colombia (solo D+1) | Ridge (`sklearn`) sobre anomalías | Nivel fijo `media30`, sin monotonía ni Huber | Temporal D1–D3 | Delega en `sklearn.Ridge`, usa pérdida L2 sensible a outliers y solo cubre D+1. |
| **W1 (`RegresionLocal`, repo)** | 17 plantas XM Colombia (solo D+1) | Regresión local ponderada gaussiana | Método perezoso $O(nk^2)$ por consulta | Temporal D1–D3 | Alto costo por consulta, requiere identidad de planta/grupo y es sensible a outliers L2. |
| **`SM-CFHR` (este trabajo)** | **17 plantas XM Colombia, 28 escenarios (D+1 y 7–28d)** | **Huber sobre CF adimensional + MAD + L2 estructurada + L-BFGS-B analítico** | **Monotonía física (`cf30 >= 0`, `rad_pron >= 0`)** | **CV causal anidada + LOGO** | **Unifica robustez Huber, equivariancia de escala entre plantas, penalización diferenciada e implementación propia NumPy/SciPy.** |

### 1.3. Afirmación Acotada y Defendible de Novedad (R1)
`SM-CFHR` **no** se presenta como un paradigma universal de aprendizaje automático ni como un avance teórico general sobre optimización convexa. Su novedad es **acotada, específica de dominio y verificable**:
> Formulación paramétrica convexa propia en espacio adimensional de factor de capacidad (`CF`), que combina (i) pérdida Huber robusta con escalado MAD resistente a desconexiones y fallos de registro en plantas reales (XM), (ii) regularización $L_2$ diferenciada por familia de predictores (dinámica temporal vs. meteorología NWP degradable con el horizonte), y (iii) restricciones físicas de monotonicidad no negativa sobre la persistencia reciente (`cf30`) y la irradiancia pronosticada (`rad_pron`), resuelta mediante un núcleo propio L-BFGS-B con gradiente analítico en NumPy/SciPy para pronóstico fotovoltaico multi-planta y multi-horizonte.

---

## 2. R2 — Implementación Propia y Compatibilidad Scikit-Learn

- **Núcleo de aprendizaje 100% NumPy/SciPy (`IMPLEMENTADO Y PROBADO`):**
  - `src/adaptive_structured_huber.py` y `src/modelo_aditivo_huber.py` implementan la pérdida Huber, el escalado robusto (`MAD`), la matriz de penalización estructurada ($\alpha_{\text{lin}}$ vs. $\alpha_{\text{meteo}}$), las cotas de caja (`bounds` para monotonía) y el optimizador `scipy.optimize.minimize(..., method="L-BFGS-B", jac=True)` con gradiente analítico exacto.
  - Ningún paso de entrenamiento o predicción de `SM-CFHR` delega en `Ridge`, `HuberRegressor`, `SGDRegressor` ni estimadores lineales de `scikit-learn`.
- **Compatibilidad de interfaz `scikit-learn` (`IMPLEMENTADO Y PROBADO`):**
  - `AdaptiveStructuredHuberRegressor` hereda de `(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin)`.
  - Implementa `get_params`, `set_params`, compatibilidad con `sklearn.base.clone`, `sklearn.pipeline.Pipeline` y serialización `joblib`/`pickle` (`< 1e-12` de diferencia en ida y vuelta).
  - La adición de esta interfaz mantiene equivalencia exacta (`< 1e-12`) con las predicciones de la configuración congelada `Fijo_Lineal_Mono`.
- **Complejidad computacional:**
  - Costo por evaluación de objetivo y gradiente: $O(n p)$ con $p \le 17$ características en la configuración fija (`n_knots=0, max_interactions=0`).
  - Convergencia determinista en el problema convexo estricto ($\alpha_{\text{lin}} > 0, \alpha_{\text{meteo}} > 0$).

---

## 3. R3 — Evaluación Empírica e Inferencia Estadística

- **Infraestructura implementada (`IMPLEMENTADO Y PROBADO EN SINTÉTICO`):**
  - `src/ashr_oof_pipeline/evaluacion_estadistica.py`:
    - `DM-HLN-HAC`: prueba Diebold-Mariano con corrección de muestra finita Harvey-Leybourne-Newbold y varianza HAC Newey-West para horizontes $h \ge 1$.
    - `Giacomini-White (2006)`: prueba de capacidad predictiva condicional con matriz HAC.
    - `Clark-West (2007)`: prueba para modelos anidados con ajuste de varianza asintótica.
    - `Model Confidence Set (MCS)`: procedimiento de eliminación secuencial $T_{\max}$ (Hansen et al., 2011) con bootstrap estacionario.
    - `apply_holm`: corrección de multiplicidad FWER Holm-Bonferroni por familia de contrastes.
    - `bootstrap_ci_difference` y tamaños de efecto pareados ($d$ de Cohen, reducción porcentual de MAE).
- **Estado confirmatorio:** `PENDIENTE DE EJECUCIÓN OOF DEFINITIVA`. No se reporta ningún valor $p$ ni veredicto empírico sobre los 28 escenarios reales hasta autorizar y ejecutar `--ejecutar-oof-definitivo`.
