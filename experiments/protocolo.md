# Protocolo Pre-Registrado de Evaluación OOF Definitiva — `SM-CFHR` (`ASHR`)

**Fecha de pre-registro:** 2026-11-23  
**Estado de ejecución:** `PRE-REGISTRADO — OOF DEFINITIVO NO EJECUTADO`  
**Configuración congelada:** `experiments/config_ashr_congelada.json`  
**SHA-256 núcleo (`src/adaptive_structured_huber.py`):** Ver `resultados/manifiesto_preparacion_entregable3.json`

---

## 1. Separación Estricta: Desarrollo vs. Evaluación Confirmatoria

- **Decisiones cerradas en fase de desarrollo (previas a este protocolo):**
  - Selección de arquitectura fija `SM-CFHR` (`n_knots=0, max_interactions=0, anchor="none", structured=True, use_mono=True`) y descarte del selector multimodo por criterio de parsimonia.
  - Definición del espacio de búsqueda continuo `(delta, alpha_lin, alpha_meteo)` y del esquema de CV causal anidada.
  - Exclusión de `CF-BRH` y fijación del inventario oficial de 22 modelos de referencia en los 28 escenarios solares.
- **Regla de no-reajuste confirmatorio:** Una vez autorizada la ejecución `--ejecutar-oof-definitivo`, no se modificará el código del modelo, ni los rangos de HPO, ni la lista de escenarios o benchmarks.

## 2. Dominio y Alcance Experimental

- **Dominio exclusivo:** 17 plantas fotovoltaicas de Colombia (XM + Open-Meteo / GEFSv12), organizadas en **28 escenarios oficiales**:
  - 4 escenarios `D+1` (`ET` / `IP` × con clima / sin clima).
  - 8 escenarios `Recursivo sin clima` (`7, 14, 21, 28` días × `ET` / `IP`).
  - 8 escenarios `Directo sin clima` (`7, 14, 21, 28` días × `ET` / `IP`).
  - 8 escenarios `Recursivo + GEFSv12` (`7, 14, 21, 28` días × `ET` / `IP`).
- **Exclusión declarada:** No se evalúan datasets externos (`cpu_act`, `sulfur`, `trafico_i94`, `44136`, `44138`, `44147`). Esta modificación de alcance respecto a la guía general del curso queda marcada como `PENDIENTE DE VALIDACIÓN DOCENTE`.

## 3. Comparación Dual de Presupuestos

Para garantizar equidad frente a los 22 benchmarks oficiales:
1. **`SM-CFHR-Fixed` ($B=0$):** Hiperparámetros congelados (`delta=1.35, alpha_lin=0.01, alpha_meteo=0.10`), 0 evaluaciones de HPO.
2. **`SM-CFHR-HPO` ($B=25$):** Optimización bayesiana TPE por pliegue externo dentro de CV causal anidada (`src/ashr_oof_pipeline/optimizacion_hiperparametros.py`, protocolo `v1.0.3-exact-recursive-state`). En escenarios recursivos multihorizonte (`H = 7, 14, 21, 28`), cada trial interno evalúa `MAE_macro_kWh` reproduciendo exactamente el mismo motor causal que la evaluación externa (`rollout_recursive_from_origin` con `PlantRecursiveState` / `PlantRecursiveWeatherState` y ventanas deslizantes reales de 7 y 30 días). Los estudios SQLite de Optuna de `v1.0.2` se rechazan automáticamente como incompatibles.

## 4. Métricas e Inferencia Estadística Pre-Especificada

- **Métrica primaria:** `MAE_macro` por planta en `kWh/día` (en directos acumulados, también reporte normalizado por horizonte).
- **Métricas secundarias:** `RMSE`, `MBE` (sesgo medio), `MAE_top5` (cola superior del 5% de errores absolutos).
- **Pruebas estadísticas:**
  - `DM-HLN-HAC` (Diebold-Mariano con corrección Harvey-Leybourne-Newbold y varianza HAC Newey-West).
  - `Giacomini-White (2006)` con HAC.
  - `Clark-West (2007)` para comparaciones anidadas.
  - `Model Confidence Set (MCS)` al nivel $\alpha = 0.10$ con bootstrap estacionario ($B = 500$).
  - Corrección de multiplicidad **Holm-Bonferroni** por familia de escenarios.
  - Reporte de tamaño de efecto ($d$ de Cohen pareado e IC 95% por bootstrap estacionario) y verificación de longitud efectiva $T_{\text{eff}} \ge 2H$.

## 5. Estudios Complementarios Obligatorios (Post-OOF)

1. **Ablación controlada (6 variantes):** `SM-CFHR_completo`, `sin_huber_L2`, `sin_l2_estructurada`, `sin_monotonia`, `escala_std_vs_mad`, `energia_bruta_vs_cf`.
2. **Robustez y sensibilidad:** Curvas 1D de `delta`, `alpha_lin`, `alpha_meteo`; estabilidad en 10 semillas (`2026..2035`); estrés con ruido y 5–10% de outliers; curvas de aprendizaje (`10%, 25%, 50%, 100%`) y escalabilidad (tiempo y memoria).
3. **Incertidumbre predictiva:** Intervalos conformes temporales al 80% y 90% nominales (cobertura global y por planta, ancho medio).
4. **Interpretabilidad:** Pesos estandarizados, importancia global y contribuciones aditivas locales.

## 6. Criterios Explícitos de Éxito / Fracaso

- **R1 (Novedad):** Afirmación acotada al dominio fotovoltaico multi-planta y multi-horizonte respaldada por protocolo WoS y matriz de antecedentes (`CUMPLIDO EN PREPARACIÓN`).
- **R2 (Implementación propia):** Núcleo 100% NumPy/SciPy + compatibilidad completa `BaseEstimator, RegressorMixin, clone, Pipeline` y 20 pruebas unitarias/anti-fuga pasando al 100% (`CUMPLIDO`).
- **R3 (Superioridad empírica):** Se considerará acreditado únicamente si `SM-CFHR` alcanza rango #1 o supervivencia exclusiva/preferente en `MCS` ($\alpha=0.10$) y diferencia estadísticamente significativa tras corrección Holm en la mayoría de las familias de escenarios con $T_{\text{eff}}$ suficiente; en caso contrario, se reportará con total honestidad científica que R3 no se cumple globalmente, detallando los regímenes donde sí aporta ventaja.
