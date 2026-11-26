# 03. Auditoría Estadística Profunda (Sección 5)

**Proyecto:** SM-CFHR (`sm-cfhr v1.0.3`)  
**Alcance:** Evaluación diagnóstica de los resultados OOF definitivos ya calculados (`28` escenarios, `14.625` unidades por variante), sin reentrenar ni modificar archivos del experimento congelado.

---

## 5.A Significancia estadística

### 1. ¿Está implementado correctamente DM-HLN-HAC?
**Respuesta corta:** La fórmula base de Diebold-Mariano (1995) con varianza HAC de Bartlett/Newey-West y corrección de Harvey, Leybourne & Newbold (1997) en `scripts/evaluacion/inferencia_estadistica_v2.py:95-122` es algebraicamente correcta **si y solo si** `horizon_h` representa el horizonte en **pasos de la serie de entrada `d`**. Sin embargo, su **invocación** desde `src/ashr_oof_pipeline/resultados_loader.py:205-206, 236-241` introduce un **error crítico de unidades (P0)**:

- En los escenarios multihorizonte (`7d, 14d, 21d, 28d`), los datos guardados en `resultados/evaluacion_ashr_oof/predicciones/*.csv` están espaciados en **bloques no solapados** cada $H$ días:
  - `D1_*`: $T = 262$ fechas (`2025-11-01..2026-07-31`, paso diario).
  - `7d`: $T = 126$ orígenes (`2024-05-26..2026-10-18`, `diffs_days = {7: 125}`).
  - `14d`: $T = 63$ orígenes (`2024-05-26..2026-10-11`, `diffs_days = {14: 62}`).
  - `21d`: $T = 41$ orígenes (`2024-06-02..2026-09-20`, `diffs_days = {21: 40}`).
  - `28d`: $T = 31$ orígenes (`2024-05-26..2026-09-13`, `diffs_days = {28: 30}`).
- Como dos orígenes consecutivos están separados por $H$ días, los horizontes de pronóstico **no se solapan** entre índices consecutivos de $d_t$. El solapamiento en unidades de la serie $d_t$ es $h_{\text{serie}} = 1$, no $H \in \{7, 14, 21, 28\}$.
- Al pasar `horizon_h = sp.horizon_days`, la línea 119 de `inferencia_estadistica_v2.py` calcula:
  $$\text{HLN}(T, H) = \sqrt{\frac{T + 1 - 2H + H(H-1)/T}{T}}$$
  Para $T = 31, H = 28$, $\text{HLN}(31, 28) = \mathbf{0.1117}$ (deflación artificial de **8.95×** del estadístico DM). Para $T = 41, H = 21$, $\text{HLN}(41, 21) = \mathbf{0.5000}$ (deflación de **2×**).

### 2. Impacto simétrico del error de unidades en `dm_hln_hac` (diagnóstico de solo lectura)
Recalculando únicamente el factor HLN con $h_{\text{serie}} = 1$ (manteniendo intactos `d_t`, `L_hac` y la varianza HAC de `tabla_inferencia.csv`) se observa que el bug ocultó **tanto victorias reales como derrotas reales** de `SM-CFHR-Fixed`:

| Escenario (`Mejor_Bib`) | $T$ | $H$ | $\text{HLN}(T, H)$ | `dm_hln` reportado ($p$) | `dm_hln` con $h_{\text{serie}}=1$ ($p$) | Efecto |
|---|---:|---:|---:|---:|---:|---|
| `RecGEFSv12_espaciotemporal_14d` | 63 | 14 | 0.786 | `-3.257` (`0.0018`) | `-4.112` (`0.0001`) | Victoria más nítida |
| `RecGEFSv12_espaciotemporal_21d` | 41 | 21 | 0.500 | `-1.276` (`0.2094`) | `-2.521` (`0.0158`) | Victoria nominal ocultada |
| `RecGEFSv12_espaciotemporal_28d` | 31 | 28 | 0.112 | `-0.535` (`0.5968`) | `-4.707` (`0.0001`) | Victoria fuerte ocultada (`d_z=-0.756`) |
| `DirectoSinClima_intraplanta_21d` | 43 | 21 | 0.523 | `+1.170` (`0.2487`) | `+2.210` (`0.0326`) | **Derrota nominal ocultada** |
| `DirectoSinClima_intraplanta_28d` | 32 | 28 | 0.140 | `+0.323` (`0.7492`) | `+2.271` (`0.0302`) | **Derrota nominal ocultada** |
| `RecGEFSv12_intraplanta_14d` | 63 | 14 | 0.786 | `+1.705` (`0.0932`) | `+2.153` (`0.0352`) | **Derrota nominal ocultada** |
| `RecSinClima_espaciotemporal_28d` | 31 | 28 | 0.112 | `-0.139` (`0.8903`) | `-1.225` (`0.2302`) | Sigue no significativo |
| `RecSinClima_intraplanta_28d` | 33 | 28 | 0.166 | `+0.131` (`0.8965`) | `+0.778` (`0.4422`) | Sigue no significativo |

*(Nota: los p-valores de la tabla anterior son crudos sin corrección de Holm y sirven únicamente para dimensionar el sesgo de la fórmula HLN mal escalada).*

### 3. Solapamiento temporal, estimador HAC y ancho de banda `L`
- **Solapamiento:** No existe solapamiento de horizontes entre orígenes de `7d/14d/21d/28d` porque los orígenes fueron muestreados exactamente cada $H$ días. Sin embargo, sí puede existir **autocorrelación residual** de orden bajo en $d_t$ por persistencia de regímenes climáticos multidiarios/estacionales (p. ej. ENOS/El Niño-La Niña o nubosidad prolongada) y por actualización de ventanas móviles (`cf30`).
- **Ancho de banda HAC:** `dm_hln_hac` utiliza la regla automática $L = \lfloor 4 (T/100)^{2/9} \rfloor$ (`L = 3..4` bloques para $T \in [31, 134]$; `L = 4..5` días para $T = 262$), con núcleo de Bartlett (`wk = 1 - k/(L+1)`). Para series no solapadas de bloques semanales/mensuales, $L \approx 3\text{–}4$ bloques cubre hasta 1–3 meses de correlación serial residual y es razonable; en contraste, `stationary_bootstrap_ci` (`resultados_loader.py` / `evaluacion_estadistica.py:151`) fija `mean_block = max(7.0, float(horizon_h))`, lo que para $H = 28$ en una serie de $T = 31$ bloques significa bloques de longitud media **28 bloques = 784 días** (casi toda la serie en un solo bloque!), distorsionando la varianza bootstrap en horizontes largos.

### 4. ¿Tiene justificación nuestra restricción `T_eff >= 2H`?
**No en la forma implementada.** En `evaluacion_estadistica.py:156`:
`"t_eff_insufficient": bool(t_eff < 2.0 * horizon_h)`
compara $T_{\text{eff}}$ (medido en **bloques disjuntos de $H$ días**) contra $2H$ (medido en **días**). Para $H = 28$, $T = 31 < 56$: es matemáticamente imposible satisfacer la condición incluso con ruido blanco puro ($T_{\text{eff}} = 31 < 56$). Esta inconsistencia dimensional explica por qué `13/28` escenarios aparecen bloqueados como `T_EFF_INSUFICIENTE`. Una regla coherente exige $T_{\text{eff}} \ge 25\text{–}30$ **unidades temporales de la serie evaluada** (o convertir ambos lados a días: $T_{\text{eff}} \cdot H \ge 2H$, es decir, $T_{\text{eff}} \ge 2$ bloques, lo cual sería demasiado laxo; por tanto el criterio sano en estadística asintótica es $T_{\text{eff,unidades}} \ge 30$).

### 5. ¿Estamos perdiendo potencia por decisiones excesivamente conservadoras?
**Sí, por cuatro causas simultáneas:**
1. **Error de unidades en HLN (`horizon_h = H`)**: reduce `dm_hln` hasta en 9× en $H=28$.
2. **Error de unidades en `T_eff < 2H`**: elimina 13 escenarios del conteo válido.
3. **Holm global sobre 56 filas** (`resultados_loader.py:274`) en lugar de Holm por familia pre-especificada (`protocolo.md:42`), multiplicando la penalización de pruebas que mezclan `Mejor_Bib` y `MM30` e incluyen filas marcadas como insuficientes.
4. **Muestreo espaciado cada $H$ días** en el protocolo OOF: en lugar de evaluar orígenes diarios solapados ($N \approx 800$ orígenes con $h = H$ y varianza HAC/Newey-West o *moving-block bootstrap* como en Linca et al., 2026, p. 19), se redujo la muestra a $T = 31$ bloques disjuntos.

### 6. Interpretación correcta de resultados no significativos
Según Hewamalage et al. (2023, p. 826) y Linca et al. (2026, p. 23), **no rechazar $H_0$ bajo $T$ pequeño ($31\text{–}63$ bloques) y corrección de Holm no equivale a demostrar igualdad ni inutilidad del modelo**; refleja falta de evidencia estadística suficiente para descartar el azar en esa muestra. Como señalan Linca et al. (2026, p. 23) al reportar $11/12$ victorias mensuales de LightGBM/XGBoost sin significancia tras Holm ($p_{\text{Holm}}=0.082$ y $0.102$), los resultados deben presentarse con transparencia distinguiendo superioridad descriptiva observada de significancia confirmatoria.

---

## 5.B Superioridad predictiva frente a modelos de biblioteca

### 1. Qué podemos afirmar hoy con respaldo sólido
- **Superioridad condicionada al régimen `Espaciotemporal + Clima` (`D1_espaciotemporal_cinco_variables` y `RecGEFSv12_espaciotemporal_*`):**
  - En `D1_espaciotemporal_cinco_variables`: `SM-CFHR-Fixed` vence claramente a `Lasso` (`mean_diff = -16,920.8 kWh/día`, $p_{\text{Holm}} = 6.39 \times 10^{-22}$, `cohen_dz = -0.898`, sobreviviente único `1/23` en MCS).
  - En `RecGEFSv12_espaciotemporal_{7,14,21,28}d`: vence descriptivamente en los 4 horizontes con `d_z` entre `-0.47` y `-0.82` y probabilidad de victoria por origen entre `83.9%` y `93.7%` (`tabla_inferencia.csv:42,46,50,54`). La falta de significancia en `21d` y `28d` proviene del error de unidades en HLN y `T_eff < 2H`, no de ausencia de señal.
- **Lo que NO podemos afirmar:**
  - **No hay superioridad global ni uniforme en los 28 escenarios.** En el régimen **Intraplanta (`IP`)**, `SM-CFHR-Fixed` no supera sistemáticamente a modelos lineales regularizados (`BayesianRidge`, `Ridge`) ni a `k-NN`/`SVR`, y presenta un **fracaso sistemático en `DirectoSinClima_intraplanta`** (especialmente en `14d, 21d, 28d` donde `k-NN` obtiene menor error con diferencias positivas de `+56k` a `+288k kWh_total`).
  - **En `RecSinClima`**, la ventaja en `espaciotemporal` (`66–94%` prob. victoria en algunos horizontes) es pequeña respecto a la variabilidad (`d_z` entre `-0.09` y `-0.18`), no significativa tras HAC.

### 2. Defectos en las herramientas actuales de comparación múltiple
- **Selección retrospectiva (`ex post`) de `Mejor_Bib`:** Elegir al ganador entre 9 bibliotecas en la misma muestra OOF y correr DM de 2 muestras contra él viola la distinción *ex ante* vs. *ex post* (Tashman, 2000, pp. 446–447) y penaliza injustamente al modelo propio si no se ajusta por búsqueda sobre 9 competidores.
- **Bug en `MCS` (`evaluacion_estadistica.py:118`):** Compara $\max(t)$ contra $\max(|t^*|)$ (bilateral) con $B=200$, lo que explica que en `tabla_mcs.csv` sobrevivan **15 a 22 de 23 modelos** en los escenarios multihorizonte, anulando la capacidad discriminativa del MCS.
- **Bug en `SPA` (`src/sm_cfhr/inference.py:117-122`):** Ignora el `se` bootstrap dependiente calculado en L117 y usa `std(d)` i.i.d. en L118 y L122.
- **Rankings medios:** Útiles como resumen descriptivo (Friedman + Nemenyi/Holm, Hewamalage et al., 2023, pp. 824–825), pero deben reportarse **separados por régimen (`ET` vs. `IP`, con/sin clima, recursivo vs. directo)** y nunca promediando `kWh` y `kWh_total` juntos ni tratando los 28 escenarios como bloques independientes.

---

## 5.C Dependencia espacial y temporal

1. **Errores temporalmente correlacionados:** En $d_t$ por origen, la correlación residual proviene de persistencia climática (nubosidad/ENOS) y en `D+1` de autocorrelación diaria (`diffs=1`). Requiere varianza HAC o *block bootstrap* con longitud de bloque acorde a la unidad de `time_col` (no `max(7, H)` bloques sobre bloques de 28 días).
2. **Dependencia meteorológica entre plantas y agrupación geográfica:** Las 17 plantas fotovoltaicas de Colombia comparten eventos sinópticos y nubosidad regional (especialmente plantas en la misma zona). Promediar las 17 plantas en un único escalar por fecha $d_t = \frac{1}{P_t}\sum_{p=1}^{P_t} d_{t,p}$ absorbe la correlación contemporánea entre plantas **para inferir sobre el promedio de cartera**, pero:
   - **Problema P1 (panel desbalanceado):** En `Rec*` y `Directo*`, el número de plantas por origen varía de `1` a `17`, y en `D+1` de `14` a `17`. Un origen temprano con 1 planta recibe el mismo peso en $d_t$ que un origen tardío con 17 plantas, rompiendo la equivalencia exacta con `MAE_macro` global y creando heteroscedasticidad en $d_t$.
   - **Problema P1 (escala en kWh):** Plantas de mayor MW dominan tanto `MAE_macro` como $d_t$ (Tashman, 2000, p. 445; Hewamalage et al., 2023, p. 802). Evaluar sobre **Factor de Capacidad (`CF`, Linca et al., 2026, p. 7)** o error relativo por planta estabiliza la escala.
3. **Escenarios que comparten observaciones:** Los 28 escenarios **no son independientes**: comparten las mismas 17 plantas, las mismas ventanas calendario (`2024-05..2026-10`), y `7d/14d/21d/28d` evalúan subconjuntos superpuestos del mismo horizonte extendido. Por ello, **no es válido** aplicar pruebas como Friedman o conteos de victorias tratando `N = 28` como muestras independientes (Oliveira et al., 2021, p. 19, separan Friedman por modelo). Las conclusiones deben agruparse por **familia de régimen** (4 familias: `D+1`, `RecSinClima`, `DirectoSinClima`, `RecGEFSv12` × `ET`/`IP`).
4. **Horizontes recursivos vs. directos:** Hewamalage et al. (2023, pp. 800–801) advierten explícitamente que los esquemas directos y recursivos tienen dinámicas de acumulación de error distintas y no deben mezclarse ni atribuirse a la arquitectura base sin controlar la estrategia multi-paso.

---

## 5.D Evaluación temporal nueva (holdout posterior)

1. **¿Tiene sentido ahora mismo?**
   - **A corto plazo (hoy `2026-11-25`): NO es estadísticamente suficiente.** Los datos actuales terminan el `2026-07-31` (`D+1`) y el `2026-10-18` (`Rec*`). Solo han transcurrido ~`1 a 3.8` meses posteriores: en `H=28` eso produciría apenas `1–4` bloques disjuntos, y en `D+1` omitiría nuevamente gran parte del ciclo anual.
   - **A mediano plazo (cuando se acumulen $\ge 12$ meses nuevos completos, p. ej. `2026-11` a `2027-10`):** **SÍ tiene alto valor científico** como prueba confirmatoria verdaderamente prospectiva (*real-time out-of-sample*, Tashman, 2000, p. 438).
2. **Especificación técnica para un holdout prospectivo válido (sin descargar ni entrenar ahora):**
   - **Datos adicionales necesarios:** Generación diaria real de XM para las mismas 17 plantas (y idealmente nuevas plantas colombianas fuera de la muestra para generalización espacial pura) + archivo histórico de pronósticos meteorológicos GEFSv12 / Open-Meteo **emitidos antes de cada origen** (como exige Linca et al., 2026, p. 8; **nunca** usar reanálisis ERA5 como entrada operacional).
   - **Periodo válido:** Mínimo **12 meses continuos** posteriores al último origen de entrenamiento/validación (`>= 2026-11-01` hasta `>= 2027-10-31`) para cubrir un ciclo anual completo de hidrología/nubosidad colombiana y aportar $\ge 13$ bloques de 28d (o $\ge 365$ orígenes diarios).
   - **Prevención de fuga temporal:** Mantener *gap* de separación igual al horizonte máximo ($28$ días) entre el fin de la historia usada para calibrar/ajustar y el primer origen del holdout; calcular todas las normalizaciones/estadísticas climatológicas usando exclusivamente datos previos al origen.
   - **Modelo congelado:** Utilizar **`SM-CFHR-Fixed`** (con parámetros congelados de `experiments/config_ashr_congelada.json`, SHA-256 verificado) frente a los benchmarks congelados y la referencia climatología–persistencia, sin re-ejecutar HPO sobre el holdout.
   - **Métricas y pruebas:** `nMAE` y `nRMSE` en `CF`, `RMSE skill score` (Yang et al., 2020), factorizaciones Murphy–Winkler, DM-HLN-HAC con $h_{\text{serie}}$ correcto, MCS unilateral (`B=2000`) y SPA corregido, separados por `ET` e `IP`.
   - **Conclusiones que permitiría:** Validar la estabilidad temporal ante deriva climática real (*concept drift*), cerrar la brecha estacional actual de `D+1` (que hoy solo cubre 9 meses) y confirmar si la ventaja en `ET + Clima` se sostiene fuera del periodo `2024–2026`.
