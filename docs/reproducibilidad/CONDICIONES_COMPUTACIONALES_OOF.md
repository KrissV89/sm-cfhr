# Condiciones Computacionales y Reproducibilidad — Evaluación OOF Definitiva (`SM-CFHR`)

Este documento registra las condiciones reales de ejecución, entorno computacional, trazabilidad criptográfica y telemetría observada de la evaluación fuera de muestra (*Out-of-Fold*, OOF) definitiva del repositorio oficial **`sm-cfhr`** bajo licencia **PolyForm Noncommercial 1.0.0**.

> **Distinción metodológica:** se separan explícitamente las **condiciones nominales** (hardware, software, hashes y protocolo congelado), las **mediciones observadas** (benchmark preliminar de escalado, Fase A completada y telemetría en vivo de Fase B) y los campos **`[PENDIENTE DE CIERRE DE FASE B]`** que solo pueden cerrarse cuando concluya y consolide `oof_v1_0_3_hpo`. No se reportan estimaciones como resultados finales.

---

## 1. Condiciones Nominales (Entorno, Código y Protocolo Congelado)

### 1.1 Hardware y Sistema Operativo
| Componente | Especificación exacta |
| :--- | :--- |
| **CPU** | `AMD Ryzen 7 8845HS w/ Radeon 780M Graphics` |
| **Núcleos / Hilos** | `8` núcleos físicos / `16` procesadores lógicos (SMT) |
| **RAM instalada / utilizable** | `16 GB` instalados / `13.81 GB` utilizables por el SO (~`2 GB` reservados por iGPU Radeon 780M) |
| **Espacio libre en disco (`C:\`)** | `121.49 GB` libres al momento del *preflight* (`doctor`) |
| **Sistema operativo** | `Windows-10-10.0.26200-SP0` (Windows 11 build `26200`) |
| **Estrategia multiprocessing** | `spawn` (nativa de Windows) |

### 1.2 Entorno Python y Dependencias Científicas (`solarpv-eda`)
- **Intérprete:** `python (entorno solarpv-eda)` — **Python `3.11.15`**
- **Dependencias principales:**
  - `numpy`: `2.3.5`
  - `scipy`: `1.16.3`
  - `pandas`: `2.3.3`
  - `scikit-learn`: `1.9.0`
  - `statsmodels`: `0.14.6`
  - `optuna`: `5.0.0`
  - `sqlalchemy`: `2.1.2`
  - `pyarrow`: `25.0.1`
  - `psutil`: `7.2.2`
  - `pytest`: `9.1.1`

### 1.3 Identificadores de Versión, Repositorio y Trazabilidad Criptográfica
- **Paquete Python (`pyproject.toml`):** `sm-cfhr 1.0.3`
- **Licencia:** `PolyForm-Noncommercial-1.0.0` (etiqueta Git `v1.0.3-polyform`)
- **Commit exacto al iniciar la corrida OOF:** `3256994` en rama `main` (`https://github.com/KrissV89/sm-cfhr.git`), árbol limpio respecto al protocolo congelado
- **`sha256_src_adaptive_structured_huber` (`src/adaptive_structured_huber.py`):**
  `27a9fda4fa8b7ef0d95faf9e925e47ade9ed8f25e9ea5d97d6d7cf1c00ddde83`
- **`config_hash` (`experiments/config_ashr_congelada.json`):**
  `ab805d56a958d58d5ae3e5a11f7e82af4f9af7c0cacfe9bfb83d138b91f0c64b`
- **Protocolo interno HPO:** `HPO_PROTOCOL_VERSION = "v1.0.3-exact-recursive-state"`
- **`hpo_protocol_hash`:**
  `40e1bfc5105b82ad6b075f81a0cd6b60d28f485e90e1327aedc146e8a2345a1e`

### 1.4 Configuración Científica y Política de Ejecución Paralela
- **Alcance experimental:** `28` escenarios oficiales, `14,625` unidades experimentales por variante (`206,073` filas de predicción consolidadas por variante), semilla global `2026`.
- **Validación causal anidada:** embargo temporal de `30` días en `D+1`, restricción $t_{\mathrm{end}} \le t_{\mathrm{val}}$ en multihorizonte y *leave-one-group-out* (LOGO) espacial en escenarios ET.
- **Variante `SM-CFHR-Fixed` (`oof_v1_0_3_fixed`):** $\delta = 1.35$, $\alpha_{\mathrm{lin}} = 0.01$, $\alpha_{\mathrm{meteo}} = 0.10$, presupuesto de búsqueda $B = 0$, ejecutada con **`8` workers**.
- **Variante `SM-CFHR-HPO` (`oof_v1_0_3_hpo`):** `25` ensayos Optuna TPE ($B = 25$) por unidad sobre `3` particiones internas, base SQLite aislada por unidad (`RDBStorage`), ejecutada con **`12` workers**.
- **Control de hilos BLAS/OpenMP por worker:** `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, `LOKY_MAX_CPU_COUNT=1` para evitar sobresuscripción multi-hilo.
- **Política de reciclaje y persistencia:** reciclaje preventivo cada **`max_tasks_per_worker = 60`** unidades o **`worker_max_rss_mb = 1536` MB**; persistencia inmediata y atómica (`fsync` + `os.replace`) organizada por horizonte (`D1/`, `7d/`, `14d/`, `21d/`, `28d/`).

---

## 2. Mediciones Observadas

### 2.1 Benchmark Preliminar de Escalado SMT (48 Unidades de Desarrollo)
Antes de lanzar Fase B (`oof_v1_0_3_hpo`), se ejecutó un benchmark controlado (`scripts/benchmark_smt_8_10_12.py`, artefactos en `results/benchmark_paralelo/benchmark_smt_8_10_12.{csv,json}`) sobre **`48` unidades de desarrollo** (`D1`–`D3`) con **`25` ensayos Optuna completos** por unidad. *Esta prueba mide escalado relativo en un subconjunto acotado de desarrollo y no representa la duración del OOF definitivo completo.*

| Workers | Duración (s) | Rendimiento (u/min) | Ganancia vs. 8w | CPU media (%) | Frecuencia (MHz) | Temp. (°C) | RSS agregado pico (MB) | Errores | Hashes SHA-256 = 8w |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **8** | `70.064` | `41.106` | `0.00%` | `60.7%` | `3040` | `53.1` | `2,375.64` | `0` | Referencia |
| **10** | `65.841` | `43.741` | `+6.41%` | `72.2%` | `3040` | `53.1` | `2,926.66` | `0` | `true` |
| **12** | `63.360` | `45.455` | **`+10.58%`** | `83.8%` | `3040` | `53.1` | `3,465.09` | `0` | `true` |

Criterio de selección aplicado: `12` workers superaron el umbral preespecificado de `>= +10%` de mejora sostenida respecto a `8` workers (`+10.58%`) sin throttling térmico (`53.1 °C`), con holgura de memoria (`3.47 GB` frente a `13.81 GB` utilizables) y equivalencia bit a bit verificada (`hashes_match_8w = true`).

### 2.2 Fase A Definitiva — `SM-CFHR-Fixed` (`oof_v1_0_3_fixed`) — COMPLETADA Y CONSOLIDADA
- **Estado:** `OOF_DEFINITIVO_COMPLETO` (`procedencia: OOF_DEFINITIVO_REAL`)
- **Inicio:** `2026-11-24T12:38:44` | **Fin ejecución:** `2026-11-24T12:47:04` | **Consolidación:** `2026-11-24T12:50:07`
- **Cobertura:** `14,625 / 14,625` unidades (`28 / 28` escenarios completos, `206,073` filas de predicción, `0` fallidas, `0` faltantes, `0` inválidas)
- **Rendimiento de pared:** `499.748 s` (`8.33 min`), `29.2647` unidades/s (`1,755.88` unidades/min) con `8` workers
- **Reciclaje y memoria:** `241` reciclajes de workers (`249` spawns totales, overhead de spawn `1.579 s`), RSS pico coordinador `206.73 MB`, RSS pico worker `241.82 MB`, RSS agregado pico **`2,067.62 MB`**
- **Errores, interrupciones y recuperaciones:** `0` errores, `0` interrupciones, `0` recuperaciones requeridas

#### Tiempos Observados por Escenario (`tiempos_por_escenario_s` en `oof_v1_0_3_fixed`)
*(Tiempo acumulado de worker por escenario extraído de `results/oof_v1_0_0/oof_v1_0_3_fixed/consolidated/manifest_ashr_oof.json`; al ejecutarse en paralelo entre `8` workers, la suma de tiempos por escenario supera el tiempo total de pared de `499.748 s`.)*

| Escenario | Tiempo acumulado (s) | Escenario | Tiempo acumulado (s) |
| :--- | :---: | :--- | :---: |
| `fixed__D1_espaciotemporal_cinco_variables` | `11.478` | `fixed__RecGEFSv12_espaciotemporal_21d` | `31.967` |
| `fixed__D1_espaciotemporal_sin_meteorologia` | `6.796` | `fixed__RecGEFSv12_espaciotemporal_28d` | `28.738` |
| `fixed__D1_intraplanta_cinco_variables` | `7.967` | `fixed__RecGEFSv12_espaciotemporal_7d` | `54.702` |
| `fixed__D1_intraplanta_sin_meteorologia` | `7.664` | `fixed__RecGEFSv12_intraplanta_14d` | `81.737` |
| `fixed__DirectoSinClima_espaciotemporal_14d` | `21.704` | `fixed__RecGEFSv12_intraplanta_21d` | `48.069` |
| `fixed__DirectoSinClima_espaciotemporal_21d` | `12.006` | `fixed__RecGEFSv12_intraplanta_28d` | `42.076` |
| `fixed__DirectoSinClima_espaciotemporal_28d` | `9.577` | `fixed__RecGEFSv12_intraplanta_7d` | `103.734` |
| `fixed__DirectoSinClima_espaciotemporal_7d` | `22.691` | `fixed__RecSinClima_espaciotemporal_14d` | `25.766` |
| `fixed__DirectoSinClima_intraplanta_14d` | `36.369` | `fixed__RecSinClima_espaciotemporal_21d` | `19.914` |
| `fixed__DirectoSinClima_intraplanta_21d` | `18.439` | `fixed__RecSinClima_espaciotemporal_28d` | `16.719` |
| `fixed__DirectoSinClima_intraplanta_28d` | `14.934` | `fixed__RecSinClima_espaciotemporal_7d` | `38.075` |
| `fixed__DirectoSinClima_intraplanta_7d` | `56.290` | `fixed__RecSinClima_intraplanta_14d` | `40.303` |
| `fixed__RecGEFSv12_espaciotemporal_14d` | `57.698` | `fixed__RecSinClima_intraplanta_21d` | `30.065` |
| | | `fixed__RecSinClima_intraplanta_28d` | `26.299` |
| | | `fixed__RecSinClima_intraplanta_7d` | `69.642` |

### 2.3 Fase B Definitiva — `SM-CFHR-HPO` (`oof_v1_0_3_hpo`) — COMPLETADA Y CONSOLIDADA
- **Estado:** `OOF_DEFINITIVO_COMPLETO` (`procedencia: OOF_DEFINITIVO_REAL`)
- **Cobertura:** `14,625 / 14,625` unidades (`28 / 28` escenarios completos, `206,073` filas de predicción, `0` fallidas, `0` faltantes, `0` inválidas)
- **Configuración:** `12` workers, `n_trials = 25`, `max_tasks_per_worker = 60`, `worker_max_rss_mb = 1536`
- **Sesión 1 (`PID=6024`, `2026-11-24T12:56:26` – `16:50:01`):** `14,015 s` (`3.89 h`), `7,689` unidades completadas, `120` reciclajes, RSS pico coordinador `206.89 MB`, worker `357.05 MB`, agregado **`4,201.10 MB`**. Detenida por traslado del portátil; auditoría SHA-256 posterior confirmó `0` artefactos corruptos y `7,669 / 7,669` bases SQLite íntegras (`PRAGMA quick_check = ok`).
- **Sesión 2 reanudada (`PID=38056`, fin `2026-11-24T23:34:10`, consolidación `23:44:01`):** `15,535.293 s` (`4.32 h`), `6,936` unidades ejecutadas (`0.4465 u/s`), `108` reciclajes (`120` spawns, overhead `1.253 s`), `0` errores, RSS pico coordinador `205.18 MB`, worker `354.29 MB`, agregado `4,155.84 MB`.
- **Total acumulado de Fase B:** **`29,550.293 s`** (`8.21 h` de cómputo activo en dos sesiones), `0.4949` unidades/s (`29.69` unidades/min), **`228` reciclajes** totales, RSS agregado pico global **`4,201.10 MB`**, `0` errores finales.

#### Tiempos Observados por Escenario (`tiempos_por_escenario_s` en `oof_v1_0_3_hpo`)
*(Tiempo acumulado de carga + cómputo en workers por escenario extraído de `results/oof_v1_0_0/oof_v1_0_3_hpo/consolidated/manifest_ashr_oof.json`.)*

| Escenario | Tiempo acumulado (s) | Escenario | Tiempo acumulado (s) |
| :--- | :---: | :--- | :---: |
| `hpo__D1_espaciotemporal_cinco_variables` | `135.185` | `hpo__RecGEFSv12_espaciotemporal_21d` | `10,153.754` |
| `hpo__D1_espaciotemporal_sin_meteorologia` | `103.593` | `hpo__RecGEFSv12_espaciotemporal_28d` | `9,470.501` |
| `hpo__D1_intraplanta_cinco_variables` | `263.831` | `hpo__RecGEFSv12_espaciotemporal_7d` | `22,374.830` |
| `hpo__D1_intraplanta_sin_meteorologia` | `209.579` | `hpo__RecGEFSv12_intraplanta_14d` | `20,948.947` |
| `hpo__DirectoSinClima_espaciotemporal_14d` | `6,553.636` | `hpo__RecGEFSv12_intraplanta_21d` | `14,020.345` |
| `hpo__DirectoSinClima_espaciotemporal_21d` | `4,213.231` | `hpo__RecGEFSv12_intraplanta_28d` | `13,004.028` |
| `hpo__DirectoSinClima_espaciotemporal_28d` | `2,993.726` | `hpo__RecGEFSv12_intraplanta_7d` | `37,009.496` |
| `hpo__DirectoSinClima_espaciotemporal_7d` | `13,209.329` | `hpo__RecSinClima_espaciotemporal_14d` | `9,975.312` |
| `hpo__DirectoSinClima_intraplanta_14d` | `11,247.972` | `hpo__RecSinClima_espaciotemporal_21d` | `7,117.162` |
| `hpo__DirectoSinClima_intraplanta_21d` | `8,635.372` | `hpo__RecSinClima_espaciotemporal_28d` | `8,004.312` |
| `hpo__DirectoSinClima_intraplanta_28d` | `6,407.377` | `hpo__RecSinClima_espaciotemporal_7d` | `17,483.858` |
| `hpo__DirectoSinClima_intraplanta_7d` | `31,957.133` | `hpo__RecSinClima_intraplanta_14d` | `18,654.436` |
| `hpo__RecGEFSv12_espaciotemporal_14d` | `15,785.789` | `hpo__RecSinClima_intraplanta_21d` | `13,785.848` |
| | | `hpo__RecSinClima_intraplanta_28d` | `11,265.496` |
| | | `hpo__RecSinClima_intraplanta_7d` | `33,944.541` |

---

## 3. Comandos Reproducibles y Criterios de Verificación

```powershell
# 1. Verificación previa de entorno, datos y hashes congelados
python -m sm_cfhr.cli doctor

# 2. Fase A definitiva (SM-CFHR-Fixed, 8 workers) y consolidación
python scripts/ejecutar_oof_paralelo.py --workers 8 --variante fixed --run-id oof_v1_0_3_fixed --confirm-oof
python scripts/ejecutar_oof_paralelo.py --consolidate --run-id oof_v1_0_3_fixed

# 3. Benchmark preliminar de escalado SMT (48 unidades de desarrollo x 25 trials)
python scripts/benchmark_smt_8_10_12.py

# 4. Fase B definitiva (SM-CFHR-HPO, 12 workers, 25 trials) y consolidación
python scripts/ejecutar_oof_paralelo.py --workers 12 --variante hpo --n-trials 25 --max-tasks-per-worker 60 --worker-max-rss-mb 1536 --run-id oof_v1_0_3_hpo --confirm-oof
python scripts/ejecutar_oof_paralelo.py --consolidate --run-id oof_v1_0_3_hpo

# 5. Monitoreo no intrusivo en vivo y reanudación idempotente ante interrupción
python scripts/ejecutar_oof_paralelo.py --status --run-id oof_v1_0_3_hpo
```

### Gobernanza durante la ejecución activa
Mientras `oof_v1_0_3_hpo` permanezca en ejecución, no se modifican archivos fuente en `src/` ni se realizan operaciones destructivas de Git. La actualización definitiva de los marcadores `[PENDIENTE DE CIERRE DE FASE B]` y el versionado final en Git se realizarán tras concluir y consolidar Fase B.
