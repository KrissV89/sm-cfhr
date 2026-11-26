# Estimación Exacta de Coste Computacional OOF (28 Escenarios)

**Fuente exacta:** Conteo directo de bloques, orígenes y unidades sobre los 28 CSVs oficiales (`resultados/preflight_ashr/estimacion_coste_28_escenarios.csv`).  
**Tiempos unitarios medidos en `D1_ET`:** `0.044 s/fit` (`Fixed`), `0.024 s/fit` (`HPO` en `B=25`: `3.652 s / 152 fits`).

---

## 1. Totales Exactos para los 28 Escenarios

| Familia (Escenarios) | Fits `SM-CFHR-Fixed` | Trials `SM-CFHR-HPO (B=25)` | Fits totales `HPO` (3 internos + refit) | Tiempo est. `Fixed` | Tiempo est. `HPO (B=25)` |
|---|---:|---:|---:|---:|---:|
| `D+1` (4 escenarios) | 144 | 3,600 | 10,944 | 6.3 s | 4.4 min |
| `Rec sin clima` (8 escenarios) | 6,335 | 158,375 | 481,460 | 278.7 s | 3.21 h |
| `Directo sin clima` (8 escenarios) | 6,138 | 153,450 | 466,488 | 270.1 s | 3.11 h |
| `Rec + GEFSv12` (8 escenarios) | 6,880 | 172,000 | 522,880 | 302.7 s | 3.49 h |
| **TOTAL (28 escenarios)** | **19,497** | **487,425** | **1,481,772** | **~857.9 s (~14.3 min)** | **~35,603 s (~9.9 h)** |

## 2. Memoria, Disco y Política de Reanudación

- **Memoria pico medida (`tracemalloc`):** **5.18 MB** (`Fixed`), **2.59 MB** por trial TPE (`HPO`).
- **Almacenamiento estimado:** `< 15 MB` para predicciones `Fixed` + tablas + figuras; `< 120 MB` incluyendo logs de `487,425` trials si se ejecuta `HPO`.
- **Seguridad y reanudación:** `run_oof_definitivo` usa escritura atómica (`*.tmp` + `os.replace`) y `checkpoint_parcial.json` por escenario, permitiendo reanudar corridas largas de `HPO` sin perder progreso ni marcar resultados incompletos como `OOF_DEFINITIVO_COMPLETO`.
