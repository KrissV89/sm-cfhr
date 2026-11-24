# Structured Monotone Capacity-Factor Huber Regressor (SM-CFHR)

> **A robust, physically constrained regression framework for multi-plant and multi-horizon photovoltaic energy forecasting, with reproducible nested cross-validation and statistical evaluation.**

---

## 1. Nombre completo
**SM-CFHR** (*Structured Monotone Capacity-Factor Huber Regressor*, implementado en `AdaptiveStructuredHuberRegressor` y expuesto mediante la API `scikit-learn` `SMCFHRRegressor`).

## 2. Problema abordado
Pronóstico diario (`D+1`) y multihorizonte (`7, 14, 21, 28` días) del factor de capacidad y la generación eléctrica fotovoltaica en flotas reales heterogéneas (17 plantas solares en Colombia), con dos regímenes de generalización: **espaciotemporal (ET)** e **intraplanta (IP)**, con y sin meteorología numérica (`Open-Meteo` / `GEFSv12`).

## 3. Contribución metodológica
Formulación paramétrica convexa acotada al dominio fotovoltaico multi-planta que unifica:
1. Transformación física al **factor de capacidad adimensional** $c_i \in [0, 1]$ (`B_i = cap_MW_D * 24,000`).
2. **Pérdida simétrica de Huber** sobre escala robusta MAD residual, resistente a desconexiones y anomalías operativas.
3. **Regularización $L_2$ diferenciada** por familia de predictores ($\alpha_{\text{lin}}$ vs. $\alpha_{\text{meteo}}$).
4. **Restricciones físicas monótonas** ($\theta_j \ge 0$ sobre `cf30` y `rad_pron`) resueltas mediante **L-BFGS-B con gradiente analítico** en NumPy/SciPy puro.

## 4. Formulación resumida
$$\min_{\theta_0 \in \mathbb{R},\; \theta \in \mathcal{C}} \; \frac{1}{n}\sum_{i=1}^n \rho_\delta\!\left(\tilde{c}_i - \theta_0 - z_i^\top \theta\right) + \frac{1}{2}\sum_{j \in \mathcal{G}_{\mathrm{lin}}} \alpha_{\mathrm{lin}}\,\theta_j^2 + \frac{1}{2}\sum_{j \in \mathcal{G}_{\mathrm{meteo}}} \alpha_{\mathrm{meteo}}\,\theta_j^2$$

## 5. Arquitectura del repositorio
- `src/sm_cfhr/`: paquete Python instalable (`estimator.py`, `cli.py`, `persistence.py`, `pipeline_stages.py`, `inference.py`, `validation.py`, `optimization.py`, `robustness.py`, `plotting.py`, `io.py`) y subpaquete `parallel/` (`work_units.py`, `storage.py`, `worker.py`, `coordinator.py`, `consolidation.py`) para ejecución multiproceso por horizontes.
- `src/ashr_oof_pipeline/`: núcleo OOF de 28 escenarios, validación cruzada causal anidada y cargador de resultados.
- `scripts/` y `scripts/evaluacion/`, `scripts/setup/`: runner secuencial (`ejecutar_oof_definitivo_ashr.py`), coordinador paralelo (`ejecutar_oof_paralelo.py`), benchmarks y validación (`benchmark_parallel_workers.py`, `validar_infraestructura_paralela.py`), postprocesamiento en 8 etapas e importador de datos locales.
- `experiments/`: configuración congelada (`config_ashr_congelada.json`) y protocolo pre-registrado (`protocolo.md`).
- `notebooks/actuales/`: notebook científico definitivo en 3 estados (A/B/C).
- `book/` y `dashboard/`: informe Jupyter Book y aplicación Dash multipágina.
- `tests/`: 8 suites automatizadas (pruebas matemáticas, anti-fuga, bloqueadores v1.1, recuperación A–L y equivalencia/reanudación multiproceso en `test_parallel_oof.py`).

## 6. Instalación (Clonado Independiente)
```bash
git clone https://github.com/KrissV89/sm-cfhr.git
cd sm-cfhr
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS:       source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-build-isolation --no-deps
```

## 7. Obtención de datos
Los datasets solares reales y los 28 CSVs oficiales de benchmarks (`654.35 MB`) **no se alojan en GitHub** por términos de redistribución XM y tamaño. Consulte [data/README.md](data/README.md) y [data/manifests/data_manifest.json](data/manifests/data_manifest.json). Una vez recibida la carpeta autorizada de datos locales:
```bash
python scripts/setup/import_local_data.py --source-root /ruta/a/datos_autorizados
```

## 8. Preparación de entorno y verificación previa (`doctor`)
```bash
python -m sm_cfhr.cli doctor
python -m sm_cfhr.cli evaluate --variante ambos --n-trials 25
```

## 9. Verificación de pruebas unitarias
```bash
python -m pytest tests/test_adaptive_structured_huber.py tests/test_selector_ashr_v2.py tests/test_bloqueadores_v1_1.py -v
```

## 10. Ejecución completa del OOF definitivo

### Ejecución paralela local recomendada (Windows `spawn` / multi-núcleo)
El coordinador paralelo (`scripts/ejecutar_oof_paralelo.py`) distribuye unidades independientes entre workers reciclables fijando `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` y `MKL_NUM_THREADS=1` para evitar sobresuscripción, persiste cada unidad atómicamente por horizonte (`D1/`, `7d/`, `14d/`, `21d/`, `28d/`) y aísla cada estudio Optuna en su propio archivo SQLite:
```bash
# 1. Dry-run previo (no entrena):
python scripts/ejecutar_oof_paralelo.py --workers 8 --variante ambos --data-root /ruta/a/datos_autorizados

# 2. Ejecución definitiva (Fixed y HPO con 25 trials, reciclaje cada 60 unidades o 1536 MB RSS):
python scripts/ejecutar_oof_paralelo.py --workers 8 --variante fixed --run-id oof_v1_0_3_fixed --data-root /ruta/a/datos_autorizados --confirm-oof
python scripts/ejecutar_oof_paralelo.py --workers 8 --variante hpo --n-trials 25 --max-tasks-per-worker 60 --worker-max-rss-mb 1536 --run-id oof_v1_0_3_hpo --data-root /ruta/a/datos_autorizados --confirm-oof
```
*(Alternativa secuencial original vía CLI: `python -m sm_cfhr.cli evaluate --confirm-oof --variante ambos --n-trials 25 --run-id oof_definitivo_v1_1`.)*

## 11. Seguimiento, recuperación y consolidación tras interrupción
```bash
# Consultar progreso en vivo desde otra terminal sin detener la corrida:
python scripts/ejecutar_oof_paralelo.py --status --run-id oof_v1_0_3_hpo

# Reanudar tras Ctrl+C (recupera locks huérfanos y continúa estudios Optuna sin trials extra):
python scripts/ejecutar_oof_paralelo.py --workers 8 --variante hpo --n-trials 25 --max-tasks-per-worker 60 --worker-max-rss-mb 1536 --run-id oof_v1_0_3_hpo --data-root /ruta/a/datos_autorizados --confirm-oof

# Consolidar fragmentos en predicciones/*.csv y manifest_ashr_oof.json para el notebook final:
python scripts/ejecutar_oof_paralelo.py --consolidate --run-id oof_v1_0_3_fixed
python scripts/ejecutar_oof_paralelo.py --consolidate --run-id oof_v1_0_3_hpo
```

## 12. Resultados producidos (Etapas 2–8 sin reentrenar)
```bash
python -m sm_cfhr.cli report
```
Genera métricas, rankings, MCS, DM-HLN-HAC, SPA, multiplicidad Holm/BH por familias, veredicto R3, estudios de ablación/robustez/incertidumbre/interpretabilidad, tablas CSV/LaTeX y 13 figuras (`PNG` + `PDF`) en `resultados/evaluacion_ashr_oof/`.

## 13. Notebook científico
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/actuales/evaluacion_cientifica_ashr_oof.ipynb
```

## 14. Jupyter Book
Ubicado en `book/` (requiere `jupyter-book >= 2.0`). Sus capítulos de Entregable 2 documentan `K1/W1` y están preparados para consumir las tablas definitivas de `SM-CFHR` tras el OOF.

## 15. Dashboard
Ubicado en `dashboard/` (`app.py`, `pages/`, `render.yaml`). Carga artefactos precalculados sin entrenar en solicitudes web; su despliegue requiere instalar las dependencias web de `dash` (`render.yaml`).

## 16. Reproducibilidad, Versiones y Validación Independiente
- **Versión del paquete y licencia:** `sm-cfhr 1.0.3` (`pyproject.toml`, tag `v1.0.3-polyform`, licencia PolyForm Noncommercial 1.0.0).
- **Configuración congelada del modelo:** `version_codigo: 1.1.0-oof-definitivo` (`experiments/config_ashr_congelada.json`), semilla `2026`.
- **Protocolo interno de Optuna/CV:** `HPO_PROTOCOL_VERSION = "v1.0.3-exact-recursive-state"`, semilla `2026`, Optuna SQLite `RDBStorage`, escritura atómica (`fsync` + `os.replace`) y verificación SHA-256 en cada checkpoint. Nota: `SM-CFHR-HPO` en `v1.0.3` utiliza el mismo motor causal `PlantRecursiveState`/`PlantRecursiveWeatherState` en la validación interna de Optuna que en la evaluación externa y rechaza estudios SQLite de `v1.0.2`.

Para generar la evidencia obligatoria de reproducibilidad (`results/runs/<run_id>/reproducibilidad/`) y comparar dos corridas independientes:
```bash
python -m sm_cfhr.cli repro-bundle --run-id oof_definitivo_v1_1
python -m sm_cfhr.cli compare-runs --dir-a /ruta/run_equipo_A --dir-b /ruta/run_equipo_B --out-md reproducibility_comparison.md
```

## 17. Referencias científicas
Ver `book/references.bib` y `docs/auditoria_entregable3/AUDITORIA_R1_R2_R3.md`.

## 18. Licencia y atribuciones
Código bajo licencia **PolyForm Noncommercial 1.0.0** ([LICENSE](LICENSE)): uso gratuito permitido para investigación, academia y fines no comerciales. Para licenciamiento comercial escribir a **edmolinares@soulspacers.com**. Autor: **Enmanuel David Diaz Molinares** (Universidad del Norte, Barranquilla, Colombia).

## 19. Limitaciones
- La evaluación confirmatoria de los 28 escenarios solares requiere importar los datos privados XM/GEFSv12 y benchmarks (`654.35 MB`).
- Sustitución de datasets externos por los 28 escenarios solares pendiente de validación docente explícita (`autorizacion_alcance_solar_docente: false`).

## 20. Responsable del repositorio
Repositorio mantenido bajo propiedad exclusiva del autor en GitHub (`KrissV89`).
