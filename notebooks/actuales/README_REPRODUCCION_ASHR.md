# Guía de Reproducción — `SM-CFHR` (`sm-cfhr 1.0.3`) y Análisis OOF Definitivo

## 1. Requisitos, versiones e identificadores
- **Repositorio oficial:** `sm-cfhr` (`PolyForm-Noncommercial-1.0.0`, etiqueta `v1.0.3-polyform`)
- **Versión del paquete:** `sm-cfhr 1.0.3` (`pyproject.toml` y `sm_cfhr.__version__`)
- **Protocolo científico HPO:** `v1.0.3-exact-recursive-state`, semilla `2026`
- **Entorno validado:** Python `3.11.15` (`conda` env `solarpv-eda`), Windows 11 (`spawn`)

```powershell
# Desde la raíz del repositorio sm-cfhr
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-build-isolation --no-deps
python -m ipykernel install --user --name solarpv-eda --display-name "Python (solarpv-eda)"
```

## 2. Reproducción exclusiva del análisis desde los artefactos OOF cerrados (sin reentrenar)
Los resultados OOF definitivos consolidados (`Fixed` + `HPO`, `14,625` unidades por variante, `28` escenarios) residen en `resultados/evaluacion_ashr_oof/`. Para regenerar todas las tablas, estudios diagnósticos (etapas 2–8), figuras y el notebook científico de **19 secciones** sin repetir el entrenamiento OOF:

```powershell
# 1. Regenerar tablas, inferencia, estudios diagnósticos reales y figuras (etapas 2-8)
python -m sm_cfhr.cli report

# 2. Regenerar el notebook desde su fuente y ejecutarlo desde kernel limpio
python scripts/evaluacion/generar_notebook_ashr.py
python -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=solarpv-eda --ExecutePreprocessor.timeout=1800 --inplace notebooks/actuales/evaluacion_cientifica_ashr_oof.ipynb
```

## 3. Verificación automatizada
```powershell
python -m pytest tests/test_notebook_audit_fixes.py tests/test_parallel_oof.py tests/test_adaptive_structured_huber.py tests/test_selector_ashr_v2.py -v
```

## 4. Reejecución OOF paralela (solo referencia de auditoría)
Ver los comandos exactos, hashes SHA-256, política de reciclaje y telemetría observada (`Fixed`: `499.748 s`, `8` workers; `HPO`: `29,550.293 s` en dos sesiones, `12` workers) en `docs/reproducibilidad/CONDICIONES_COMPUTACIONALES_OOF.md`.
