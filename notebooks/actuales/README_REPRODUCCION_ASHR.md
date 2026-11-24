# Guía de Reproducción — `SM-CFHR` / `ASHR` y Pipeline OOF

## 1. Requisitos e instalación

Entorno validado: **Python `3.11.15`** (`solarpv-eda`). Todos los comandos deben ejecutarse desde la **raíz del repositorio** (`InvML/`):

```bash
pip install -r requirements-ashr.txt
```

Para registrar el kernel de Jupyter:

```bash
python -m ipykernel install --user --name solarpv-eda --display-name "Python (solarpv-eda)"
```

## 2. Verificación integral previa (20+ pruebas, no ejecuta OOF definitivo)

```bash
python -m pytest tests/test_adaptive_structured_huber.py tests/test_selector_ashr_v2.py tests/test_ashr_oof_pipeline.py tests/test_auditoria_integral_entregable3.py -v
python scripts/evaluacion/ejecutar_oof_definitivo_ashr.py
```

## 3. Ejecución del OOF definitivo (solo cuando esté autorizada)

```bash
python scripts/evaluacion/ejecutar_oof_definitivo_ashr.py --ejecutar-oof-definitivo
```

Artefactos generados por esa corrida:
- `resultados/evaluacion_ashr_oof/predicciones/*.csv` (28 escenarios alineados)
- `resultados/evaluacion_ashr_oof/manifest_ashr_oof.json` (estatus, tiempos y SHA-256)

## 4. Reproducción del notebook científico (16 secciones, modo dual A/B)

Ubicación del notebook: `notebooks/actuales/evaluacion_cientifica_ashr_oof.ipynb`.

Ejecutar el notebook **nunca** lanza entrenamientos OOF definitivos:
- **Estado A** (sin `resultados/evaluacion_ashr_oof/`): verifica configuración, compatibilidad `scikit-learn` e inventario de 28 escenarios, reportando estado `PENDIENTE`.
- **Estado B** (con manifiesto OOF válido): carga los 28 escenarios, exporta tablas en `resultados/evaluacion_ashr_oof/tablas_paper/` y figuras en `figuras/ashr_oof/`.

Para regenerar y ejecutar desde un kernel limpio:

```bash
python scripts/evaluacion/generar_notebook_ashr.py
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=solarpv-eda --inplace notebooks/actuales/evaluacion_cientifica_ashr_oof.ipynb
```

## 5. Documentos de auditoría y protocolo pre-registrado

- `experiments/protocolo.md`
- `docs/auditoria_entregable3/MATRIZ_CUMPLIMIENTO_ENTREGABLE3.md`
- `docs/auditoria_entregable3/AUDITORIA_R1_R2_R3.md`
- `docs/auditoria_entregable3/AUDITORIA_BENCHMARKS_Y_PRESUPUESTOS.md`
- `docs/auditoria_entregable3/PROTOCOLO_OPTIMIZACION_ASHR.md`
- `docs/auditoria_entregable3/AUDITORIA_DASHBOARD_Y_E2.md`
