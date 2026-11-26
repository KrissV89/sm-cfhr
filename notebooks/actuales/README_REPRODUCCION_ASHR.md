# Guía de Reproducción — `SM-CFHR` (`v3.0.0`)

## 1. Requisitos, versiones e identificadores
- **Repositorio:** `https://github.com/KrissV89/sm-cfhr` (`PolyForm-Noncommercial-1.0.0`, etiqueta `v3.0.0`)
- **Versión del paquete:** `sm-cfhr` (`pyproject.toml` y `sm_cfhr.__version__`)
- **Protocolo causal HPO:** `v1.0.3-exact-recursive-state`, semilla `2026`
- **Entorno validado:** Python `3.11.15` (`conda` env `solarpv-eda`)

```powershell
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-build-isolation --no-deps
python -m ipykernel install --user --name solarpv-eda --display-name "Python (solarpv-eda)"
```

## 2. Reproducción desde los artefactos OOF congelados (sin reentrenar)
Las predicciones OOF congeladas (`Fixed` + `HPO`, `14,625` unidades por variante, `28` escenarios) residen en `resultados/evaluacion_ashr_oof/`. Para regenerar tablas, figuras y cuaderno principal sin repetir el entrenamiento OOF:

```powershell
python -m sm_cfhr.cli report
python scripts/evaluacion/generar_notebook_ashr.py
python -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=solarpv-eda --ExecutePreprocessor.timeout=1800 --inplace notebooks/actuales/evaluacion_cientifica_ashr_oof.ipynb
```

## 3. Experimentos de equivalencia de representación y ablaciones (`v3.0.0`)
```powershell
python scripts/evaluacion/ejecutar_equivalencia_representacion_v3.py
python scripts/evaluacion/ejecutar_ablaciones_y_sesgo_v3.py
```

## 4. Manuscrito LaTeX y Material Suplementario
```powershell
cd docs/paper_sm_cfhr
pdflatex main.tex; bibtex main; pdflatex main.tex; pdflatex main.tex
pdflatex suplemento.tex; bibtex suplemento; pdflatex suplemento.tex; pdflatex suplemento.tex
```

## 5. Verificación automatizada
```powershell
python -m pytest tests/test_notebook_editorial_integrity.py tests/test_equivalencia_y_ablaciones_v3.py tests/test_auditoria_inferencia_v2.py tests/test_adaptive_structured_huber.py tests/test_notebook_audit_fixes.py -v
```
