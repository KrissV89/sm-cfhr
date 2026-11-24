# Checklist de Congelamiento y Reproducibilidad Pre-OOF

- **Commit base inspeccionado:** `cae2e98aa5bb60a59f1c86aca7bbe22ade1dc26d`
- **Entorno Python:** `C:\Users\Enma\miniconda3\envs\solarpv-eda\python.exe` (`3.11.15`)
- **Hash congelado de `src/adaptive_structured_huber.py`:** `ed89b2aadab9d3a365bee985c98bfb006c71d5da16ecb1694d757b2d87d72bad` (coincide con `experiments/config_ashr_congelada.json`).
- **Conjuntos confirmatorios:** `resultados/evaluacion_ashr_oof/` **NO EXISTE** (`oof_definitivo_ejecutado: false`).
- **Pruebas automatizadas:** **47/47 PASSED** (`208.03 s`).

## Pasos Previos al Disparo del OOF Definitivo

1. Revisar `git status` y confirmar que no se incluyen archivos temporales en el commit de congelamiento.
2. Crear commit y tag anotado:
   ```bash
   git add src/ scripts/ experiments/ notebooks/ docs/ tests/ resultados/manifiesto_preparacion_entregable3.json
   git commit -m "Freeze SM-CFHR / ASHR pre-OOF definitive pipeline (47/47 tests pass)"
   git tag -a sm-cfhr-oof-v1.0 -m "Congelamiento oficial pre-OOF 28 escenarios"
   ```
3. Verificar dry-run:
   ```bash
   python scripts/evaluacion/ejecutar_oof_definitivo_ashr.py --variante ambos --n-trials 25
   ```
4. Ejecutar evaluación definitiva (primero `fixed` ~14 min; opcional `ambos` ~10 h con checkpointing):
   ```bash
   python scripts/evaluacion/ejecutar_oof_definitivo_ashr.py --ejecutar-oof-definitivo --variante fixed
   ```
5. Re-ejecutar el notebook científico en kernel limpio y actualizar `book/` y `app.py` con los artefactos definitivos.
