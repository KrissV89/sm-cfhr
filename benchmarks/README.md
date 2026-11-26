# Benchmarks Oficiales (22 Modelos de Referencia en 28 Escenarios)

Los 28 archivos de predicciones oficiales de benchmarks suman **654.35 MB** en `resultados/benchmark_*` y se excluyen del repositorio Git por tamaño.
Para importarlos localmente con verificación de estructura:

```bash
python scripts/setup/import_local_data.py --source-root /ruta/a/carpeta_con_datos
python -m sm_cfhr.cli doctor
```
