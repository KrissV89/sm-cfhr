# SM-CFHR — Normalización Causal en Factor de Capacidad y Regresor Huber Estructurado

> **Marco reproducible para pronóstico fotovoltaico diario ($D+1$) y subestacional ($7$--$28$ días) multi-planta y multi-horizonte, con validación cruzada causal anidada, formalización en Lean~4 y experimentos controlados de equivalencia de representación.**

---

## 1. Descripción científica

**SM-CFHR** (*Structured Monotone Capacity-Factor Huber Regressor*, implementado en `AdaptiveStructuredHuberRegressor` y expuesto mediante `SMCFHRRegressor` compatible con `scikit-learn`) aborda el pronóstico determinístico sobre una cohorte real de **17 centrales solares en Colombia** ($G = 7$ zonas espaciales, enero de 2024 – octubre de 2026; capacidades netas entre $9{,}8$ y $388{,}0\text{ MW}$, cociente de escala $39{,}6\times$).

El estudio evalúa dos regímenes —**transferencia espaciotemporal (`ET`)** mediante *leave-one-group-out* y **pronóstico intraplanta (`IP`)**— con y sin variables meteorológicas (`Open-Meteo` / `NOAA GEFSv12`) bajo estrategias $D+1$, recursiva y directa acumulada ($28$ configuraciones en total).

### Hallazgo empírico central
- **Normalización causal por capacidad (`24.000 * C_{i,t}`):** al equiparar la representación de entrada y objetivo a factor de capacidad causal (`CF`) en lugar de energía bruta (`kWh`), las siete familias evaluadas (`BayesianRidge`, `ElasticNet`, `HuberRegressor`, `Ridge`, `RandomForest`, `XGBoost` y `SM-CFHR`) reducen su $\mathrm{MAE}_{\mathrm{CF,macro}}$ medio entre **$-10{,}13\%$ y $-23{,}24\%$**.
- **Algoritmo frente a representación:** bajo representación `CF` equiparada, `SM-CFHR` empata o queda a $+0{,}13\%$--$+1{,}58\%$ de `BayesianRidge-CF` y `ElasticNet-CF` en transferencia espaciotemporal (rango $1$ únicamente en `RecGEFSv12_ET_7d`, $-0{,}56\%$, en condición fija, y en `D1_ET_5var`, $-0{,}09\%$, en validación interna de $3$ pliegues) y pierde en régimen intraplanta ($+3{,}23\%$ a $+16{,}24\%$).
- **Configuración fija frente a HPO:** `SM-CFHR-Fixed` ($B=0$) supera a `SM-CFHR-HPO` ($B=25$ ensayos Optuna TPE) en **$19/28$ escenarios** ($14/14$ en `ET`) con un coste computacional **$59\times$ menor** ($499{,}75\text{ s}$ frente a $29\,550{,}3\text{ s}$).

En consecuencia, la contribución principal verificada es el **marco de normalización causal por capacidad vigente al origen, despliegue multihorizonte sin fuga e inferencia macro-alineada**, no una superioridad algorítmica universal de SM-CFHR frente a lineales regularizados en factor de capacidad.

---

## 2. Formulación resumida

$$\min_{b \in \mathbb{R},\; \mathbf{w} \in \mathcal{C}} \; \frac{1}{n}\sum_{k=1}^{n} \rho_\delta\!\left(\tilde{y}_k - b - \mathbf{z}_k^\top \mathbf{w}\right) + \frac{1}{2}\mathbf{w}^\top D_\alpha \mathbf{w}, \qquad \mathcal{C} = \{\mathbf{w} \in \mathbb{R}^p : w_j \ge 0 \;\forall j \in \mathcal{M}_+\}$$

donde $\tilde{y}_k = (y_k - m_y)/\sigma_y$ emplea mediana y MAD en entrenamiento, $D_\alpha = \diag(d_1,\dots,d_p)$ separa penalizaciones $\alpha_{\mathrm{lin}} = 0{,}01$ y $\alpha_{\mathrm{meteo}} = 0{,}10$, y $\mathcal{M}_+$ restringe la persistencia reciente (`cf30`) y la irradiación pronosticada (`rad_pron`), resuelto mediante L-BFGS-B con gradiente analítico exacto.

---

## 3. Organización del repositorio

La documentación se clasifica en tres niveles explícitos (véase [`docs/README.md`](docs/README.md)):
- **Nivel A (Publicación científica):** [`docs/paper_sm_cfhr/`](docs/paper_sm_cfhr/) (`main.pdf`, `suplemento.pdf`, fuentes LaTeX y figuras vectoriales) y [`notebooks/actuales/evaluacion_cientifica_ashr_oof.ipynb`](notebooks/actuales/evaluacion_cientifica_ashr_oof.ipynb).
- **Nivel B (Documentación técnica y reproducibilidad):** `README.md`, [`PROVENANCE.md`](PROVENANCE.md), [`src/sm_cfhr/README.md`](src/sm_cfhr/README.md), [`notebooks/actuales/README_REPRODUCCION_ASHR.md`](notebooks/actuales/README_REPRODUCCION_ASHR.md), [`data/README.md`](data/README.md) y [`docs/reproducibilidad/`](docs/reproducibilidad/).
- **Nivel C (Historial interno y auditorías):** [`docs/revisiones/`](docs/revisiones/), [`docs/auditoria_entregable3/`](docs/auditoria_entregable3/), [`docs/auditoria_preflight_final/`](docs/auditoria_preflight_final/), [`docs/investigacion/`](docs/investigacion/), [`book/`](book/) y [`dashboard/`](dashboard/).

Componentes de código y verificación:
- `src/adaptive_structured_huber.py` y `src/sm_cfhr/`: estimador `SMCFHRRegressor`, inferencia (`inference.py`, `inference_v2.py`), métricas auditadas (`metrics_audit.py`), equivalencia de representación (`representation_benchmark.py`) y coordinador paralelo (`parallel/`).
- `src/ashr_oof_pipeline/`: particiones causales anidadas, despliegue recursivo sin fuga y evaluación OOF de $28$ escenarios.
- `scripts/evaluacion/`: reproducción de OOF, inferencia `v2`, experimento de equivalencia `v3` (`ejecutar_equivalencia_representacion_v3.py`) y ablaciones/sesgo `v3` (`ejecutar_ablaciones_y_sesgo_v3.py`).
- `formal/sm_cfhr_lean/`: formalización en Lean~4 (`v4.36.0` / Mathlib) de $26$ declaraciones sobre $\mathbb{R}$.
- `tests/`: pruebas unitarias, anti-fuga, inferencia estadística, integridad de cuadernos y esquemas `v3`.

---

## 4. Instalación y obtención de datos

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

Los datasets solares reales y los $28$ CSVs de referencia (`654.35 MB`) no se alojan en GitHub por términos de redistribución de XM y tamaño (véanse [`data/README.md`](data/README.md) y [`data/manifests/data_manifest.json`](data/manifests/data_manifest.json)). Tras obtener los archivos locales autorizados:

```bash
python scripts/setup/import_local_data.py --source-root /ruta/a/datos_autorizados
python -m sm_cfhr.cli doctor
```

---

## 5. Reproducción de experimentos y pruebas

### Pruebas automatizadas
```bash
python -m pytest tests/test_adaptive_structured_huber.py tests/test_auditoria_inferencia_v2.py tests/test_equivalencia_y_ablaciones_v3.py tests/test_notebook_editorial_integrity.py -v
```

### Equivalencia de representación y ablaciones controladas (`v3.0.0`)
```bash
python scripts/evaluacion/ejecutar_equivalencia_representacion_v3.py
python scripts/evaluacion/ejecutar_ablaciones_y_sesgo_v3.py
```

### Reproducción OOF completa y reportes (opcional, requiere datos importados)
```bash
python scripts/ejecutar_oof_paralelo.py --workers 8 --variante fixed --run-id oof_v1_0_3_fixed --data-root /ruta/a/datos_autorizados --confirm-oof
python scripts/ejecutar_oof_paralelo.py --workers 8 --variante hpo --n-trials 25 --max-tasks-per-worker 60 --worker-max-rss-mb 1536 --run-id oof_v1_0_3_hpo --data-root /ruta/a/datos_autorizados --confirm-oof
python -m sm_cfhr.cli report
```

### Cuadernos y manuscrito LaTeX
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/actuales/evaluacion_cientifica_ashr_oof.ipynb
cd docs/paper_sm_cfhr && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## 6. Limitaciones conocidas
- Cohorte única de $17$ centrales fotovoltaicas colombianas ($G = 7$ conglomerados espaciales), sin conjuntos de datos externos no solares.
- Reutilización adaptativa del panel histórico enero 2024 – octubre 2026 (requiere validación prospectiva congelada independiente).
- En transferencia espaciotemporal (`ET`), la central evaluada aporta su historial causal previo al origen (`cf30`, `cf7`, `cf1`), por lo que no equivale a predicción *zero-shot*.
- La pérdida de Huber acota la influencia de anomalías en el objetivo, pero no protege frente a puntos de apalancamiento en covariables.
- Los intervalos conformes temporales muestran subcobertura bajo deriva entre bloques ($83{,}24\%$ para $90\%$ nominal; $69{,}25\%$ para $80\%$).

---

## 7. Licencia y citación
Software original distribuido bajo **PolyForm Noncommercial 1.0.0** ([LICENSE](LICENSE)); para uso comercial contactar a **edmolinares@soulspacers.com**. Los datos externos (XM, Open-Meteo, Global Energy Monitor) y dependencias de terceros conservan sus propias licencias. Ver [`CITATION.cff`](CITATION.cff) y [`PROVENANCE.md`](PROVENANCE.md).
