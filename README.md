# SM-CFHR — Regresor Huber Estructurado y Monótono en Factor de Capacidad

> **Marco reproducible para pronóstico fotovoltaico diario ($D+1$) y subestacional ($7$--$28$ días) multi-planta y multi-horizonte, con validación cruzada causal anidada, formalización en Lean~4 y evaluación estadística reconciliada.**

---

## 1. Descripción científica

**SM-CFHR** (*Structured Monotone Capacity-Factor Huber Regressor*, implementado en `AdaptiveStructuredHuberRegressor` y expuesto mediante `SMCFHRRegressor` compatible con `scikit-learn`) aborda el pronóstico determinístico sobre una cohorte real de **17 centrales solares en Colombia** ($G = 7$ zonas espaciales, enero de 2024 – octubre de 2026; capacidades netas entre $9{,}8$ y $388{,}0\text{ MW}$, cociente de escala $39{,}6\times$).

El estudio evalúa dos regímenes —**transferencia espaciotemporal (`ET`)** mediante *leave-one-group-out* y **pronóstico intraplanta (`IP`)**— con y sin variables meteorológicas (`Open-Meteo` / `NOAA GEFSv12`) bajo estrategias $D+1$, recursiva y directa acumulada ($28$ configuraciones en total frente a $22$ modelos de referencia en energía bruta).

### Hallazgos empíricos congelados
- **Desempeño frente a referencias en energía bruta:** `SM-CFHR-Fixed` ($B=0$) alcanza rango medio $2{,}93/23$ en $\mathrm{MAE}_{\mathrm{macro}}$ ($\text{kWh}$, $18/28$ victorias descriptivas frente a `Mejor_Bib`) y rango medio $4{,}00/23$ en factor de capacidad adimensional ($\mathrm{MAE}_{\mathrm{CF,macro}}$, $16/28$ victorias).
- **Inferencia reconciliada (`v2.0.0`):** sobre el estimando exacto $\mathrm{MAE}_{\mathrm{macro}}$ con bloques disjuntos ($h_{\mathrm{serie}}=1$), se acreditan $5/5$ victorias Holm en transferencia espaciotemporal con clima (`ET + Clima`, $p_{\mathrm{Holm}} \le 5{,}21\times 10^{-6}$), $0$ victorias en régimen intraplanta o sin clima y derrotas hasta $+24{,}24\%$ en estrategia directa intraplanta sin clima (`DirectoSinClima_IP`).
- **Configuración fija frente a HPO:** `SM-CFHR-Fixed` ($B=0$) supera a `SM-CFHR-HPO` ($B=25$ ensayos Optuna TPE) en **$19/28$ escenarios** ($14/14$ en `ET`) con un coste computacional **$59\times$ menor** ($499{,}75\text{ s}$ frente a $29\,550{,}3\text{ s}$).

---

## 2. Formulación resumida

$$\min_{b \in \mathbb{R},\; \mathbf{w} \in \mathcal{C}} \; \frac{1}{n}\sum_{k=1}^{n} \rho_\delta\!\left(\tilde{y}_k - b - \mathbf{z}_k^\top \mathbf{w}\right) + \frac{1}{2}\mathbf{w}^\top D_\alpha \mathbf{w}, \qquad \mathcal{C} = \{\mathbf{w} \in \mathbb{R}^p : w_j \ge 0 \;\forall j \in \mathcal{M}_+\}$$

donde $\tilde{y}_k = (y_k - m_y)/\sigma_y$ emplea mediana y MAD en entrenamiento, $D_\alpha = \diag(d_1,\dots,d_p)$ separa penalizaciones $\alpha_{\mathrm{lin}} = 0{,}01$ y $\alpha_{\mathrm{meteo}} = 0{,}10$, y $\mathcal{M}_+$ restringe la persistencia reciente (`cf30`) y la irradiación pronosticada (`rad_pron`), resuelto mediante L-BFGS-B con gradiente analítico exacto.

---

## 3. Organización del repositorio

La documentación se clasifica en tres niveles explícitos (véase [`docs/README.md`](docs/README.md)):
- **Nivel A (Publicación científica):** [`docs/paper_sm_cfhr/`](docs/paper_sm_cfhr/) (`main.pdf`, `suplemento.pdf`, fuentes LaTeX y figuras vectoriales) y [`notebooks/actuales/evaluacion_cientifica_ashr_oof.ipynb`](notebooks/actuales/evaluacion_cientifica_ashr_oof.ipynb).
- **Nivel B (Documentación técnica y reproducibilidad):** `README.md`, [`PROVENANCE.md`](PROVENANCE.md), [`src/sm_cfhr/README.md`](src/sm_cfhr/README.md), [`notebooks/actuales/README_REPRODUCCION_ASHR.md`](notebooks/actuales/README_REPRODUCCION_ASHR.md), [`data/README.md`](data/README.md) y [`docs/reproducibilidad/`](docs/reproducibilidad/).
- **Nivel C (Historial interno y auditorías):** [`docs/revisiones/`](docs/revisiones/), [`docs/auditoria_entregable3/`](docs/auditoria_entregable3/), [`docs/auditoria_preflight_final/`](docs/auditoria_preflight_final/), [`docs/investigacion/`](docs/investigacion/), [`book/`](book/), [`dashboard/`](dashboard/) y [`archive/experimentos_no_autorizados_cf_benchmarks/`](archive/experimentos_no_autorizados_cf_benchmarks/) (aislado y excluido del estudio oficial).

Componentes principales de código y verificación:
- `src/adaptive_structured_huber.py` y `src/sm_cfhr/`: estimador `SMCFHRRegressor`, inferencia (`inference.py`, `inference_v2.py`), métricas auditadas (`metrics_audit.py`) y coordinador paralelo (`parallel/`).
- `src/ashr_oof_pipeline/`: particiones causales anidadas, despliegue recursivo sin fuga y evaluación OOF de $28$ escenarios.
- `scripts/` y `scripts/evaluacion/`: reproducción de OOF, inferencia `v2` y generadores de cuadernos.
- `formal/sm_cfhr_lean/`: formalización en Lean~4 (`v4.36.0` / Mathlib) de $26$ declaraciones sobre $\mathbb{R}$.
- `tests/`: pruebas unitarias, anti-fuga, inferencia estadística e integridad de cuadernos.

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

## 5. Reproducción y pruebas

### Pruebas automatizadas
```bash
python -m pytest tests/test_adaptive_structured_huber.py tests/test_auditoria_inferencia_v2.py tests/test_notebook_editorial_integrity.py tests/test_notebook_audit_fixes.py -v
```

### Reportes desde OOF congelado (sin reentrenar)
```bash
python -m sm_cfhr.cli report
```

### Cuadernos y manuscrito LaTeX
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/actuales/evaluacion_cientifica_ashr_oof.ipynb
cd docs/paper_sm_cfhr && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## 6. Limitaciones conocidas
- Las $22$ referencias comparadas operaron en energía bruta (`kWh`), por lo que la separación controlada entre el efecto de la normalización por capacidad y el del algoritmo queda abierta como trabajo futuro.
- Cohorte única de $17$ centrales fotovoltaicas colombianas ($G = 7$ conglomerados espaciales), sin conjuntos de datos externos no solares.
- Reutilización adaptativa del panel histórico enero 2024 – octubre 2026 (requiere validación prospectiva independiente).
- En transferencia espaciotemporal (`ET`), la central evaluada aporta su historial causal previo al origen (`cf30`, `cf7`, `cf1`), por lo que no equivale a predicción *zero-shot*.
- La pérdida de Huber acota la influencia de anomalías en el objetivo, pero no protege frente a puntos de apalancamiento en covariables.
- Los intervalos conformes temporales muestran subcobertura bajo deriva entre bloques ($83{,}24\%$ para $90\%$ nominal; $69{,}25\%$ para $80\%$).

---

## 7. Licencia y citación
Software original distribuido bajo **PolyForm Noncommercial 1.0.0** ([LICENSE](LICENSE)); para uso comercial contactar a **edmolinares@soulspacers.com**. Los datos externos (XM, Open-Meteo, Global Energy Monitor) y dependencias de terceros conservan sus propias licencias. Ver [`CITATION.cff`](CITATION.cff) y [`PROVENANCE.md`](PROVENANCE.md).
