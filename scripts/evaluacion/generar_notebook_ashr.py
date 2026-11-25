"""Generator script for the definitive 19-section scientific notebook of SM-CFHR / ASHR."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def build_notebook(out_path: Path) -> Path:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python (solarpv-eda)", "language": "python", "name": "solarpv-eda"},
        "language_info": {"name": "python", "version": "3.11.15"},
    }
    cells = []
    cell_counter = 0

    def _next_id(prefix: str) -> str:
        nonlocal cell_counter
        cell_counter += 1
        return f"smcfhr-{prefix}-{cell_counter:02d}"

    def md(txt: str) -> None:
        c = nbf.v4.new_markdown_cell(txt.strip())
        c["id"] = _next_id("md")
        cells.append(c)

    def code(txt: str) -> None:
        c = nbf.v4.new_code_cell(txt.strip())
        c["id"] = _next_id("code")
        cells.append(c)

    # 1. Resumen y alcance
    md("""# Evaluación Científica Definitiva — `SM-CFHR` / `ASHR` (`AdaptiveStructuredHuberRegressor`)

Este cuaderno constituye el documento computacional central de la evaluación científica de **SM-CFHR** (*Structured Monotone Capacity-Factor Huber Regressor*). Reproduce íntegramente el análisis, tablas, figuras y auditorías a partir de los artefactos OOF definitivos cerrados en `resultados/evaluacion_ashr_oof/` **sin volver a entrenar** los 28 escenarios:
- **Estado A (`PENDING`):** sin resultados definitivos en `resultados/evaluacion_ashr_oof/`; presenta formulación, protocolo, compatibilidad `scikit-learn` e inventario sin mostrar cifras ficticias.
- **Estado B (`VALID`):** con manifiesto válido y 28 escenarios completos (`Fixed` + `HPO`), verifica integridad SHA-256, audita alineación, Optuna y Nested CV, ejecuta inferencia estadística y consolida tablas/figuras.
- **Estado C (`INVALID` / parcial):** detecta ausencias o inconsistencias de hash/esquema y bloquea conclusiones confirmatorias.""")

    md("""## 1. Introducción, alcance, datos causales y versiones

- **Alcance exclusivo:** 17 plantas fotovoltaicas de Colombia (registros XM y pronósticos meteorológicos Open-Meteo / GEFSv12) organizadas en **28 escenarios oficiales** (`4 D+1`, `8 Recursivo sin clima`, `8 Directo sin clima`, `8 Recursivo + GEFSv12`).
- **Identificadores de versión:** paquete `sm-cfhr 1.0.3` (`pyproject.toml` y `sm_cfhr.__version__`), licencia `PolyForm-Noncommercial-1.0.0`, etiqueta `v1.0.3-polyform`, protocolo interno `v1.0.3-exact-recursive-state`, commit base OOF `3256994`, semilla `2026`.
- **Nota de alcance académico:** No se incorporan datasets externos de otros dominios; la sustitución por los 28 escenarios solares queda documentada como modificación de alcance pendiente de validación docente explícita.""")

    code("""import json
import sys
from pathlib import Path
from IPython.display import display
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_CUR = Path.cwd().resolve()
ROOT = next(p for p in (_CUR, *_CUR.parents) if (p / "experiments" / "config_ashr_congelada.json").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sm_cfhr
from sm_cfhr import (
    AdaptiveStructuredHuberRegressor,
    SCENARIOS_28,
    SMCFHRRegressor,
    analyze_effective_design,
    evaluate_r3_superiority,
    export_full_scientific_artifacts,
    summarize_hpo_trials,
    verify_benchmark_inventory,
)
from sm_cfhr.plotting import (
    export_all_figures,
    plot_ablation_bar,
    plot_conformal_coverage,
    plot_feature_importance,
    plot_hp_sensitivity,
    plot_learning_and_scalability,
    plot_nested_cv_scheme,
    plot_optuna_convergence,
    plot_robustness_stress,
)
from sm_cfhr.pipeline_stages import apply_family_multiplicity
from src.ashr_oof_pipeline import (
    build_descriptive_summary,
    export_paper_tables,
    load_frozen_config,
    load_oof_bundle,
    validate_oof_artifacts,
)
from src.ashr_oof_pipeline.oof_runner import run_oof_definitivo

CFG = load_frozen_config(ROOT)
CONTRATO_RUTAS = CFG.get("rutas_contrato", {})
OOF_DIR = ROOT / CONTRATO_RUTAS.get("oof_dir", "resultados/evaluacion_ashr_oof")
HPO_TRIALS_DIR = ROOT / CONTRATO_RUTAS.get("hpo_trials_dir", "resultados/evaluacion_ashr_oof/hpo_trials")
FIG_DIR = OOF_DIR / "figuras"
TAB_DIR = OOF_DIR / "tablas_paper"
SCOPE_AUTHORIZED = bool(CFG.get("autorizacion_alcance_solar_docente", False))

print("Paquete sm_cfhr versión:", sm_cfhr.__version__)
print("Denominación científica:", CFG["denominacion_cientifica"])
print("Protocolo HPO congelado:", "v1.0.3-exact-recursive-state")
print("SHA-256 núcleo congelado:", CFG["sha256_src_adaptive_structured_huber"])""")

    # 2. Literatura y R1
    md("""## 2. Revisión de literatura y novedad acotada (R1)

El protocolo formal de literatura (`docs/auditoria_entregable3/AUDITORIA_R1_R2_R3.md`) documenta 24 exportaciones de **Web of Science Core Collection** (`fuentes/bibliografia/wos/`), análisis de 8 trabajos cercanos (Pico et al. 2026; Trigo-González et al. 2019; Almeida et al. 2015; Yang et al. 2020; Huber 1964; Owen 2007; K1/W1) y *snowballing*.

**Afirmación acotada de novedad:** Formulación paramétrica convexa propia en espacio adimensional de factor de capacidad (`CF`) que combina pérdida Huber con escala MAD, regularización $L_2$ diferenciada por familia y restricciones físicas monótonas (`cf30 >= 0`, `rad_pron >= 0`), optimizada con L-BFGS-B analítico propio en NumPy/SciPy.""")

    # 3. Formulacion matematica
    md(r"""## 3. Formulación matemática y propiedades de `SM-CFHR`

Sea $\mathrm{cap}_i > 0$ la capacidad instalada previa al origen y $B_i = \mathrm{cap}_i \cdot 24{,}000\;\text{kWh/día}$. En diarios/recursivos $c_i = \operatorname{clip}(y_i/B_i, 0, 1)$; en directos acumulados $c_i = \operatorname{clip}(Y_{H,i}/(H B_i), 0, 1)$.

Con respuesta robusta $\tilde{c}_i = (c_i - m_c)/\hat{\sigma}_c$ ($\hat{\sigma}_c = \max(1.4826\,\mathrm{MAD}(c), 10^{-6})$) y predictores estandarizados $z_i$:

$$\min_{\theta_0 \in \mathbb{R},\; \theta \in \mathcal{C}} \; \frac{1}{n}\sum_{i=1}^n \rho_\delta\!\left(\tilde{c}_i - \theta_0 - z_i^\top \theta\right) + \frac{1}{2}\sum_{j \in \mathcal{G}_{\mathrm{lin}}} \alpha_{\mathrm{lin}}\,\theta_j^2 + \frac{1}{2}\sum_{j \in \mathcal{G}_{\mathrm{meteo}}} \alpha_{\mathrm{meteo}}\,\theta_j^2$$

sujeto a $\mathcal{C} = \{\theta : \theta_j \ge 0 \;\forall j \in \mathcal{M}_{\mathrm{pos}}\}$.""")

    # 4. Paquete sm_cfhr e integracion sklearn (R2)
    md("""## 4. Paquete `sm_cfhr` e integración `scikit-learn` (R2)

`SMCFHRRegressor` y `AdaptiveStructuredHuberRegressor` heredan de `(BaseEstimator, RegressorMixin)` sin delegar el algoritmo en estimadores de `scikit-learn`. Se verifica `clone`, `Pipeline` y `get_params` sin ejecutar entrenamiento OOF.""")

    code("""hp_cfg = CFG["parametros_modelo"]["fixed_config"]
sklearn_demo_model = SMCFHRRegressor(
    delta=CFG["parametros_modelo"]["delta"],
    alpha_lin=hp_cfg["alpha_lin"],
    alpha_meteo=hp_cfg["alpha_meteo"],
    structured=hp_cfg["structured"],
    use_mono=hp_cfg["use_mono"],
    random_state=CFG["parametros_modelo"]["random_state"],
)
cloned_demo = clone(sklearn_demo_model)
pipe_demo = Pipeline([("scaler", StandardScaler()), ("sm_cfhr", cloned_demo)])

print("Es BaseEstimator:", isinstance(sklearn_demo_model, BaseEstimator))
print("Es RegressorMixin:", isinstance(sklearn_demo_model, RegressorMixin))
print("get_params == clone.get_params:", sklearn_demo_model.get_params() == cloned_demo.get_params())
print("Pasos de Pipeline:", [name for name, _ in pipe_demo.steps])""")

    # 5. Optuna TPE
    md("""## 5. Optimización bayesiana Optuna TPE y auditoría de estudios

Se comparan dos variantes del mismo núcleo fijo (`n_knots=0, max_interactions=0, anchor="none", structured=True, use_mono=True`):
- **`SM-CFHR-Fixed` ($B=0$):** `delta=1.35, alpha_lin=0.01, alpha_meteo=0.10`.
- **`SM-CFHR-HPO` ($B=25$):** Optuna `TPESampler` sobre `delta ∈ [0.5, 3.0]`, `alpha_lin ∈ [1e-4, 1.0]` (log) y `alpha_meteo ∈ [1e-3, 10.0]` (log, vinculado a `alpha_lin` si no hay meteorología), optimizando `MAE_macro_kWh` dentro de 3 particiones causales internas más 1 *refit* final por unidad.

**Nota de auditoría de estudios:** cada estudio Optuna queda unívocamente identificado por `(scenario_id, outer_fold)`. De las `14,625` unidades HPO totales, `14,552` ejecutaron estudio Optuna (`14,544` con `25` filas y `25` trials válidos; `8` unidades reanudadas registraron `26` filas con `1` trial fallido/interrumpido y exactamente `25` trials `COMPLETE` válidos), mientras que `73` unidades tempranas sin historial suficiente usaron la configuración por defecto sin HPO.""")

    code("""hpo_space_df = pd.DataFrame([
    {"hiperparametro": "delta", "escala": "uniforme", "min": 0.5, "max": 3.0, "valor_fixed": CFG["parametros_modelo"]["delta"]},
    {"hiperparametro": "alpha_lin", "escala": "log-uniforme", "min": 1e-4, "max": 1.0, "valor_fixed": hp_cfg["alpha_lin"]},
    {"hiperparametro": "alpha_meteo", "escala": "log-uniforme (si weather_on)", "min": 1e-3, "max": 10.0, "valor_fixed": hp_cfg["alpha_meteo"]},
])
display(hpo_space_df)

hpo_trials_df = pd.DataFrame()
if HPO_TRIALS_DIR.exists():
    trial_files = sorted(HPO_TRIALS_DIR.glob("*.csv"))
    if trial_files:
        hpo_trials_df = pd.concat([pd.read_csv(f).assign(scenario_id=f.stem) for f in trial_files], ignore_index=True)
        hpo_summary_df = summarize_hpo_trials(hpo_trials_df)
        print(
            f"Estudios Optuna únicos (scenario_id, outer_fold): {len(hpo_summary_df):,} | "
            f"Trials válidos totales: {int(hpo_summary_df['n_valid_trials'].sum()):,} | "
            f"Min/Max trials válidos por estudio: {int(hpo_summary_df['n_valid_trials'].min())}/{int(hpo_summary_df['n_valid_trials'].max())}"
        )
        hpo_by_scenario = (
            hpo_summary_df.groupby("scenario_id", as_index=False)
            .agg(
                n_estudios=("outer_fold", "count"),
                trials_validos_por_estudio_min=("n_valid_trials", "min"),
                trials_validos_por_estudio_max=("n_valid_trials", "max"),
                mediana_delta=("best_delta", "median"),
                mediana_alpha_lin=("best_alpha_lin", "median"),
                mediana_alpha_meteo=("best_alpha_meteo", "median"),
                tiempo_total_trials_s=("total_wall_time_s", "sum"),
            )
        )
        display(hpo_by_scenario)
        plot_optuna_convergence(hpo_trials_df, FIG_DIR)
else:
    print("Logs de trials Optuna TPE: PENDIENTE DE EJECUCIÓN OOF HPO.")""")

    # 6. Nested CV
    md("""## 6. Validación cruzada causal anidada temporal y espacial

- **D+1 (`d1`):** ventana creciente con embargo (`30d` cuando el historial lo permite), $\\max(\\text{date}_{\\text{train}}) < \\min(\\text{date}_{\\text{val}})$.
- **Directo (`direct`):** $\\max(\\text{t\\_end\\_target}_{\\text{train}}) \\le \\min(\\text{t\\_origin}_{\\text{val}})$.
- **Recursivo (`recursive`):** $\\max(\\text{origin}_{\\text{train}}) + H \\le \\min(\\text{origin}_{\\text{val}})$.
- **Espaciotemporal (`ET`):** retención de grupo espacial completo en validación interna (`spatial_disjoint=True`).

La tabla siguiente proviene de `particiones/nested_cv_audit.csv`, reconstruida directamente sobre los paneles reales mediante `build_causal_inner_splits` y `audit_inner_splits` para folds representativos de las 4 suites (en ET e IP).""")

    code("""nested_audit_path = OOF_DIR / "particiones" / "nested_cv_audit.csv"
nested_audit_df = pd.DataFrame()
if nested_audit_path.exists():
    nested_audit_df = pd.read_csv(nested_audit_path)
    print(
        f"Auditoría Nested CV: {len(nested_audit_df)} splits representativos | "
        f"Gap efectivo mínimo: {int(nested_audit_df['effective_gap_days'].min())} días | "
        f"Espacialmente disjuntos (ET): {bool(nested_audit_df.loc[nested_audit_df['task']=='espaciotemporal', 'spatial_disjoint'].all())}"
    )
    display(nested_audit_df)
    plot_nested_cv_scheme(nested_audit_df, FIG_DIR)
else:
    print("Trazabilidad de folds de Nested CV definitiva: PENDIENTE DE GENERACIÓN DE AUDITORÍA.")""")

    # 7. Protocolo e inventario 28 escenarios y 22 benchmarks
    md("""## 7. Protocolo experimental e inventario de 28 escenarios y 22 benchmarks

La comparación principal conservadora enfrenta `SM-CFHR-Fixed` ($B=0$) contra los **22 modelos oficiales** (`9` librerías, `12` custom, `MM30`; `CF-BRH` y `R-ILWR-PA v3` excluidos). `SM-CFHR-HPO` ($B=25$) se reporta en paralelo declarando la asimetría frente a rejillas de 4 puntos.""")

    code("""inv_df = verify_benchmark_inventory(ROOT)
print(f"Escenarios verificados: {len(inv_df)}/28 | Archivos OK: {inv_df['file_exists'].all()} | Columnas OK: {inv_df['cols_ok'].all()} | CF-BRH excluido: {inv_df['cf_brh_excluded'].all()}")
display(inv_df)""")

    # 8. Carga, deteccion Estado A/B/C y auditoria de alineacion
    md("""## 8. Detección de estado, integridad SHA-256 y auditoría de alineación

Esta celda verifica manifiesto, hashes SHA-256 y la alineación exacta de claves `(code, date/origen/lead_dias)` entre `SM-CFHR-Fixed`, `SM-CFHR-HPO` y los 22 benchmarks oficiales. En `D1` y `Recursivo`, los orígenes extra son `0`; en los 8 escenarios `DirectoSinClima`, `ashr_df` contenía `2`–`63` orígenes tempranos (`len(train) >= 30`) fuera del soporte del CSV oficial del benchmark, por lo que `load_oof_bundle` intersecta sobre las claves comunes exactas del benchmark antes de evaluar.""")

    code("""dry_status = run_oof_definitivo(execute=False, root=ROOT)
val_status = validate_oof_artifacts(OOF_DIR, CFG["sha256_src_adaptive_structured_huber"])
state = val_status["state"]
print("Estado detectado del notebook:", state)
for rsn in val_status["reasons"]:
    print(" -", rsn)

bundle = None
hpo_preds_available = False
if state == "VALID":
    bundle = load_oof_bundle(OOF_DIR, root=ROOT)
    hpo_files = list((OOF_DIR / "predicciones").glob("*_hpo.csv"))
    hpo_preds_available = len(hpo_files) == len(SCENARIOS_28)
    print(f"Estado B ACTIVO — 28 escenarios validados y alineados (HPO disponible: {hpo_preds_available}).")
    align_path = OOF_DIR / "inferencia" / "auditoria_alineacion_benchmarks.csv"
    if align_path.exists():
        align_df = pd.read_csv(align_path)
        print(
            f"Claves Fixed == HPO en 28/28 escenarios: {bool(align_df['fixed_hpo_keys_identical'].all())} | "
            f"Escenarios con orígenes extra excluidos: {int((align_df['n_extra_excluded'] > 0).sum())}/28 (todos DirectoSinClima)"
        )
        display(align_df)
elif state == "INVALID":
    print("ESTADO C (INCONSISTENTE / PARCIAL): se bloquean conclusiones confirmatorias.")
else:
    print("ESTADO A (PENDIENTE): ejecutar OOF definitivo antes de analizar.")""")

    # 9. Resultados OOF globales y comparacion Fixed vs HPO
    md("""## 9. Resultados OOF globales y comparación `SM-CFHR-Fixed` vs. `SM-CFHR-HPO`

- La métrica confirmatoria oficial es **`MAE_macro`** (media por planta `code` y luego promedio entre plantas, en `kWh/dia` para `D+1`/recursivo y `kWh_total` para directo).
- Se reporta **`MAE_micro`** (media global por observación) únicamente como métrica secundaria explícita. Sobre el soporte común alineado, **ambas métricas (`MAE_macro` y `MAE_micro`) coinciden en `9 / 28` victorias de `HPO` sobre `Fixed`** (todas en escenarios `intraplanta`, mientras `Fixed` gana en `14/14` escenarios `espaciotemporal` y `5/14` `intraplanta`). La discrepancia previa de 10 victorias micro en un borrador anterior provenía de incluir `63` orígenes tempranos no alineados en `DirectoSinClima_intraplanta_7d`.""")

    code("""if bundle is not None:
    print("=== Resumen por régimen y tarea (SM-CFHR-Fixed) ===")
    display(bundle["tabla_rankings_regimen"])
    print("=== Resultados completos por escenario (SM-CFHR-Fixed) ===")
    display(bundle["tabla_28_escenarios"])
    fh_path = OOF_DIR / "metricas" / "comparacion_fixed_vs_hpo.csv"
    if fh_path.exists():
        fh_df = pd.read_csv(fh_path)
        print(
            f"=== Comparación Fixed (B=0) vs HPO (B=25) sobre soporte alineado ===\\n"
            f"Victorias HPO vs Fixed (MAE_macro confirmatorio): {int((fh_df['diff_macro_HPO_minus_Fixed'] < 0).sum())}/28 | "
            f"(MAE_micro secundario): {int((fh_df['diff_micro_HPO_minus_Fixed'] < 0).sum())}/28\\n"
            f"Rango medio en 23 — Fixed: {fh_df['Rank_Fixed_en_23'].mean():.2f} | HPO: {fh_df['Rank_HPO_en_23'].mean():.2f}"
        )
        display(fh_df)
    export_all_figures(bundle, FIG_DIR)
else:
    print("Tablas OOF completas y figuras 1–5: PENDIENTE DE ESTADO B.")""")

    # 10. Inferencia estadistica
    md("""## 10. Inferencia estadística (DM-HLN-HAC, MCS, SPA post-hoc y multiplicidad)

- `DM-HLN-HAC` y `MCS` ($\alpha=0{,}10$, $B=200$) integran la evaluación automática del flujo OOF sobre series temporales agregadas por fecha/origen.
- `SPA` ($B=500$, longitud media de bloque $7$, [tabla_spa.csv](inferencia/tabla_spa.csv)) se ejecuta como análisis complementario post-hoc bajo $H_0$: `MM30` no es superada y bajo $H_0$: `ASHR` no es superado.
- **Cautela interpretativa:** ganar descriptivamente (`18/28` vs mejor biblioteca) o permanecer en el MCS (`27/28`) **no** equivale a demostrar superioridad estadística general cuando solo `2/15` escenarios con $T_{\\text{eff}} \\ge 2H$ alcanzan significancia tras Holm–Bonferroni.""")

    code("""if bundle is not None:
    t_inf = apply_family_multiplicity(bundle["tabla_inferencia"])
    t_mcs = bundle["tabla_mcs"].copy()
    print("=== Inferencia pareada DM-HLN-HAC + Holm/BH por familia ===")
    display(t_inf)
    print(f"=== Model Confidence Set (alpha=0.10): ASHR sobrevive en {int(t_mcs['ashr_in_mcs_90'].sum())}/{len(t_mcs)} escenarios ===")
    display(t_mcs)
    spa_path = OOF_DIR / "inferencia" / "tabla_spa.csv"
    if spa_path.exists():
        spa_df = pd.read_csv(spa_path)
        print("=== Prueba global SPA (post-hoc, B=500) ===")
        display(spa_df)
else:
    print("Inferencia estadística y tabla MCS: PENDIENTE DE ESTADO B.")""")

    # 11. Multiplicidad y diseno efectivo
    md("""## 11. Diseño efectivo, dependencia temporal y análisis de errores por planta

Se evalúa $T_{\\text{eff}} \\ge 2H$ (excluido de inferencia confirmatoria en `13/28` escenarios principalmente de horizontes `21d` y `28d`) y se resume la distribución de errores por planta (`17` centrales entre `9.8` y `388.0 MW`).""")

    code("""if bundle is not None:
    design_df = analyze_effective_design(bundle)
    display(design_df)
    plant_summary = (
        bundle["tabla_por_planta"]
        .groupby("code", as_index=False)
        .agg(
            n_escenarios=("escenario", "count"),
            mae_ashr_medio=("MAE_ASHR", "mean"),
            mae_mm30_medio=("MAE_MM30", "mean"),
            escenarios_mejor_que_mm30=("diff_vs_MM30", lambda s: int((s < 0).sum())),
        )
        .sort_values("mae_ashr_medio", ascending=False)
    )
    display(plant_summary)
else:
    design_df = pd.DataFrame()
    print("Diagnóstico de diseño efectivo T_eff: PENDIENTE DE ESTADO B.")""")

    # 12. Evaluador formal R3
    md("""## 12. Evaluador formal de superioridad (R3)

Evalúa por separado las cuatro condiciones exigidas (`CUMPLE`, `NO_CUMPLE`, `NO_VERIFICABLE`, `PENDIENTE_AUTORIZACION`), manteniendo un veredicto estrictamente conservador.""")

    code("""if bundle is not None:
    r3_verdict_df = evaluate_r3_superiority(bundle, scope_authorized=SCOPE_AUTHORIZED, alpha=0.05)
    display(r3_verdict_df)
else:
    r3_verdict_df = pd.DataFrame()
    print("Evaluador formal R3: PENDIENTE DE ESTADO B.")""")

    # 13. Ablaciones
    md("""## 13. Estudios diagnósticos complementarios: ablación controlada (`D+1` real)

Estudio diagnóstico ejecutado sobre el panel real `D+1` con partición cronológica estricta (`max_train_date < min_val_date`, sin fuga temporal). No sustituye el OOF confirmatorio de 28 escenarios; aísla la contribución marginal de la normalización a factor de capacidad (`CF`), la pérdida de Huber, la regularización estructurada y las restricciones de monotonía.""")

    code("""abl_path = OOF_DIR / "ablaciones" / "tabla_ablaciones.csv"
abl_df = pd.DataFrame()
if bundle is not None and abl_path.exists():
    abl_df = pd.read_csv(abl_path)
    display(abl_df)
    plot_ablation_bar(abl_df, FIG_DIR)
else:
    print("Tabla de ablaciones: PENDIENTE (ejecutar `python -m sm_cfhr.cli report`).")""")

    # 14. Sensibilidad y robustez
    md("""## 14. Sensibilidad de hiperparámetros, semillas, estrés ante atípicos y escalabilidad (`D+1` real)

Incluye barridos 1D de `delta`, `alpha_lin` y `alpha_meteo`, estabilidad en 10 semillas (`2026..2035`), contaminación sintética sobre validación limpia (`0%`, `5%`, `10%` de atípicos) y curvas de aprendizaje/tiempo de ajuste.""")

    code("""sens_path = OOF_DIR / "sensibilidad" / "tabla_sensibilidad_hp.csv"
seed_path = OOF_DIR / "sensibilidad" / "tabla_estabilidad_semillas.csv"
stress_path = OOF_DIR / "robustez" / "tabla_estres_atipicos.csv"
lc_path = OOF_DIR / "robustez" / "tabla_aprendizaje_escalabilidad.csv"
if bundle is not None and sens_path.exists() and stress_path.exists() and lc_path.exists():
    sens_df = pd.read_csv(sens_path)
    stress_df = pd.read_csv(stress_path)
    lc_df = pd.read_csv(lc_path)
    display(sens_df)
    if seed_path.exists():
        seed_df = pd.read_csv(seed_path)
        display(seed_df.groupby("variant", as_index=False)["mae_macro_kwh"].agg(["mean", "std", "min", "max"]))
    display(stress_df)
    display(lc_df)
    plot_hp_sensitivity(sens_df, FIG_DIR)
    plot_robustness_stress(stress_df, FIG_DIR)
    plot_learning_and_scalability(lc_df, FIG_DIR)
else:
    print("Estudios de sensibilidad y robustez: PENDIENTE (ejecutar `python -m sm_cfhr.cli report`).")""")

    # 15. Incertidumbre predictiva
    md("""## 15. Incertidumbre predictiva: intervalos conformes temporales (`D+1` real)

Cuantiles conformes de muestra finita calibrados estrictamente con residuos pasados (`val_cal` anterior en fecha a `val_test`) para coberturas nominales **80 %** y **90 %**, reportando cobertura empírica real y subcobertura ante cambio de régimen temporal.""")

    code("""conf_path = OOF_DIR / "incertidumbre" / "tabla_conformal_resumen.csv"
conf_plant_path = OOF_DIR / "incertidumbre" / "tabla_conformal_por_planta.csv"
if bundle is not None and conf_path.exists():
    conf_df = pd.read_csv(conf_path)
    display(conf_df)
    if conf_plant_path.exists():
        display(pd.read_csv(conf_plant_path))
    plot_conformal_coverage(conf_df, FIG_DIR)
else:
    print("Calibración conformal temporal: PENDIENTE (ejecutar `python -m sm_cfhr.cli report`).")""")

    # 16. Interpretabilidad
    md("""## 16. Interpretabilidad global, restricciones monótonas y descomposición aditiva local

En `SM-CFHR` (modelo paramétrico lineal en predictores estandarizados), la descomposición $\\phi_{q,j} = \\hat{\\sigma}_c (z_{q,j} - \\bar{z}_{\\mathrm{train},j})\\hat{w}_j$ es **exacta** por instancia y verifica directamente las restricciones monótonas $\\hat{w}_{\\mathrm{cf30}} \\ge 0$ y $\\hat{w}_{\\mathrm{rad\\_pron}} \\ge 0$.""")

    code("""interp_path = OOF_DIR / "interpretabilidad" / "tabla_importancia_global.csv"
local_path = OOF_DIR / "interpretabilidad" / "ejemplos_locales.json"
if bundle is not None and interp_path.exists():
    interp_df = pd.read_csv(interp_path)
    mono_ok = bool((interp_df.loc[interp_df["feature"].isin(["cf30", "rad_pron"]), "std_coef"] >= 0.0).all())
    print("Restricciones monótonas (cf30 >= 0 y rad_pron >= 0) verificadas:", mono_ok)
    display(interp_df)
    if local_path.exists():
        display(pd.DataFrame(json.loads(local_path.read_text(encoding="utf-8"))).drop(columns=["attributions"], errors="ignore"))
    plot_feature_importance(interp_df, FIG_DIR)
else:
    print("Interpretabilidad global/local: PENDIENTE (ejecutar `python -m sm_cfhr.cli report`).")""")

    # 17. Coste computacional
    md("""## 17. Coste computacional medido y escalabilidad paralela

Se distinguen explícitamente:
- **Tiempo de pared real:** `SM-CFHR-Fixed` (`499.748 s` = `8.33 min`, `8` workers, `241` reciclajes, RSS agregado pico `2,067.62 MB`); `SM-CFHR-HPO` (`29,550.293 s` = `8.21 h` acumuladas en dos sesiones con reanudación verificada tras traslado, `12` workers, `228` reciclajes, RSS agregado pico `4,201.10 MB`).
- **Suma de tiempos acumulados por escenario (`tabla_coste`):** suma del tiempo de cómputo de los workers en paralelo (supera al tiempo de pared).
- **Benchmark separado de escalabilidad SMT (`48` unidades de desarrollo, `25` trials):** `8w` (`70.064 s`, `41.11 u/min`), `10w` (`65.841 s`, `+6.41%`), `12w` (`63.360 s`, `+10.58%`, `3,465.09 MB` RSS agregado, `53.1 °C`). *Fixed (8w) y HPO (12w) no constituyen una comparación controlada a igualdad de paralelismo.*""")

    code("""if bundle is not None:
    m_fixed_p = ROOT / "results" / "oof_v1_0_0" / "oof_v1_0_3_fixed" / "consolidated" / "manifest_ashr_oof.json"
    m_hpo_p = ROOT / "results" / "oof_v1_0_0" / "oof_v1_0_3_hpo" / "consolidated" / "manifest_ashr_oof.json"
    cost_summary = pd.DataFrame([
        {"variante": "SM-CFHR-Fixed (B=0)", "workers": 8, "unidades": 14625, "tiempo_pared_s": 499.748, "tiempo_pared_min": round(499.748 / 60.0, 2), "reciclajes": 241, "rss_agregado_pico_mb": 2067.62},
        {"variante": "SM-CFHR-HPO (B=25, 2 sesiones)", "workers": 12, "unidades": 14625, "tiempo_pared_s": 29550.293, "tiempo_pared_min": round(29550.293 / 60.0, 2), "reciclajes": 228, "rss_agregado_pico_mb": 4201.10},
    ])
    display(cost_summary)
    if m_hpo_p.exists():
        m_hpo = json.loads(m_hpo_p.read_text(encoding="utf-8"))
        t_hpo = m_hpo.get("tiempos_por_escenario_s", {})
        c_df = bundle["tabla_coste"].copy().rename(columns={"tiempo_s": "tiempo_acum_workers_fixed_s"})
        c_df["tiempo_acum_workers_hpo_s"] = c_df["scenario_id"].map(lambda s: float(t_hpo.get(f"hpo__{s}", np.nan)))
        display(c_df)""")

    # 18. Exportacion integral y discusion
    md("""## 18. Exportación integral automática y resumen descriptivo""")

    code("""resumen_desc = build_descriptive_summary(bundle)
print(json.dumps(resumen_desc, indent=2, ensure_ascii=False))
if bundle is not None:
    exported_paper = export_paper_tables(bundle, TAB_DIR)
    full_manifest = export_full_scientific_artifacts(
        bundle,
        OOF_DIR,
        extra_artifacts={
            "conclusiones/veredicto_r3": r3_verdict_df,
            "inferencia/diseno_efectivo": design_df,
            "conclusiones/resumen_descriptivo": resumen_desc,
        },
    )
    print("Exportación completa:", full_manifest["n_archivos_exportados"], "archivos con SHA-256.")""")

    # 19. Conclusiones y reproducibilidad
    md("""## 19. Conclusiones científicas verificadas, limitaciones y reproducibilidad

1. **Desempeño empírico real (`SM-CFHR-Fixed`, $B=0$):** obtiene rango medio **`2.93 / 23`**, **`18 / 28`** victorias descriptivas frente al mejor modelo de biblioteca (`22 / 28` frente al mejor modelo custom, `15 / 28` frente a `MM30`) y permanece en el MCS ($\alpha=0{,}10$) en **`27 / 28`** escenarios. Destaca especialmente en **generalización espaciotemporal (`ET`)** con meteorología (`D1 ET 5 vars`: `-22.16%` vs mejor biblioteca; `RecGEFSv12 ET 7–28d`: rango medio `1.0` y `-11.84%` medio), mientras que pierde sistemáticamente en **`DirectoSinClima` intraplanta** (`+13.27%` medio, hasta `+24.24%` en `28d`).
2. **`SM-CFHR-Fixed` vs. `SM-CFHR-HPO` ($B=25$):** sobre el soporte común alineado, `HPO` mejora a `Fixed` únicamente en **`9 / 28`** escenarios (todos `intraplanta`), mientras `Fixed` gana en `19 / 28` (`14 / 14` en `ET`) con mejor rango medio (`2.93` vs `3.86`) y un coste computacional **`59×` menor en tiempo de pared** (`8.33 min` vs `8.21 h`). Esto respalda el uso de `SM-CFHR-Fixed` como configuración principal parsimoniosa.
3. **Inferencia estadística y veredicto R3:** solo **`2 / 15`** escenarios con $T_{\\text{eff}} \\ge 2H$ resultan significativos tras Holm–Bonferroni frente al mejor modelo de biblioteca y `0 / 15` superan $|d_{\\mathrm{Cohen}}| \\ge 0{,}20$. En consecuencia, **no se declara superioridad estadística general** (`R3`: `Condicion_1: NO_CUMPLE`, `Condicion_2: NO_VERIFICABLE`, `Condicion_3: NO_CUMPLE`, `Condicion_4: PENDIENTE_AUTORIZACION`).
4. **Estudios diagnósticos complementarios (`D+1` real):** la normalización adimensional por capacidad instalada (`CF`) es el componente estructural más determinante (`0.0349` vs `0.2306` CF al usar energía bruta); la pérdida de Huber preserva estabilidad ante `5%–10%` de contaminación (`0.03488 -> 0.03491` CF vs `0.03496 -> 0.03749` en `L2`); las restricciones monótonas (`cf30 >= 0`, `rad_pron >= 0`) se cumplen con `cf30` y `anom_rad` como principales predictores; y los intervalos conformes temporales muestran subcobertura moderada (`83.2%` para `90%` nominal; `69.2%` para `80%` nominal) atribuible a deriva temporal entre bloques.
5. **Reproducibilidad sin reentrenar:** ejecutar `python -m sm_cfhr.cli report` y luego `python scripts/evaluacion/generar_notebook_ashr.py` + `jupyter nbconvert --to notebook --execute --inplace notebooks/actuales/evaluacion_cientifica_ashr_oof.ipynb`.""")

    nb["cells"] = cells
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out_path)
    return out_path


if __name__ == "__main__":
    p = Path(__file__).resolve().parents[2] / "notebooks" / "actuales" / "evaluacion_cientifica_ashr_oof.ipynb"
    print("Generated:", build_notebook(p))
