"""Generator script for the definitive 20-section scientific notebook of SM-CFHR / ASHR."""

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

Este cuaderno constituye el documento computacional central de la evaluación científica de **SM-CFHR** (*Structured Monotone Capacity-Factor Huber Regressor*). Opera en tres estados explícitos sin requerir intervención manual:
- **Estado A (`PENDING`):** sin resultados definitivos en `resultados/evaluacion_ashr_oof/`; presenta formulación, protocolo, configuración congelada, compatibilidad `scikit-learn` e inventario sin ejecutar OOF ni mostrar cifras ficticias.
- **Estado B (`VALID`):** con manifiesto válido y 28 escenarios completos; carga, verifica integridad, ejecuta inferencia estadística, evalúa R3, construye tablas/figuras y exporta todo automáticamente.
- **Estado C (`INVALID` / parcial):** detecta ausencias o inconsistencias de hash/esquema y bloquea conclusiones confirmatorias.""")

    # 2. Introduccion y datos causales
    md("""## 1. Introducción, alcance y datos causales

- **Alcance exclusivo:** 17 plantas fotovoltaicas de Colombia (registros XM y pronósticos meteorológicos Open-Meteo / GEFSv12) organizadas en **28 escenarios oficiales** (`4 D+1`, `8 Recursivo sin clima`, `8 Directo sin clima`, `8 Recursivo + GEFSv12`).
- **Nota de alcance académico:** No se incorporan datasets externos de otros dominios; la sustitución por los 28 escenarios solares queda documentada como modificación de alcance pendiente de validación docente explícita.""")

    code("""import json
import sys
from pathlib import Path
from IPython.display import display
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Soporte limpio para paquete instalado (`pip install -e .`) o ejecucion desde raiz del repo
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
    apply_holm,
    audit_inner_splits,
    benjamini_hochberg_adjust,
    bootstrap_ci_difference,
    build_causal_inner_splits,
    clark_west_test,
    cliffs_delta,
    cohens_d_paired,
    compute_interpretability_summary,
    compute_temporal_conformal_intervals,
    evaluate_r3_superiority,
    export_full_scientific_artifacts,
    giacomini_white_audited,
    model_confidence_set,
    run_ablation_suite,
    run_hp_sensitivity,
    run_learning_and_scalability_curves,
    run_noise_outlier_stress,
    run_seed_stability_hpo,
    spa_test,
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
    build_frozen_ashr,
    export_paper_tables,
    load_frozen_config,
    load_oof_bundle,
    validate_oof_artifacts,
)
from src.ashr_oof_pipeline.oof_runner import (
    align_predictions,
    extract_cf_features,
    load_official_benchmarks,
    run_oof_definitivo,
)

CFG = load_frozen_config(ROOT)
CONTRATO_RUTAS = CFG.get("rutas_contrato", {})
OOF_DIR = ROOT / CONTRATO_RUTAS.get("oof_dir", "resultados/evaluacion_ashr_oof")
HPO_TRIALS_DIR = ROOT / CONTRATO_RUTAS.get("hpo_trials_dir", "resultados/evaluacion_ashr_oof/hpo_trials")
FIG_DIR = OOF_DIR / "figuras"
TAB_DIR = OOF_DIR / "tablas_paper"
SCOPE_AUTHORIZED = bool(CFG.get("autorizacion_alcance_solar_docente", False))

print("Paquete sm_cfhr versión:", sm_cfhr.__version__)
print("Denominación científica:", CFG["denominacion_cientifica"])
print("SHA-256 núcleo congelado:", CFG["sha256_src_adaptive_structured_huber"])""")

    # 3. Literatura y R1
    md("""## 2. Revisión de literatura y novedad acotada (R1)

El protocolo formal de literatura (`docs/auditoria_entregable3/AUDITORIA_R1_R2_R3.md`) documenta 24 exportaciones de **Web of Science Core Collection** (`fuentes/bibliografia/wos/`), análisis de 8 trabajos cercanos (Pico et al. 2026; Trigo-González et al. 2019; Almeida et al. 2015; Yang et al. 2020; Huber 1964; Owen 2007; K1/W1) y *snowballing*.

**Afirmación acotada de novedad:** Formulación paramétrica convexa propia en espacio adimensional de factor de capacidad (`CF`) que combina pérdida Huber con escala MAD, regularización $L_2$ diferenciada por familia y restricciones físicas monótonas (`cf30 >= 0`, `rad_pron >= 0`), optimizada con L-BFGS-B analítico propio en NumPy/SciPy.""")

    # 4. Formulacion matematica
    md(r"""## 3. Formulación matemática y propiedades de `SM-CFHR`

Sea $\mathrm{cap}_i > 0$ la capacidad instalada previa al origen y $B_i = \mathrm{cap}_i \cdot 24{,}000\;\text{kWh/día}$. En diarios/recursivos $c_i = \operatorname{clip}(y_i/B_i, 0, 1)$; en directos acumulados $c_i = \operatorname{clip}(Y_{H,i}/(H B_i), 0, 1)$.

Con respuesta robusta $\tilde{c}_i = (c_i - m_c)/\hat{\sigma}_c$ ($\hat{\sigma}_c = \max(1.4826\,\mathrm{MAD}(c), 10^{-6})$) y predictores estandarizados $z_i$:

$$\min_{\theta_0 \in \mathbb{R},\; \theta \in \mathcal{C}} \; \frac{1}{n}\sum_{i=1}^n \rho_\delta\!\left(\tilde{c}_i - \theta_0 - z_i^\top \theta\right) + \frac{1}{2}\sum_{j \in \mathcal{G}_{\mathrm{lin}}} \alpha_{\mathrm{lin}}\,\theta_j^2 + \frac{1}{2}\sum_{j \in \mathcal{G}_{\mathrm{meteo}}} \alpha_{\mathrm{meteo}}\,\theta_j^2$$

sujeto a $\mathcal{C} = \{\theta : \theta_j \ge 0 \;\forall j \in \mathcal{M}_{\mathrm{pos}}\}$.""")

    # 5. Paquete sm_cfhr e integracion sklearn (R2)
    md("""## 4. Paquete `sm_cfhr` e integración `scikit-learn` (R2)

`SMCFHRRegressor` y `AdaptiveStructuredHuberRegressor` heredan de `(BaseEstimator, RegressorMixin)` sin delegar el algoritmo en estimadores de `scikit-learn`. A continuación se muestra la interfaz pública estándar (`clone`, `Pipeline`, `get_params` y serialización `joblib`, sin ejecutar OOF).""")

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

    # 6. Optuna TPE
    md("""## 5. Optimización bayesiana Optuna TPE y espacio de búsqueda

Se comparan dos variantes del mismo núcleo fijo (`n_knots=0, max_interactions=0, anchor="none", structured=True, use_mono=True`):
- **`SM-CFHR-Fixed` ($B=0$):** `delta=1.35, alpha_lin=0.01, alpha_meteo=0.10`.
- **`SM-CFHR-HPO` ($B=25$):** Optuna `TPESampler` sobre `delta ∈ [0.5, 3.0]`, `alpha_lin ∈ [1e-4, 1.0]` (log) y `alpha_meteo ∈ [1e-3, 10.0]` (log, vinculado a `alpha_lin` cuando no hay meteorología), optimizando exactamente `MAE_macro_kWh` dentro de CV causal anidada.""")

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
        display(hpo_summary_df)
        plot_optuna_convergence(hpo_trials_df, FIG_DIR)
else:
    print("Logs de trials Optuna TPE: PENDIENTE DE EJECUCIÓN OOF HPO.")""")

    # 7. Nested CV
    md("""## 6. Validación cruzada causal anidada temporal y espacial

- **D+1 (`d1`):** ventana creciente con embargo (`30d` cuando el historial lo permite), $\\max(\\text{date}_{\\text{train}}) < \\min(\\text{date}_{\\text{val}})$.
- **Directo (`direct`):** $\\max(\\text{t\\_end\\_target}_{\\text{train}}) \\le \\min(\\text{t\\_origin}_{\\text{val}})$.
- **Recursivo (`recursive`):** $\\max(\\text{origin}_{\\text{train}}) + H \\le \\min(\\text{origin}_{\\text{val}})$.
- **Espaciotemporal (`ET`):** retención de grupo espacial completo en validación interna (`spatial_disjoint=True`).""")

    code("""nested_audit_path = OOF_DIR / "particiones" / "nested_cv_audit.csv"
nested_audit_df = pd.DataFrame()
if nested_audit_path.exists():
    nested_audit_df = pd.read_csv(nested_audit_path)
    display(nested_audit_df)
    plot_nested_cv_scheme(nested_audit_df.head(30), FIG_DIR)
else:
    print("Trazabilidad de folds de Nested CV definitiva: PENDIENTE DE EJECUCIÓN OOF.")""")

    # 8. Protocolo e inventario 28 escenarios y 22 benchmarks
    md("""## 7. Protocolo experimental e inventario de 28 escenarios y 22 benchmarks

La comparación principal conservadora enfrenta `SM-CFHR-Fixed` ($B=0$) contra los **22 modelos oficiales** (`9` librerías, `12` custom, `MM30`; `CF-BRH` y `R-ILWR-PA v3` excluidos). `SM-CFHR-HPO` ($B=25$) se reporta en paralelo declarando la asimetría frente a rejillas de 4 puntos.""")

    code("""inv_df = verify_benchmark_inventory(ROOT)
print(f"Escenarios verificados: {len(inv_df)}/28 | Archivos OK: {inv_df['file_exists'].all()} | Columnas OK: {inv_df['cols_ok'].all()} | CF-BRH excluido: {inv_df['cf_brh_excluded'].all()}")
display(inv_df)""")

    # 9. Carga y deteccion Estado A / B / C
    md("""## 8. Detección de estado (Estado A / B / C) y validación de integridad

Esta celda **nunca** dispara el entrenamiento OOF (`execute=False`). Verifica manifiesto, hashes SHA-256 y presencia de los 28 escenarios.""")

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
elif state == "INVALID":
    print("ESTADO C (INCONSISTENTE / PARCIAL): se bloquean conclusiones confirmatorias.")
else:
    print("ESTADO A (PENDIENTE): ejecutar posteriormente `python scripts/evaluacion/ejecutar_oof_definitivo_ashr.py --ejecutar-oof-definitivo`.")""")

    # 10. Resultados OOF globales y por escenario
    md("""## 9. Resultados OOF globales, por escenario, régimen y planta

Se muestran las tablas completas (sin truncamiento como única vista), separando explícitamente los escenarios diarios/recursivos (`kWh/dia`) de los directos acumulados (`kWh_total`).""")

    code("""if bundle is not None:
    display(bundle["tabla_rankings_regimen"])
    display(bundle["tabla_28_escenarios"])
    display(bundle["tabla_por_planta"])
    if hpo_preds_available:
        fh_rows = []
        for sp in SCENARIOS_28:
            df_f = pd.read_csv(OOF_DIR / "predicciones" / f"{sp.scenario_id}.csv")
            df_h = pd.read_csv(OOF_DIR / "predicciones" / f"{sp.scenario_id}_hpo.csv")
            merged = df_f.merge(df_h, on=list(sp.join_keys), suffixes=("_Fixed", "_HPO"), validate="one_to_one")
            assert len(merged) == len(df_f) == len(df_h), f"Cardinalidad desigual Fixed/HPO en {sp.scenario_id}"
            mae_f = float(np.mean(np.abs(merged[f"{sp.pred_col}_Fixed"] - merged[f"{sp.target_col}_Fixed"])))
            mae_h = float(np.mean(np.abs(merged[f"{sp.pred_col}_HPO"] - merged[f"{sp.target_col}_HPO"])))
            fh_rows.append({"scenario_id": sp.scenario_id, "unit": sp.unit, "n_obs": len(merged), "MAE_Fixed": mae_f, "MAE_HPO": mae_h, "diff_HPO_minus_Fixed": mae_h - mae_f})
        display(pd.DataFrame(fh_rows))
    export_all_figures(bundle, FIG_DIR)
else:
    print("Tablas OOF completas y figuras 1–5: PENDIENTE DE ESTADO B.")""")

    # 11. Inferencia estadistica
    md("""## 10. Inferencia estadística (DM-HLN-HAC, MCS, SPA, GW auditada y Clark–West)

- `DM-HLN-HAC` y `MCS` ($\alpha=0.10$) constituyen las pruebas confirmatorias principales para series dependientes.
- `Giacomini–White` se ejecuta con diagnóstico de autocorrelación lag-1 (`giacomini_white_audited`): cuando `tau=1` y existe autocorrelación residual (`gw_autocorr_warning=True`), se marca como advertencia y se complementa con `SPA`.
- `Clark–West` se aplica exclusivamente en comparaciones anidadas (`SM-CFHR` vs `MM30`).""")

    code("""if bundle is not None:
    t_inf = apply_family_multiplicity(bundle["tabla_inferencia"])
    t_mcs = bundle["tabla_mcs"].copy()
    display(t_inf)
    display(t_mcs)
else:
    print("Inferencia estadística y tabla MCS: PENDIENTE DE ESTADO B.")""")

    # 12. Multiplicidad y diseno efectivo
    md("""## 11. Comparaciones múltiples y análisis de diseño efectivo / potencia

Control FWER mediante **Holm–Bonferroni** por familia preespecificada y control FDR mediante **Benjamini–Hochberg**. Se evalúa $T_{\\text{eff}} \\ge 2H$ y se documentan las limitaciones de potencia espacial/temporal sin aplicar Nadeau–Bengio indebidamente.""")

    code("""if bundle is not None:
    design_df = analyze_effective_design(bundle)
    display(design_df)
else:
    design_df = pd.DataFrame()
    print("Diagnóstico de diseño efectivo T_eff: PENDIENTE DE ESTADO B.")""")

    # 13. Evaluador formal R3
    md("""## 12. Evaluador formal de superioridad (R3 — Sección 6.4)

Evalúa por separado las cuatro condiciones exigidas (`CUMPLE`, `NO_CUMPLE`, `NO_APLICABLE`, `NO_VERIFICABLE`, `PENDIENTE_AUTORIZACION`, `EVIDENCIA_INSUFICIENTE`).""")

    code("""if bundle is not None:
    r3_verdict_df = evaluate_r3_superiority(bundle, scope_authorized=SCOPE_AUTHORIZED, alpha=0.05)
    display(r3_verdict_df)
else:
    r3_verdict_df = pd.DataFrame()
    print("Evaluador formal R3: PENDIENTE DE ESTADO B (autorización alcance solar:", SCOPE_AUTHORIZED, ").")""")

    # 14. Ablaciones
    md("""## 13. Estudios de ablación controlada

Evalúa sobre particiones válidas el aporte marginal de cada componente: `SM-CFHR completo`, `sin Huber (L2)`, `sin L2 estructurada`, `sin monotonía`, `STD vs MAD` y `energía bruta vs factor de capacidad`.""")

    code("""abl_path = OOF_DIR / "ablaciones" / "tabla_ablaciones.csv"
abl_df = pd.DataFrame()
if bundle is not None and abl_path.exists():
    abl_df = pd.read_csv(abl_path)
    display(abl_df)
    plot_ablation_bar(abl_df, FIG_DIR)
else:
    print("Tabla de ablaciones OOF: PENDIENTE DE ESTADO B / ejecución de `run_ablation_suite`.")""")

    # 15. Sensibilidad y robustez
    md("""## 14. Sensibilidad de hiperparámetros, semillas, robustez, aprendizaje y escalabilidad

Incluye barridos 1D de `delta`, `alpha_lin`, `alpha_meteo`, estabilidad en 10 semillas (`2026..2035`), contaminación con ruido/outliers, curvas de aprendizaje (`10%, 25%, 50%, 100%`) y memoria pico.""")

    code("""sens_path = OOF_DIR / "sensibilidad" / "tabla_sensibilidad_hp.csv"
stress_path = OOF_DIR / "robustez" / "tabla_estres_atipicos.csv"
lc_path = OOF_DIR / "robustez" / "tabla_aprendizaje_escalabilidad.csv"
if bundle is not None and sens_path.exists() and stress_path.exists() and lc_path.exists():
    sens_df = pd.read_csv(sens_path)
    stress_df = pd.read_csv(stress_path)
    lc_df = pd.read_csv(lc_path)
    display(sens_df)
    display(stress_df)
    display(lc_df)
    plot_hp_sensitivity(sens_df, FIG_DIR)
    plot_robustness_stress(stress_df, FIG_DIR)
    plot_learning_and_scalability(lc_df, FIG_DIR)
else:
    print("Estudios de sensibilidad y robustez OOF: PENDIENTE DE ESTADO B.")""")

    # 16. Incertidumbre predictiva
    md("""## 15. Incertidumbre predictiva: intervalos conformes temporales

Cuantiles conformes de muestra finita para niveles nominales **80 %** y **90 %**, cobertura empírica global, por planta y por horizonte, y ancho medio.""")

    code("""conf_path = OOF_DIR / "incertidumbre" / "tabla_conformal_resumen.csv"
if bundle is not None and conf_path.exists():
    conf_df = pd.read_csv(conf_path)
    display(conf_df)
    plot_conformal_coverage(conf_df, FIG_DIR)
else:
    print("Calibración conformal OOF: PENDIENTE DE ESTADO B.")""")

    # 17. Interpretabilidad
    md("""## 16. Interpretabilidad global y local

Coeficientes estandarizados, importancia global $|w_j (z_j - \\bar{z}_j)|$ y descomposición aditiva exacta por instancia (distinta de SHAP general sobre modelos no lineales arbitrarios).""")

    code("""interp_path = OOF_DIR / "interpretabilidad" / "tabla_importancia_global.csv"
if bundle is not None and interp_path.exists():
    interp_df = pd.read_csv(interp_path)
    display(interp_df)
    plot_feature_importance(interp_df, FIG_DIR)
else:
    print("Interpretabilidad global/local OOF: PENDIENTE DE ESTADO B.")""")

    # 18. Coste computacional
    md("""## 17. Coste computacional medido

Registra tiempos reales medidos por escenario, fits efectivos, trials Optuna y memoria pico (sin confundir estimaciones previas con mediciones reales).""")

    code("""if bundle is not None:
    display(bundle["tabla_coste"])
    print("Tiempo total medido (s):", bundle["manifest"].get("tiempo_total_s", "N/A"))
else:
    print("Coste computacional real: PENDIENTE DE ESTADO B.")""")

    # 19. Exportacion integral y discusion
    md("""## 18. Exportación integral automática, discusión y limitaciones""")

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

    # 20. Conclusiones y reproducibilidad
    md("""## 19. Conclusiones e integración con Jupyter Book

- Todas las conclusiones numéricas definitivas permanecen bloqueadas hasta ejecutar `--ejecutar-oof-definitivo` (**Estado B**).
- Las tablas exportadas en `resultados/evaluacion_ashr_oof/tablas_paper/` y figuras en `resultados/evaluacion_ashr_oof/figuras/` mantienen nombres estables para su importación directa en `book/`.""")

    nb["cells"] = cells
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out_path)
    return out_path


if __name__ == "__main__":
    p = Path(__file__).resolve().parents[2] / "notebooks" / "actuales" / "evaluacion_cientifica_ashr_oof.ipynb"
    print("Generated:", build_notebook(p))
