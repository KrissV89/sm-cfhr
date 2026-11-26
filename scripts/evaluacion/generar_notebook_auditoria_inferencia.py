"""Generator for the complementary statistical inference audit notebook (v2).

Creates `notebooks/actuales/auditoria_inferencia_estadistica.ipynb` consuming
`resultados/inferencia_revision_independiente_v2/` without retraining or modifying
frozen OOF predictions.
"""

from __future__ import annotations

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
NB_PATH = ROOT / "notebooks" / "actuales" / "auditoria_inferencia_estadistica.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src.strip() + "\n")


def build_notebook() -> nbf.NotebookNode:
    cells = []

    cells.append(md("""
# Auditoría Independiente, Reconciliación de Estimandos y Corrección Estadística (v2) — SM-CFHR

**Propósito:** Reconciliar el estimando del ranking (`MAE_macro` por planta) con la inferencia estadística (`DM-HLN-HAC`, bootstrap sincronizado, `SPA` y `MCS`), documentar la política de casos degenerados, explicar las divergencias frente a `arch 8.0.0`, incorporar el verdadero `MAE_CF_macro` con capacidad instalada variante en el tiempo y presentar análisis exploratorios P2 sobre las predicciones OOF congeladas (`28` escenarios, `14,625` unidades por variante), **sin reentrenar modelos ni alterar `resultados/evaluacion_ashr_oof/`**.

---

## 1. Reconciliación de estimandos (`MAE_macro` vs. promedio por origen) y cambios de signo

En el flujo OOF congelado, el ranking ordena los modelos por `MAE_macro` (media aritmética sobre las `17` plantas), mientras que las pruebas temporales (`DM-HLN`, `MCS`, `SPA`) promediaban las pérdidas por fecha/origen. Debido al desbalance del panel (`1` a `17` plantas por origen), ambas agregaciones ponderan distinto y producen **4 inversiones de signo** entre $\\Delta\\mathrm{MAE}_{\\mathrm{macro}}$ y $\\Delta\\mathrm{MAE}_{\\mathrm{origen}}$.
"""))

    cells.append(code("""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

_CUR = Path.cwd().resolve()
ROOT = next(p for p in (_CUR, *_CUR.parents) if (p / "resultados" / "inferencia_revision_independiente_v2").exists())
REV_DIR = ROOT / "resultados" / "inferencia_revision_independiente_v2"

meta = json.loads((REV_DIR / "metadatos_revision_v2.json").read_text(encoding="utf-8"))
rec = pd.read_csv(REV_DIR / "02_estimandos_y_dm_hln" / "reconciliacion_estimandos_28_escenarios.csv")
all9 = pd.read_csv(REV_DIR / "02_estimandos_y_dm_hln" / "comparacion_todas_9_bibliotecas_macro.csv")
ovl = pd.read_csv(REV_DIR / "02_estimandos_y_dm_hln" / "verificacion_no_solapamiento_28_escenarios.csv")
spa = pd.read_csv(REV_DIR / "03_spa_y_mcs_reconciliados" / "spa_reconciliacion_28_escenarios.csv")
mcs = pd.read_csv(REV_DIR / "03_spa_y_mcs_reconciliados" / "mcs_reconciliacion_28_escenarios.csv")
sens_teff = pd.read_csv(REV_DIR / "04_sensibilidad" / "sensibilidad_umbral_teff_macro_y_origen.csv")
met = pd.read_csv(REV_DIR / "05_metricas_corregidas" / "metricas_completas_28_escenarios_cf_y_kwh.csv")
cap_chg = pd.read_csv(REV_DIR / "05_metricas_corregidas" / "plantas_cambio_capacidad.csv")

print(f"Versión v2: {meta['version']} | B_SPA/MCS={meta['B_spa_mcs']} | B_boot={meta['B_bootstrap_macro']}")
mb = rec[rec["tipo_referencia"] == "Mejor_Bib_Retrospectivo"]
flips = mb[mb["sign_flip_macro_vs_origin"]][[
    "escenario", "modelo_referencia", "diff_MAE_macro", "diff_temporal_origen",
    "dm_hln_macro_v2", "p_dm_macro_v2", "p_holm_macro_fam", "boot_macro_ci95_lo", "boot_macro_ci95_hi"
]]
print(f"Inversiones de signo reproducidas: {len(flips)}/28")
display(flips)
"""))

    cells.append(md("""
**Especificación inferencial (Sección 1):**
- **Métrica y estimando:** $\\Delta\\mathrm{MAE}_{\\mathrm{macro}}$ en kWh (`SM-CFHR-Fixed - Mejor_Bib`), descompuesto exactamente en la serie temporal macro-alineada $z_t^{\\mathrm{macro}}$ cuya media coincide idénticamente con $\\Delta\\mathrm{MAE}_{\\mathrm{macro}}$.
- **Unidad estadística y procedimiento:** Orígenes/fechas ($T \\in [31, 262]$), `dm_hln_hac_v2` + `macro_synchronized_block_bootstrap` (`B=2000`) en `inference_v2` v2.0.0.
- **Limitaciones:** `Mejor_Bib` es retrospectivo; ninguno de los 4 escenarios con cambio de signo resulta significativo ($p \\ge 0.28$).

---

## 2. DM-HLN-HAC sobre `MAE_macro`, no solapamiento y referencias preespecificadas

Verificamos en los `28` escenarios que `median_step_days == horizon_days` (`h_overlap_real == 1`, bloques disjuntos), mantenemos HAC ante autocorrelación residual y contrastamos tanto `Mejor_Bib` como las referencias preespecificadas (`MM30` y `SM-CFHR-HPO`) y todas las `9` bibliotecas por separado.
"""))

    cells.append(code("""
print("=== Escenarios significativos tras Holm por suite sobre MAE_macro (vs Mejor_Bib retrospectivo) ===")
display(mb[mb["p_holm_macro_fam"] < 0.05][[
    "escenario", "modelo_referencia", "diff_MAE_macro", "dm_hln_macro_v2",
    "p_dm_macro_v2", "p_holm_macro_fam", "T_eff_macro", "nivel_fiabilidad_teff",
    "boot_macro_ci95_lo", "boot_macro_ci95_hi"
]])

print("=== Referencias preespecificadas sobre MAE_macro (Holm por suite) ===")
for ref_t in ["MM30_Preespecificado", "SM_CFHR_HPO_Preespecificado"]:
    sub = rec[rec["tipo_referencia"] == ref_t]
    print(f"  {ref_t}: victorias desc.={(sub['diff_MAE_macro']<0).sum()}/28 | Holm sig favor={( (sub['p_holm_macro_fam']<0.05)&(sub['diff_MAE_macro']<0) ).sum()}/28 | Holm sig contra={( (sub['p_holm_macro_fam']<0.05)&(sub['diff_MAE_macro']>0) ).sum()}/28")

print("=== Todas las 9 bibliotecas sobre MAE_macro (Holm por escenario m=9 y global m=252) ===")
display(all9.groupby("biblioteca").agg(
    victorias_desc=("diff_MAE_macro", lambda s: int((s < 0).sum())),
    sig_holm_escenario=("p_holm_por_escenario", lambda s: int((s < 0.05).sum())),
    sig_holm_global_252=("p_holm_global_252", lambda s: int((s < 0.05).sum())),
))
"""))

    cells.append(md("""
**Especificación inferencial (Sección 2):**
- **Métrica y estimando:** `MAE_macro` (kWh), serie macro-alineada $z_t^{\\mathrm{macro}}$ (`h_overlap=1`, HAC Newey-West automático).
- **Resultados:** Bajo `MAE_macro` exacto, todos los `5` escenarios `ET + Clima` (`D1`, `7d`, `14d`, `21d`, `28d`) son significativos tras Holm frente a `Mejor_Bib` ($p_{\\mathrm{Holm}} \\le 5.21\\times 10^{-6}$, $T_{\\mathrm{eff,macro}} \\ge 30.82$, fiabilidad estándar). Frente a `MM30` hay `7/28` victorias Holm (`0` derrotas) y frente a `HPO` `4/28` victorias Holm (`1/28` derrota).

---

## 3. Reconciliación de SPA y MCS frente a `arch 8.0.0` (`B=5000`) y sensibilidad de `T_eff`

La auditoría del código fuente de `arch 8.0.0` demostró que `arch.bootstrap.SPA(..., studentize=True)` almacena el argumento pero **no studentiza** en `_simulate_values()` ni en `compute()`, calculando el estadístico no studentizado $\\max_k \\bar{d}_k$. Al alinear esa diferencia y usar $B=5000$ réplicas en `MAE_macro`, tanto SPA como MCS quedan completamente reconciliados.
"""))

    cells.append(code("""
spa_mac = spa[spa["estimando"] == "MAE_macro"]
mcs_mac = mcs[mcs["estimando"] == "MAE_macro"]

print(f"SPA (MAE_macro): max |unstud - arch_def| = {(spa_mac['p_spa_ASHR_unstud'] - spa_mac['p_spa_ASHR_arch_default']).abs().max():.4f}")
print(f"SPA (MAE_macro): max |stud_pr - arch_prestud| = {(spa_mac['p_spa_ASHR_stud_pr_cons'] - spa_mac['p_spa_ASHR_arch_prestud']).abs().max():.4f}")
print(f"MCS (MAE_macro, B=5000): coincidencia exacta de cardinalidad = {(mcs_mac['diff_n_surv_23']==0).sum()}/28 | Jaccard mín = {mcs_mac['jaccard_23_v2_vs_arch'].min():.4f}")
print(f"MCS (MAE_macro): ASHR en MCS-23 = {mcs_mac['ashr_in_mcs23_v2'].sum()}/28 | Único superviviente entre 9 bibliotecas = {((mcs_mac['ashr_in_mcs_bib10_v2'])&(mcs_mac['n_surv_bib10_v2']==1)).sum()}/28")

print("=== Sensibilidad completa del umbral T_eff (sin fijar un único umbral arbitrario) ===")
display(sens_teff)
"""))

    cells.append(md("""
**Especificación inferencial (Sección 3):**
- **Procedimiento y versión:** `spa_test_v2` (kernel Politis-Romano de Hansen 2005, Ec. 7) y `mcs_t_max_v2` ($T_{\\max}$ unilateral, `B=5000`) en `inference_v2` v2.0.0, contrastados frente a `arch 8.0.0`.
- **Estabilidad de `T_eff`:** En `MAE_macro`, cualquier umbral entre `2*h_overlap`, `15`, `20`, `25` y `30` mantiene exactamente `5` victorias significativas tras Holm (`ET + Clima`) y `0` derrotas Holm.

---

## 4. Métricas en Factor de Capacidad real (`MAE_CF_macro`) y análisis exploratorios P2

Separamos el verdadero `MAE_CF_macro` (dividiendo por $24\\,000 \\cdot \\mathrm{cap\\_MW}_{i,t}$, considerando las `6` plantas con ampliaciones de capacidad) de `nMAE_rel_generacion_media_planta`. En `MAE_CF_macro`, la ponderación ya no está dominada por plantas grandes (`GYPO`, 370–388 MW), desplazando el rango medio de `2.93/23` (`kWh`) a `4.00/23` (`CF`) y las victorias frente a `Mejor_Bib` de `18/28` a `16/28`.
"""))

    cells.append(code("""
print("Plantas con cambios temporales de capacidad instalada:")
display(cap_chg)

print(f"Cambios de rango kWh vs CF: {(met['cambio_rango_kWh_vs_CF']!=0).sum()}/28 | Rango medio kWh={met['rango_en_23_kWh'].mean():.2f} vs CF={met['rango_en_23_CF'].mean():.2f}")
display(met[[
    "escenario", "rango_en_23_kWh", "rango_en_23_CF",
    "MAE_macro_kWh_Fixed", "MAE_CF_macro_Fixed", "nMAE_rel_generacion_media_planta_Fixed",
    "diff_pct_kWh_vs_MejorBib", "diff_pct_CF_vs_MejorBib", "Skill_RMSE_vs_MM30_pct"
]].head(12))

for fig_name in [
    "real_vs_predicho_d1.png",
    "real_vs_predicho_multihorizonte_rec.png",
    "real_vs_predicho_directo_acumulado.png",
    "factorial_descriptivo_regimenes.png",
]:
    fp = REV_DIR / "06_exploratorio_p2" / fig_name
    if fp.exists():
        display(Image(filename=str(fp)))
"""))

    cells.append(md("""
## 5. Síntesis de conclusiones verificables

1. **Correspondencia métrica–inferencia:** Con `z_t^macro` y bootstrap sincronizado por bloques temporales, todas las pruebas contrastan el mismo estimando `MAE_macro` que el ranking principal.
2. **Evidencia confirmatoria vs. exploratoria:** La ventaja estadísticamente robusta tras Holm se concentra en **ET + Clima** (`5/28` escenarios sobre `MAE_macro`, todos con $T_{\\mathrm{eff}} \\ge 30.82$), mientras que en **Intraplanta** y especialmente en **`DirectoSinClima_intraplanta`** SM-CFHR pierde competitividad frente a `k-NN` y `BayesianRidge`.
3. **Transparencia de escala:** Al evaluar en `MAE_CF_macro` adimensional, el desempeño relativo se modera (`16/28` victorias, rango medio `4.00/23`), confirmando que parte de la ventaja en `kWh` proviene del peso de las centrales de mayor capacidad en la cohorte.
"""))

    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {
        "display_name": "Python (solarpv-eda)",
        "language": "python",
        "name": "solarpv-eda",
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NB_PATH)
    print(f"Generated notebook: {NB_PATH} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    build_notebook()
