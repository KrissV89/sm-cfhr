"""Integrate verified OOF results ZIP into the SM-CFHR LaTeX paper without retraining.

When no --results-zip or --results-dir is supplied, validates `resultados_pendientes.yaml`
and generates clean LaTeX placeholder tables so the manuscript compiles cleanly without
fabricated empirical numbers.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

EXPECTED_SRC_SHA256 = "27a9fda4fa8b7ef0d95faf9e925e47ade9ed8f25e9ea5d97d6d7cf1c00ddde83"


def _tex_escape(val: Any) -> str:
    s = str(val)
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _write_placeholder_tables(tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)

    (tables_dir / "tab_res_familias.tex").write_text(
        r"""\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llcccc@{}}
\toprule
Familia de escenarios & Régimen & $N_{\text{esc}}$ & Victorias vs.\ Mejor Bib. & Mediana Skill (\%) & Rango mediano (/23) \\
\midrule
$D+1$ (con y sin clima) & ET / IP & 4 & [PENDIENTE] & [PENDIENTE] & [PENDIENTE] \\
Recursivo sin clima (7--28d) & ET / IP & 8 & [PENDIENTE] & [PENDIENTE] & [PENDIENTE] \\
Directo sin clima (7--28d) & ET / IP & 8 & [PENDIENTE] & [PENDIENTE] & [PENDIENTE] \\
Recursivo + GEFSv12 (7--28d) & ET / IP & 8 & [PENDIENTE] & [PENDIENTE] & [PENDIENTE] \\
\midrule
\textbf{Total global} & \textbf{ET + IP} & \textbf{28} & \textbf{[PENDIENTE]} & \textbf{[PENDIENTE]} & \textbf{[PENDIENTE]} \\
\bottomrule
\end{tabular}
""",
        encoding="utf-8",
    )

    (tables_dir / "tab_res_inferencia.tex").write_text(
        r"""\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lccccc@{}}
\toprule
Familia (ref.\ Mejor Bib.) & $N_{\text{vál}}$ ($T_{\text{eff}}\ge 2H$) & DM--HLN $p_{\text{Holm}}<0{,}05$ & MCS (90\%) sup. & SPA $p<0{,}05$ & $|d|$ med. \\
\midrule
$D+1$ & [PENDIENTE] & [PENDIENTE] & [PENDIENTE] & [PENDIENTE] & [PENDIENTE] \\
Recursivo sin clima & [PENDIENTE] & [PENDIENTE] & [PENDIENTE] & [PENDIENTE] & [PENDIENTE] \\
Directo sin clima & [PENDIENTE] & [PENDIENTE] & [PENDIENTE] & [PENDIENTE] & [PENDIENTE] \\
Recursivo + GEFSv12 & [PENDIENTE] & [PENDIENTE] & [PENDIENTE] & [PENDIENTE] & [PENDIENTE] \\
\bottomrule
\end{tabular}
""",
        encoding="utf-8",
    )

    (tables_dir / "tab_res_veredicto_r3.tex").write_text(
        r"""\begin{tabular}{@{}p{3.6cm}p{2.6cm}p{5.5cm}@{}}
\toprule
Condición formal R3 & Estado & Evidencia y criterio aplicado \\
\midrule
C1. Métrica externa & \texttt{NO\_VERIFICABLE} & [PENDIENTE: victorias $\mathrm{MAE}_{\mathrm{macro}}$ en 28 escenarios] \\
C2. Tres pruebas calibradas & \texttt{NO\_VERIFICABLE} & [PENDIENTE: DM--HLN--HAC + Holm, MCS ($\alpha=0{,}10$), SPA] \\
C3. Tamaño del efecto & \texttt{NO\_VERIFICABLE} & [PENDIENTE: $|d| \ge 0{,}20$ o $|\delta_{\mathrm{Cliff}}| \ge 0{,}147$] \\
C4. Alcance y consistencia & \texttt{PENDIENTE\_AUTORIZACION} & 28 escenarios solares (17 plantas); sin datasets externos \\
\bottomrule
\end{tabular}
""",
        encoding="utf-8",
    )

    (tables_dir / "tab_res_ablaciones.tex").write_text(
        r"""\begin{tabular}{@{}lccc@{}}
\toprule
Variante de ablación & $\mathrm{MAE}_{\mathrm{CF}}$ & $\mathrm{MAE}_{\mathrm{kWh}}$ (kWh/día) & $\Delta$ vs.\ completo (\%) \\
\midrule
\texttt{SM-CFHR\_completo} & [PENDIENTE] & [PENDIENTE] & $0{,}00$ \\
\texttt{sin\_huber\_L2} ($\delta=10^6$) & [PENDIENTE] & [PENDIENTE] & [PENDIENTE] \\
\texttt{sin\_l2\_estructurada} & [PENDIENTE] & [PENDIENTE] & [PENDIENTE] \\
\texttt{sin\_monotonia} & [PENDIENTE] & [PENDIENTE] & [PENDIENTE] \\
\texttt{escala\_std\_vs\_mad} & [PENDIENTE] & [PENDIENTE] & [PENDIENTE] \\
\texttt{energia\_bruta\_vs\_cf} & [PENDIENTE] & [PENDIENTE] & [PENDIENTE] \\
\bottomrule
\end{tabular}
""",
        encoding="utf-8",
    )

    (tables_dir / "tab_res_conformal.tex").write_text(
        r"""\begin{tabular}{@{}ccccc@{}}
\toprule
Cobertura nominal & $\alpha$ & Semiancho $q_{1-\alpha}$ & Ancho medio ($2q$) & Cobertura empírica global \\
\midrule
$80\%$ & $0{,}20$ & [PENDIENTE] & [PENDIENTE] & [PENDIENTE] \\
$90\%$ & $0{,}10$ & [PENDIENTE] & [PENDIENTE] & [PENDIENTE] \\
\bottomrule
\end{tabular}
""",
        encoding="utf-8",
    )


def _df_to_latex_rows(df: pd.DataFrame, cols: list[str], max_rows: int = 30) -> str:
    lines = []
    for _, row in df.head(max_rows).iterrows():
        vals = []
        for c in cols:
            v = row.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.4g}" if pd.notna(v) else "--")
            else:
                vals.append(_tex_escape(v))
        lines.append(" & ".join(vals) + r" \\")
    return "\n".join(lines)


def _integrate_from_dir(res_dir: Path, paper_dir: Path) -> None:
    manifest_p = res_dir / "manifest_ashr_oof.json"
    if not manifest_p.exists():
        raise FileNotFoundError(f"No se encontró manifest_ashr_oof.json en {res_dir}")
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    if manifest.get("sha256_src_adaptive_structured_huber") != EXPECTED_SRC_SHA256:
        raise ValueError("El SHA-256 del núcleo en el manifiesto no coincide con v1.0.2 congelado.")
    if manifest.get("estatus") != "OOF_DEFINITIVO_COMPLETO":
        raise ValueError(f"El OOF no está completo: estatus={manifest.get('estatus')}")

    tables_dir = paper_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    r3_p = res_dir / "conclusiones" / "veredicto_r3.csv"
    if r3_p.exists():
        r3_df = pd.read_csv(r3_p)
        rows = _df_to_latex_rows(r3_df, ["condicion", "estado", "evidencia"])
        (tables_dir / "tab_res_veredicto_r3.tex").write_text(
            "\\begin{tabular}{@{}p{3.6cm}p{2.6cm}p{5.5cm}@{}}\n\\toprule\n"
            "Condición formal R3 & Estado & Evidencia y criterio aplicado \\\\\n\\midrule\n"
            f"{rows}\n\\bottomrule\n\\end{{tabular}}\n",
            encoding="utf-8",
        )

    abl_p = res_dir / "ablaciones" / "tabla_ablaciones.csv"
    if abl_p.exists():
        abl_df = pd.read_csv(abl_p)
        base = float(abl_df.loc[abl_df["ablation_variant"] == "SM-CFHR_completo", "mae_kwh"].iloc[0])
        abl_df["delta_pct"] = 100.0 * (abl_df["mae_kwh"] - base) / base
        rows = _df_to_latex_rows(abl_df, ["ablation_variant", "mae_cf", "mae_kwh", "delta_pct"])
        (tables_dir / "tab_res_ablaciones.tex").write_text(
            "\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
            "Variante de ablación & $\\mathrm{MAE}_{\\mathrm{CF}}$ & $\\mathrm{MAE}_{\\mathrm{kWh}}$ & $\\Delta$ (\\%) \\\\\n\\midrule\n"
            f"{rows}\n\\bottomrule\n\\end{{tabular}}\n",
            encoding="utf-8",
        )

    conf_p = res_dir / "incertidumbre" / "tabla_conformal_resumen.csv"
    if conf_p.exists():
        cdf = pd.read_csv(conf_p)
        rows = _df_to_latex_rows(cdf, ["nominal_coverage", "alpha", "conformal_halfwidth", "mean_width", "empirical_coverage"])
        (tables_dir / "tab_res_conformal.tex").write_text(
            "\\begin{tabular}{@{}ccccc@{}}\n\\toprule\n"
            "Cobertura nominal & $\\alpha$ & Semiancho $q$ & Ancho medio & Cobertura empírica \\\\\n\\midrule\n"
            f"{rows}\n\\bottomrule\n\\end{{tabular}}\n",
            encoding="utf-8",
        )

    yaml_p = paper_dir / "resultados_pendientes.yaml"
    data = yaml.safe_load(yaml_p.read_text(encoding="utf-8"))
    data["status_global"] = "INTEGRADO_DESDE_OOF_REAL"
    for item in data.get("placeholders", []):
        csv_rel = item.get("ruta_csv_origen", "")
        if csv_rel and (res_dir / csv_rel).exists():
            item["estado"] = "INTEGRADO"
    yaml_p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrate OOF results ZIP or generate clean placeholder tables.")
    parser.add_argument("--paper-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results-zip", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    yaml_p = args.paper_dir / "resultados_pendientes.yaml"
    if not yaml_p.exists():
        raise FileNotFoundError(f"Falta {yaml_p}")
    meta = yaml.safe_load(yaml_p.read_text(encoding="utf-8"))
    print(f"Placeholders registrados: {len(meta.get('placeholders', []))} | Estado: {meta.get('status_global')}")

    if args.results_zip is None and args.results_dir is None:
        if not args.dry_run:
            _write_placeholder_tables(args.paper_dir / "tables")
        print("Modo placeholder activo: tablas generadas sin cifras ficticias.")
        return

    if args.dry_run:
        print("Dry-run completado sin modificar archivos.")
        return

    if args.results_zip is not None:
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(args.results_zip, "r") as zf:
                zf.extractall(tmp)
            tmp_root = Path(tmp)
            candidates = list(tmp_root.rglob("manifest_ashr_oof.json"))
            if not candidates:
                raise FileNotFoundError("El ZIP no contiene manifest_ashr_oof.json")
            _integrate_from_dir(candidates[0].parent, args.paper_dir)
    elif args.results_dir is not None:
        _integrate_from_dir(args.results_dir, args.paper_dir)

    print("Integración completada con verificación de manifiesto.")


if __name__ == "__main__":
    main()
