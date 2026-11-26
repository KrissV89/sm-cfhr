"""Centralized atomic persistence and provenance export for scientific OOF artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

ARTIFACT_SUBDIRS = (
    "configuraciones",
    "particiones",
    "optimizacion",
    "hpo_trials",
    "predicciones",
    "metricas",
    "inferencia",
    "multiplicidad",
    "robustez",
    "sensibilidad",
    "ablaciones",
    "incertidumbre",
    "interpretabilidad",
    "coste",
    "tablas_paper",
    "figuras",
    "conclusiones",
    "reproducibilidad",
)


def atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically using a temporary file and os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_df(df: pd.DataFrame, path_stem: Path, write_tex: bool = False) -> dict[str, Path]:
    """Persist DataFrame atomically to CSV (and optionally LaTeX / Parquet if pyarrow exists)."""
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    csv_p = path_stem.with_suffix(".csv")
    csv_tmp = csv_p.with_suffix(".csv.tmp")
    df.to_csv(csv_tmp, index=False)
    os.replace(csv_tmp, csv_p)
    out["csv"] = csv_p

    if write_tex:
        tex_p = path_stem.with_suffix(".tex")
        atomic_write_text(tex_p, df.to_latex(index=False, float_format="%.3f"))
        out["tex"] = tex_p

    try:
        import pyarrow  # noqa: F401

        pq_p = path_stem.with_suffix(".parquet")
        pq_tmp = pq_p.with_suffix(".parquet.tmp")
        df.to_parquet(pq_tmp, index=False)
        os.replace(pq_tmp, pq_p)
        out["parquet"] = pq_p
    except ImportError:
        pass
    return out


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_artifact_tree(out_dir: Path) -> dict[str, Path]:
    """Create the standardized directory tree inside `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs: dict[str, Path] = {}
    for sub in ARTIFACT_SUBDIRS:
        p = out_dir / sub
        p.mkdir(parents=True, exist_ok=True)
        dirs[sub] = p
    return dirs


def export_full_scientific_artifacts(
    bundle: dict[str, Any],
    out_dir: Path,
    extra_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist all OOF metrics, inference, R3 verdict, robustness, and reproducibility metadata."""
    dirs = ensure_artifact_tree(out_dir)
    exported: dict[str, str] = {}

    # Core bundle tables
    table_map = {
        "metricas/tabla_28_escenarios": bundle["tabla_28_escenarios"],
        "metricas/tabla_rankings_regimen": bundle["tabla_rankings_regimen"],
        "metricas/tabla_por_planta": bundle["tabla_por_planta"],
        "inferencia/tabla_inferencia": bundle["tabla_inferencia"],
        "inferencia/tabla_mcs": bundle.get("tabla_mcs", pd.DataFrame()),
        "coste/tabla_coste": bundle["tabla_coste"],
    }
    for rel, df in table_map.items():
        if df is not None and not df.empty:
            res = atomic_write_df(df, out_dir / rel, write_tex=True)
            for fmt, p in res.items():
                exported[f"{rel}.{fmt}"] = sha256_path(p)

    # Extra modular artifacts (R3 verdict, design, HPO, nested CV, ablation, sensitivity, etc.)
    if extra_artifacts:
        for rel, obj in extra_artifacts.items():
            target = out_dir / rel
            if isinstance(obj, pd.DataFrame):
                if not obj.empty:
                    res = atomic_write_df(obj, target, write_tex=True)
                    for fmt, p in res.items():
                        exported[f"{rel}.{fmt}"] = sha256_path(p)
            elif isinstance(obj, dict):
                json_p = target.with_suffix(".json")
                atomic_write_text(json_p, json.dumps(obj, indent=2, ensure_ascii=False, default=str))
                exported[f"{rel}.json"] = sha256_path(json_p)
            elif isinstance(obj, str):
                md_p = target.with_suffix(".md")
                atomic_write_text(md_p, obj)
                exported[f"{rel}.md"] = sha256_path(md_p)

    rep_manifest = {
        "procedencia": bundle.get("manifest", {}).get("procedencia", "DESCONOCIDA"),
        "n_archivos_exportados": len(exported),
        "hashes_sha256": exported,
    }
    rep_path = dirs["reproducibilidad"] / "manifiesto_exportaciones.json"
    atomic_write_text(rep_path, json.dumps(rep_manifest, indent=2, ensure_ascii=False))
    return rep_manifest


__all__ = [
    "ARTIFACT_SUBDIRS",
    "atomic_write_df",
    "atomic_write_text",
    "ensure_artifact_tree",
    "export_full_scientific_artifacts",
    "sha256_path",
]
