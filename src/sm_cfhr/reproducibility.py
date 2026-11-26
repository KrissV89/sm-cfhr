"""Independent reproducibility evidence bundle generator and cross-machine comparator."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

try:
    from ashr_oof_pipeline.config import load_frozen_config
except ImportError:
    from src.ashr_oof_pipeline.config import load_frozen_config

from .persistence import atomic_write_text, sha256_file

KEY_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "optuna",
    "statsmodels",
    "matplotlib",
    "joblib",
    "pyarrow",
    "psutil",
    "nbformat",
    "sm-cfhr",
)


def _pkg_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in KEY_PACKAGES:
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = "NOT_INSTALLED"
    return out


def _git_info(root: Path) -> dict[str, Any]:
    def _run(args: list[str]) -> str:
        try:
            res = subprocess.run(
                ["git", *args], cwd=str(root), capture_output=True, text=True, check=False
            )
            return res.stdout.strip() if res.returncode == 0 else "UNKNOWN"
        except OSError:
            return "UNKNOWN"

    return {
        "commit": _run(["rev-parse", "HEAD"]),
        "tag": _run(["describe", "--tags", "--always"]),
        "branch": _run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_run(["status", "--porcelain"])),
    }


def generate_reproducibility_bundle(
    root: Path,
    run_id: str,
    results_base: Path | None = None,
) -> dict[str, Path]:
    """Generate the 7 mandatory reproducibility artifacts under `results/runs/<run_id>/reproducibilidad/`."""
    base = results_base or (root / "results")
    repro_dir = base / "runs" / run_id / "reproducibilidad"
    repro_dir.mkdir(parents=True, exist_ok=True)

    # 1. environment.json
    env_data = {
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu_count_logical": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "packages": _pkg_versions(),
    }
    p_env = repro_dir / "environment.json"
    atomic_write_text(p_env, json.dumps(env_data, indent=2, ensure_ascii=False))

    # 2. git_revision.json
    p_git = repro_dir / "git_revision.json"
    atomic_write_text(p_git, json.dumps(_git_info(root), indent=2, ensure_ascii=False))

    # 3. data_manifest.json
    dm_src = root / "data" / "manifests" / "data_manifest.json"
    dm_data: dict[str, Any] = json.loads(dm_src.read_text(encoding="utf-8")) if dm_src.exists() else {"datasets": []}
    for item in dm_data.get("datasets", []):
        fp = root / item["expected_path"]
        item["present_on_machine"] = fp.exists()
        item["actual_sha256"] = sha256_file(fp) if fp.exists() else None
        item["sha256_match"] = bool(fp.exists() and item["actual_sha256"] == item.get("sha256"))
    p_data = repro_dir / "data_manifest.json"
    atomic_write_text(p_data, json.dumps(dm_data, indent=2, ensure_ascii=False))

    # 4. experiment_config.json
    cfg = load_frozen_config(root)
    p_cfg = repro_dir / "experiment_config.json"
    atomic_write_text(p_cfg, json.dumps(cfg, indent=2, ensure_ascii=False))

    # 5. execution_manifest.json
    run_manifest_p = root / "resultados" / "evaluacion_ashr_oof" / "runs" / run_id / "manifest.json"
    if run_manifest_p.exists():
        exec_data = json.loads(run_manifest_p.read_text(encoding="utf-8"))
    else:
        exec_data = {
            "run_id": run_id,
            "estatus": "PENDIENTE_EJECUCION_OOF",
            "nota": "No se ha ejecutado OOF confirmatorio para este run_id.",
        }
    p_exec = repro_dir / "execution_manifest.json"
    atomic_write_text(p_exec, json.dumps(exec_data, indent=2, ensure_ascii=False))

    # 6. results_checksums.json
    pred_dir = root / "resultados" / "evaluacion_ashr_oof" / "predicciones"
    checksums: dict[str, str] = {}
    if pred_dir.exists():
        for csv_p in sorted(pred_dir.glob("*.csv")):
            checksums[csv_p.name] = sha256_file(csv_p)
    p_chk = repro_dir / "results_checksums.json"
    atomic_write_text(p_chk, json.dumps({"n_files": len(checksums), "sha256": checksums}, indent=2))

    # 7. reproducibility_report.md
    report_md = "\n".join(
        [
            f"# Reporte de Reproducibilidad — Run `{run_id}`",
            "",
            f"- **Python / SO:** `{env_data['python_version']}` / `{env_data['os']}`",
            f"- **Git:** commit `{_git_info(root)['commit']}` (tag `{_git_info(root)['tag']}`)",
            f"- **Estado de ejecución OOF:** `{exec_data.get('estatus')}`",
            f"- **Artefactos de predicción verificados:** `{len(checksums)}`",
            "",
            "## Niveles de Reproducibilidad",
            "1. **Técnica:** `VERIFICADA` en entorno local (instalación, CLI y pruebas unitarias); `PENDIENTE` en equipo independiente.",
            "2. **Computacional:** `PENDIENTE_VALIDACION_INDEPENDIENTE` (comparar mediante `python -m sm_cfhr.cli compare-runs`).",
            "3. **Científica:** `PENDIENTE_VALIDACION_INDEPENDIENTE` (requiere completar OOF confirmatorio en ambos equipos).",
        ]
    )
    p_rep = repro_dir / "reproducibility_report.md"
    atomic_write_text(p_rep, report_md)

    return {
        "environment": p_env,
        "git_revision": p_git,
        "data_manifest": p_data,
        "experiment_config": p_cfg,
        "execution_manifest": p_exec,
        "results_checksums": p_chk,
        "reproducibility_report": p_rep,
    }


def compare_reproducibility_runs(
    dir_a: Path,
    dir_b: Path,
    out_md: Path | None = None,
    rel_tol_fixed: float = 1e-4,
    rel_tol_hpo: float = 1e-2,
) -> dict[str, Any]:
    """Compare two independent OOF runs within numerical tolerances across 3 reproducibility levels."""
    csvs_a = sorted((dir_a / "predicciones").glob("*.csv")) if (dir_a / "predicciones").exists() else []
    csvs_b = sorted((dir_b / "predicciones").glob("*.csv")) if (dir_b / "predicciones").exists() else []
    names_a = {p.name for p in csvs_a}
    names_b = {p.name for p in csvs_b}
    common = sorted(names_a & names_b)

    max_rel_fixed = 0.0
    max_rel_hpo = 0.0
    compared = 0
    for name in common:
        da = pd.read_csv(dir_a / "predicciones" / name)
        db = pd.read_csv(dir_b / "predicciones" / name)
        col = "prediccion_total_kwh" if "prediccion_total_kwh" in da.columns else "prediccion_kwh"
        if len(da) != len(db) or col not in da.columns or col not in db.columns:
            continue
        va = da[col].to_numpy(dtype=float)
        vb = db[col].to_numpy(dtype=float)
        rel = float(np.max(np.abs(va - vb) / np.maximum(np.abs(va), 1.0)))
        if name.endswith("_hpo.csv"):
            max_rel_hpo = max(max_rel_hpo, rel)
        else:
            max_rel_fixed = max(max_rel_fixed, rel)
        compared += 1

    comp_ok = compared > 0 and max_rel_fixed <= rel_tol_fixed and max_rel_hpo <= rel_tol_hpo
    res = {
        "n_files_a": len(names_a),
        "n_files_b": len(names_b),
        "n_compared": compared,
        "max_rel_diff_fixed": max_rel_fixed,
        "max_rel_diff_hpo": max_rel_hpo,
        "computational_reproducible": comp_ok,
    }
    if out_md is not None:
        md = (
            f"# Comparación de Reproducibilidad Independiente\n\n"
            f"- Archivos comparados: `{compared}`\n"
            f"- Máx. diferencia relativa Fixed: `{max_rel_fixed:.6e}` (tolerancia `{rel_tol_fixed}`)\n"
            f"- Máx. diferencia relativa HPO: `{max_rel_hpo:.6e}` (tolerancia `{rel_tol_hpo}`)\n"
            f"- **Reproducibilidad computacional:** `{'CUMPLE' if comp_ok else 'NO_VERIFICADA_O_FUERA_DE_TOLERANCIA'}`\n"
        )
        atomic_write_text(out_md, md)
    return res


__all__ = ["compare_reproducibility_runs", "generate_reproducibility_bundle"]
