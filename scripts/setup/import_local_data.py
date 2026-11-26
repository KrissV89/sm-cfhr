"""Import local authorized solar datasets and official benchmark CSVs with SHA-256 verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import authorized solar datasets and benchmarks from a local folder.")
    parser.add_argument("--source-root", type=Path, required=True, help="Path containing base_generacion_clima/, datos_pronostico*, and resultados/benchmark_*.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    manifest = json.loads((repo_root / "data" / "manifests" / "data_manifest.json").read_text(encoding="utf-8"))

    for item in manifest["datasets"]:
        rel = item["expected_path"]
        src = args.source_root / rel
        dst = repo_root / rel
        if not src.exists():
            print(f"[MISSING] {src}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        actual_sha = sha256_file(dst)
        ok = actual_sha == item["sha256"]
        print(f"[{'OK' if ok else 'HASH_MISMATCH'}] {rel} -> {actual_sha}")

    for bench_dir in ("benchmark_produccion_definitivo", "benchmark_multihorizonte", "benchmark_multihorizonte_directo", "benchmark_multihorizonte_clima"):
        src_b = args.source_root / "resultados" / bench_dir
        dst_b = repo_root / "resultados" / bench_dir
        if src_b.exists():
            shutil.copytree(src_b, dst_b, dirs_exist_ok=True)
            print(f"[OK] Imported {bench_dir}")
        else:
            print(f"[MISSING] {src_b}")


if __name__ == "__main__":
    main()
