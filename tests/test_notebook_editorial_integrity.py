"""Editorial, structural, and scientific integrity tests for SM-CFHR notebooks and generators."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
MAIN_NB = ROOT / "notebooks" / "actuales" / "evaluacion_cientifica_ashr_oof.ipynb"
GEN_SCRIPT = ROOT / "scripts" / "evaluacion" / "generar_notebook_ashr.py"

FORBIDDEN_EDITORIAL_PATTERNS = [
    "guion para la defensa",
    "exposición de 20 minutos",
    "criterio docente",
    "criterio verbal",
    "qué estamos viendo:",
    "qué encontramos:",
    "qué significa:",
    "¡perfecto",
    "vamos a descubrir",
]

EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
TOKEN_RE = re.compile(r"(?:ghp_|gho_|github_pat_|sk-[A-Za-z0-9]{20,})")


def _collect_notebook_text(nb: nbf.NotebookNode) -> str:
    parts: list[str] = []
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            parts.append(cell.source)
        elif cell.cell_type == "code":
            for out in cell.get("outputs", []):
                if "text" in out:
                    t_val = out["text"]
                    parts.append("".join(t_val) if isinstance(t_val, list) else str(t_val))
                data = out.get("data", {})
                for mime in ("text/markdown", "text/plain"):
                    if mime in data:
                        val = data[mime]
                        parts.append("".join(val) if isinstance(val, list) else str(val))
    return "\n".join(parts)


def test_main_notebook_editorial_and_scientific_integrity() -> None:
    nb = nbf.read(MAIN_NB, as_version=4)
    nbf.validate(nb)

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) >= 10
    for cell in code_cells:
        outputs = cell.get("outputs", [])
        assert outputs, f"Executed code cell {cell.get('id')} has no outputs"
        for out in outputs:
            assert out.get("output_type") != "error", f"Error output in cell {cell.get('id')}: {out}"

    full_text = _collect_notebook_text(nb)
    low_text = full_text.lower()

    for pat in FORBIDDEN_EDITORIAL_PATTERNS:
        assert pat not in low_text, f"Forbidden editorial pattern found: {pat!r}"

    assert not EMOJI_RE.search(full_text), "Decorative emojis found in notebook text"
    assert not TOKEN_RE.search(full_text), "Potential credential/token pattern found"

    # Negative results & scientific limitations must remain intact
    for required in ("DirectoSinClima", "NO_CUMPLE", "autorizacion_alcance_solar_docente"):
        assert required in full_text, f"Required scientific limitation/result missing: {required}"


def test_generator_produces_clean_valid_notebook(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("generar_notebook_ashr", GEN_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out_nb = tmp_path / "regenerated.ipynb"
    mod.build_notebook(out_nb)
    nb = nbf.read(out_nb, as_version=4)
    nbf.validate(nb)

    md_text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    low_md = md_text.lower()
    for pat in FORBIDDEN_EDITORIAL_PATTERNS:
        assert pat not in low_md, f"Generator emits forbidden editorial pattern: {pat!r}"
    assert not EMOJI_RE.search(md_text), "Generator emits decorative emojis"
    assert "# Capítulo 8. Síntesis de hallazgos y limitaciones" in md_text
