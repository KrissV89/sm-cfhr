"""Generate clean vector conceptual/mathematical figures for the SM-CFHR paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
})

OUT_DIR = Path(__file__).resolve().parents[1] / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _box(ax, x, y, w, h, text, fc="#F5F7FA", ec="#2B4C7E", lw=1.2, fs=8.5, bold=False):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=fc, edgecolor=ec, linewidth=lw,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight=weight, color="#1A202C")


def _arrow(ax, x1, y1, x2, y2, color="#2D3748", lw=1.2, style="-|>"):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11, color=color, linewidth=lw)
    ax.add_patch(arr)


def make_fig_arquitectura() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    _box(ax, 0.2, 2.7, 2.1, 1.2,
         "1. Magnitudes físicas\ny origen $t$\n$y_{i,s}$ (kWh/día), $C_{i,t}$ (MW)\n$E^{\\max}_{i,t} = 24000\\,C_{i,t}$",
         fc="#EDF2F7", ec="#2B4C7E", bold=False)

    _box(ax, 2.7, 2.7, 2.3, 1.2,
         "2. Espacio adimensional\n$\\mathrm{cf}_{i,s} \\in [0,1]$\nCalendario (4) + $\\mathrm{cf}_{30}$\n+ $\\Delta_{7,30},\\Delta_{1,7}$ (amort.)\n+ NWP (5) + Anom. (5)",
         fc="#EBF8FF", ec="#2B6CB0")

    _box(ax, 5.4, 2.7, 2.3, 1.2,
         "3. Preprocesamiento\n(solo entrenamiento)\nMediana $\\mu_{\\mathrm{train}}$\n$Z = (X - \\bar{X}) / s_X$\n$\\tilde{y} = (y_{\\mathrm{cf}} - m_y) / \\sigma_{\\mathrm{MAD}}$",
         fc="#E6FFFA", ec="#2C7A7B")

    _box(ax, 8.1, 2.7, 2.2, 1.2,
         "4. Optimización convexa\nL-BFGS-B ($O(np)$)\nPérdida Huber $\\rho_\\delta$\n$L_2$ diag. $(\\alpha_{\\mathrm{lin}}, \\alpha_{\\mathrm{meteo}})$\nCotas $w_j \\geq 0\\;(j\\in\\mathcal{M}_+)$",
         fc="#FFF5F5", ec="#9B2C2C")

    _box(ax, 1.6, 0.5, 3.3, 1.25,
         "5. Reconstrucción y clipping físico\n$\\hat{y}_{\\mathrm{cf}} = \\mathrm{clip}(m_y + \\sigma_{\\mathrm{MAD}}(\\hat{b} + \\mathbf{z}^\\top \\hat{\\mathbf{w}}),\\, 0,\\, 1)$\n$\\hat{Y} = \\hat{y}_{\\mathrm{cf}} \\cdot 24000\\,C_{i,t} \\cdot H_{\\mathrm{dir}}$ (kWh)",
         fc="#FEFCBF", ec="#975A16")

    _box(ax, 5.5, 0.5, 3.6, 1.25,
         "6. Modos de evaluación congelados\n• SM-CFHR-Fixed ($B=0$): $\\delta=1.35, \\alpha_{\\mathrm{lin}}=0.01, \\alpha_{\\mathrm{meteo}}=0.10$\n• SM-CFHR-HPO ($B=25$): TPE en CV causal anidada\n• Despliegue $D+1$, Directo ($H_{\\mathrm{dir}}=H$) y Recursivo",
         fc="#F0FFF4", ec="#276749")

    _arrow(ax, 2.3, 3.3, 2.7, 3.3)
    _arrow(ax, 5.0, 3.3, 5.4, 3.3)
    _arrow(ax, 7.7, 3.3, 8.1, 3.3)
    _arrow(ax, 9.2, 2.7, 9.2, 2.15)
    _arrow(ax, 9.2, 2.15, 3.25, 2.15)
    _arrow(ax, 3.25, 2.15, 3.25, 1.75)
    _arrow(ax, 4.9, 1.12, 5.5, 1.12)

    ax.set_title("Arquitectura matemática y flujo de estimación de SM-CFHR", pad=8, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_sm_cfhr_arquitectura.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig_huber_estructura() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0))

    r = np.linspace(-3.5, 3.5, 500)
    delta = 1.35
    huber = np.where(np.abs(r) <= delta, 0.5 * r**2, delta * (np.abs(r) - 0.5 * delta))
    quad = 0.5 * r**2
    psi = np.where(np.abs(r) <= delta, r, delta * np.sign(r))

    ax1.plot(r, quad, "--", color="#718096", lw=1.4, label=r"Cuadrática $\frac{1}{2}r^2$")
    ax1.plot(r, huber, "-", color="#2B6CB0", lw=2.0, label=r"Huber $\rho_\delta(r)$ ($\delta=1.35$)")
    ax1.plot(r, psi, "-.", color="#C53030", lw=1.7, label=r"Influencia $\psi_\delta(r)$")
    ax1.axvline(-delta, color="#A0AEC0", ls=":", lw=1.0)
    ax1.axvline(delta, color="#A0AEC0", ls=":", lw=1.0)
    ax1.text(delta + 0.08, 4.6, r"$+\delta$", fontsize=8.5, color="#4A5568")
    ax1.text(-delta - 0.35, 4.6, r"$-\delta$", fontsize=8.5, color="#4A5568")
    ax1.set_ylim(-1.8, 5.2)
    ax1.set_xlabel(r"Residuo estandarizado $r$")
    ax1.set_ylabel("Valor / Gradiente")
    ax1.set_title("(a) Pérdida de Huber y función de influencia")
    ax1.legend(frameon=True, loc="upper center")
    ax1.grid(alpha=0.25)

    w_cf = np.linspace(-0.4, 1.2, 200)
    w_rad = np.linspace(-0.4, 1.2, 200)
    W_CF, W_RAD = np.meshgrid(w_cf, w_rad)
    # Anisotropic L2 contours around an unconstrained minimum
    Z_obj = 0.5 * (0.01 * (W_CF - 0.65)**2 + 0.10 * (W_RAD + 0.12)**2) + 0.002 * (W_CF - 0.65) * (W_RAD + 0.12)

    ax2.axvspan(0, 1.2, ymin=0.25, ymax=1.0, color="#EBF8FF", alpha=0.65, label=r"Región factible $\mathcal{C}$ ($w_{\mathrm{cf30}}\geq 0,\, w_{\mathrm{rad}}\geq 0$)")
    cs = ax2.contour(W_CF, W_RAD, Z_obj, levels=10, colors="#2C5282", linewidths=1.0)
    ax2.clabel(cs, inline=True, fontsize=7, fmt="%.3f")
    ax2.axhline(0, color="#9B2C2C", lw=1.5)
    ax2.axvline(0, color="#9B2C2C", lw=1.5)
    ax2.plot(0.65, -0.12, "o", color="#718096", ms=6, label="Óptimo sin restricción")
    ax2.plot(0.65, 0.0, "s", color="#C53030", ms=7, label="Óptimo KKT factible (cota activa)")
    ax2.set_xlim(-0.35, 1.15)
    ax2.set_ylim(-0.35, 0.95)
    ax2.set_xlabel(r"$w_{\mathrm{cf30}}$ ($\alpha_{\mathrm{lin}} = 0.01$)")
    ax2.set_ylabel(r"$w_{\mathrm{rad\_pron}}$ ($\alpha_{\mathrm{meteo}} = 0.10$)")
    ax2.set_title(r"(b) Penalización $L_2$ anisotrópica y cota activa")
    ax2.legend(frameon=True, loc="upper right", fontsize=7.5)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_huber_estructura.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig_validacion_anidada() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 4.2))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    # Timeline bar
    ax.text(0.3, 3.65, "Esquema externo (Outer CV): separación temporal causal + Leave-One-Group-Out en ET ($G=7$ zonas)",
            fontsize=9.5, fontweight="bold", color="#1A202C")

    _box(ax, 0.3, 2.75, 4.4, 0.65, "Entrenamiento externo $\\mathcal{D}_{\\mathrm{train}}$\n$t \\leq t_{\\mathrm{origen}}$ (y $g(j) \\neq g(i)$ en ET)", fc="#EBF8FF", ec="#2B6CB0")
    _box(ax, 4.85, 2.75, 1.3, 0.65, "Embargo causal\n(30 días / $t_{\\mathrm{end}} \\leq t_0$)", fc="#FEFCBF", ec="#B7791F")
    _box(ax, 6.3, 2.75, 3.6, 0.65, "Evaluación externa OOF (intocable por HPO)\n• $D+1$: bloques $D_1, D_2, D_3$\n• Multihorizonte: orígenes contiguos ($H\\in\\{7,14,21,28\\}$)", fc="#F0FFF4", ec="#276749")

    # Inner CV block
    ax.text(0.3, 2.05, "Validación interna anidada (solo para SM-CFHR-HPO, 25 trials TPE dentro de $\\mathcal{D}_{\\mathrm{train}}$):",
            fontsize=9.0, fontweight="bold", color="#2D3748")

    for k, (y0, lbl) in enumerate([(1.35, "Pliegue interno 1"), (0.75, "Pliegue interno 2"), (0.15, "Pliegue interno 3")]):
        w_tr = 2.6 + 0.55 * (2 - k)
        _box(ax, 0.3, y0, w_tr, 0.45, f"{lbl}: Train interno (y $g \\neq g_k$ en ET)", fc="#EDF2F7", ec="#4A5568", fs=8.0)
        _box(ax, 0.35 + w_tr, y0, 0.85, 0.45, "Gap $\\geq 30$d", fc="#FEFCBF", ec="#B7791F", fs=7.5)
        _box(ax, 1.25 + w_tr, y0, 1.5, 0.45, "Val. interna (rollout)", fc="#FFF5F5", ec="#9B2C2C", fs=8.0)

    _box(ax, 6.5, 0.35, 3.4, 1.35,
         "Presupuesto y equidad\n• SM-CFHR-Fixed: $B = 0$ (sin CV interna)\n• SM-CFHR-HPO: $B = 25$ (Optuna TPE)\n• Benchmarks: rejilla de $4$ configs.\n• Imputación y MAD reajustados por pliegue",
         fc="#FAF5FF", ec="#553C9A", fs=8.0)

    _arrow(ax, 5.9, 1.55, 6.5, 1.02)
    ax.set_title("Protocolo de validación cruzada causal anidada (sin fuga temporal ni espacial)", pad=6, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_validacion_anidada.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig_despliegue_multihorizonte() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 4.3))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 4.3)
    ax.axis("off")

    _box(ax, 0.25, 2.8, 9.7, 1.05,
         "A. Esquema Directo ($H \\in \\{7, 14, 21, 28\\}$ días, sin meteorología)\n"
         "Entrenamiento con $t_{\\mathrm{end}} = t_{\\mathrm{origen}} + H \\leq t_0$; objetivo escalado $Y_{i,t}^{(H)} / (H \\cdot 24000\\,C_{i,t}) \\in [0,1]$.\n"
         "Predicción en un solo paso ($h=1$, $H_{\\mathrm{dir}}=H$) $\\longrightarrow \\hat{Y}_{i,t}^{(H)}$ en kWh totales (o MWh) sin realimentación.",
         fc="#EBF8FF", ec="#2B6CB0", fs=8.5)

    _box(ax, 0.25, 0.35, 9.7, 2.1, "", fc="#F7FAFC", ec="#276749", lw=1.3)
    ax.text(5.1, 2.22, "B. Esquema Recursivo ($H \\in \\{7, 14, 21, 28\\}$ días, sin clima o con GEFSv12)",
            ha="center", va="center", fontsize=9.2, fontweight="bold", color="#22543D")

    _box(ax, 0.45, 0.65, 2.1, 1.25,
         "Estado al origen $t_0$\n$C_{i,t_0}$ fija\nVentanas $y_s$ ($s \\leq t_0$)\nBase NWP 30d ($\\leq t_0$)",
         fc="#EDF2F7", ec="#4A5568", fs=8.0)

    _box(ax, 2.9, 0.65, 2.3, 1.25,
         "Paso $h \\in \\{1,\\dots,H\\}$\nAmortiguamiento $0.85^{h-1}$\nsobre $\\Delta_{7,30}, \\Delta_{1,7}$\n+ Pronóstico GEFSv12($t_0+h$)",
         fc="#E6FFFA", ec="#2C7A7B", fs=8.0)

    _box(ax, 5.55, 0.65, 2.0, 1.25,
         "Estimador SM-CFHR\n$\\hat{\\mathrm{cf}}_{i,t_0+h\\mid t_0} \\in [0,1]$\n$\\hat{y}_{i,t_0+h\\mid t_0}$ (kWh/día)",
         fc="#FFF5F5", ec="#9B2C2C", fs=8.0)

    _box(ax, 7.9, 0.65, 1.85, 1.25,
         "Actualización causal\nSolo usa $\\hat{y}_{i,t_0+h\\mid t_0}$\n(nunca $y$ real futuro)\npara paso $h+1$",
         fc="#FEFCBF", ec="#975A16", fs=8.0)

    _arrow(ax, 2.55, 1.27, 2.9, 1.27)
    _arrow(ax, 5.2, 1.27, 5.55, 1.27)
    _arrow(ax, 7.55, 1.27, 7.9, 1.27)
    _arrow(ax, 8.82, 0.65, 8.82, 0.45)
    _arrow(ax, 8.82, 0.45, 4.05, 0.45)
    _arrow(ax, 4.05, 0.45, 4.05, 0.65)

    ax.set_title("Despliegue multihorizonte: formulación directa acumulada frente a realimentación recursiva causal", pad=6, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_despliegue_multihorizonte.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig_pipeline_reproduccion() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 3.6))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    _box(ax, 0.25, 1.85, 2.1, 1.15,
         "1. Datos y manifiesto\n`import_local_data.py`\nVerificación SHA-256\n(`doctor`)", fc="#EDF2F7", ec="#2B4C7E", fs=8.0)
    _box(ax, 2.75, 1.85, 2.2, 1.15,
         "2. OOF resumible\nFragmentos Parquet\nOptuna SQLite (`RDB`)\nGuardia de memoria RSS", fc="#EBF8FF", ec="#2B6CB0", fs=8.0)
    _box(ax, 5.35, 1.85, 2.2, 1.15,
         "3. Post-OOF (8 etapas)\nInferencia (DM, MCS, SPA)\nAblaciones y Conformal\nVeredicto R3 estricto", fc="#F0FFF4", ec="#276749", fs=8.0)
    _box(ax, 7.95, 1.85, 2.0, 1.15,
         "4. Reproducibilidad\n`repro-bundle`\n`compare-runs`\nIntegración al paper", fc="#FAF5FF", ec="#553C9A", fs=8.0)

    _box(ax, 1.4, 0.3, 7.4, 0.9,
         "Invariantes de trazabilidad: tag congelado `v1.0.2` (commit `615a5af`)  |  Escritura atómica (`fsync` + `os.replace`)\n"
         "Prohibición de datos sintéticos en modo confirmatorio  |  Alineación estricta sin descartes silenciosos",
         fc="#FEFCBF", ec="#975A16", fs=8.2)

    _arrow(ax, 2.35, 2.42, 2.75, 2.42)
    _arrow(ax, 4.95, 2.42, 5.35, 2.42)
    _arrow(ax, 7.55, 2.42, 7.95, 2.42)

    ax.set_title("Arquitectura de ejecución resumible, verificación de integridad y reproducción independiente", pad=6, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_pipeline_reproduccion.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_fig_arquitectura()
    make_fig_huber_estructura()
    make_fig_validacion_anidada()
    make_fig_despliegue_multihorizonte()
    make_fig_pipeline_reproduccion()
    print("Generated 5 conceptual PDFs in", OUT_DIR)
