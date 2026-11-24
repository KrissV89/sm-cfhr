"""Generate clean vector conceptual/mathematical figures for the SM-CFHR paper.

Graphical standard:
- Vector PDF with embedded TrueType fonts (pdf.fonttype = 42, 0 Type 3 fonts).
- Sans-serif typography at 12-13 pt reference scale.
- All labels, units, titles, annotations, and legends in English.
- Grayscale-distinguishable line styles, markers, and hatches.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 12.0,
    "axes.titlesize": 13.0,
    "axes.labelsize": 12.0,
    "xtick.labelsize": 11.0,
    "ytick.labelsize": 11.0,
    "legend.fontsize": 10.5,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT_DIR = Path(__file__).resolve().parents[1] / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _box(ax, x, y, w, h, text, fc="#F5F7FA", ec="#2B4C7E", lw=1.3, fs=11.0, bold=False, ls="-"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight=weight, color="#1A202C")


def _arrow(ax, x1, y1, x2, y2, color="#2D3748", lw=1.4, style="-|>"):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13, color=color, linewidth=lw)
    ax.add_patch(arr)


def make_fig_arquitectura() -> None:
    fig, ax = plt.subplots(figsize=(13.0, 5.6))
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    _box(ax, 0.25, 3.25, 2.75, 1.6,
         "1. Physical inputs at origin t\n$y_{i,s}$ (kWh/day), $C_{i,t}$ (MW)\n$E^{\\max}_{i,t} = 24000\\,C_{i,t}$",
         fc="#EDF2F7", ec="#2B4C7E", fs=11.0)

    _box(ax, 3.45, 3.25, 2.95, 1.6,
         "2. Dimensionless CF space\nCalendar (4) + $\\mathrm{cf}_{30}$\n$+\\,\\gamma_h\\Delta_{7,30},\\,\\gamma_h\\Delta_{1,7}$\n+ NWP (5) + Anomalies (5)",
         fc="#EBF8FF", ec="#2B6CB0", fs=11.0)

    _box(ax, 6.85, 3.25, 2.85, 1.6,
         "3. Fold preprocessing\nMedian imputation $\\mu_{\\mathrm{train}}$\n$Z = (X - \\bar{X}) / \\tilde{s}$\n$\\tilde{y} = (y_{\\mathrm{cf}} - m_y) / \\sigma_{\\mathrm{MAD}}$",
         fc="#E6FFFA", ec="#2C7A7B", fs=11.0)

    _box(ax, 10.15, 3.25, 2.6, 1.6,
         "4. Convex optimization\nL-BFGS-B ($O(np)$/eval)\nHuber $\\rho_\\delta$ + diag. $L_2$\n$w_j \\geq 0\\;(j \\in \\mathcal{M}_+)$",
         fc="#FFF5F5", ec="#9B2C2C", fs=11.0)

    _box(ax, 1.5, 0.55, 4.7, 1.65,
         "5. Physical reconstruction and clipping\n$\\widehat{\\mathrm{cf}}_q = \\mathrm{clip}(m_y + \\sigma_{\\mathrm{MAD}}(\\hat{b} + \\mathbf{z}_q^\\top \\hat{\\mathbf{w}}),\\, 0,\\, 1)$\n$\\hat{y}_q = \\widehat{\\mathrm{cf}}_q \\cdot 24000\\,C_{i,t} \\cdot H_{\\mathrm{dir}}$ (kWh)",
         fc="#FEFCBF", ec="#975A16", fs=11.0)

    _box(ax, 6.9, 0.55, 4.8, 1.65,
         "6. Evaluation configurations\nFixed ($B=0$): $\\delta=1.35,\\,\\alpha_{\\mathrm{lin}}=0.01,\\,\\alpha_{\\mathrm{meteo}}=0.10$\nHPO ($B=25$): Optuna TPE in nested causal CV\nModes: $D+1$, Direct ($H_{\\mathrm{dir}}=H$), Recursive",
         fc="#F0FFF4", ec="#276749", fs=11.0)

    _arrow(ax, 3.0, 4.05, 3.45, 4.05)
    _arrow(ax, 6.4, 4.05, 6.85, 4.05)
    _arrow(ax, 9.7, 4.05, 10.15, 4.05)
    _arrow(ax, 11.45, 3.25, 11.45, 2.6)
    _arrow(ax, 11.45, 2.6, 3.85, 2.6)
    _arrow(ax, 3.85, 2.6, 3.85, 2.2)
    _arrow(ax, 6.2, 1.37, 6.9, 1.37)

    ax.set_title("Mathematical Architecture and Estimation Pipeline of SM-CFHR", pad=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_sm_cfhr_arquitectura.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig_huber_estructura() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.8))

    r = np.linspace(-3.5, 3.5, 500)
    delta = 1.35
    huber = np.where(np.abs(r) <= delta, 0.5 * r**2, delta * (np.abs(r) - 0.5 * delta))
    quad = 0.5 * r**2
    psi = np.where(np.abs(r) <= delta, r, delta * np.sign(r))

    ax1.plot(r, quad, "--", color="#4A5568", lw=2.0, label=r"Quadratic $\frac{1}{2}r^2$")
    ax1.plot(r, huber, "-", color="#1A365D", lw=2.4, label=r"Huber $\rho_\delta(r)$ ($\delta=1.35$)")
    ax1.plot(r, psi, "-.", color="#9B2C2C", lw=2.0, label=r"Derivative $\psi_\delta(r)$")
    ax1.axvline(-delta, color="#718096", ls=":", lw=1.2)
    ax1.axvline(delta, color="#718096", ls=":", lw=1.2)
    ax1.text(delta + 0.1, -1.45, r"$+\delta$", fontsize=11.5, color="#2D3748")
    ax1.text(-delta - 0.52, -1.45, r"$-\delta$", fontsize=11.5, color="#2D3748")
    ax1.set_ylim(-1.8, 5.4)
    ax1.set_xlabel(r"Standardized residual $r$")
    ax1.set_ylabel("Loss / Gradient value")
    ax1.set_title("(a) Huber loss and bounded derivative")
    ax1.legend(frameon=True, loc="upper center")
    ax1.grid(alpha=0.25)

    w_cf = np.linspace(-0.4, 1.2, 200)
    w_rad = np.linspace(-0.4, 1.2, 200)
    W_CF, W_RAD = np.meshgrid(w_cf, w_rad)
    Z_obj = 0.5 * (0.01 * (W_CF - 0.65)**2 + 0.10 * (W_RAD + 0.12)**2) + 0.002 * (W_CF - 0.65) * (W_RAD + 0.12)

    ax2.axvspan(0, 1.2, ymin=0.27, ymax=1.0, color="#EBF8FF", alpha=0.7,
                label=r"Feasible set $\mathcal{C}$ ($w_{\mathrm{cf30}}\geq 0,\, w_{\mathrm{rad}}\geq 0$)")
    cs = ax2.contour(W_CF, W_RAD, Z_obj, levels=9, colors="#2C5282", linewidths=1.1, linestyles="-")
    ax2.clabel(cs, inline=True, fontsize=9.5, fmt="%.3f")
    ax2.axhline(0, color="#9B2C2C", lw=1.8, ls="-")
    ax2.axvline(0, color="#9B2C2C", lw=1.8, ls="-")
    ax2.plot(0.65, -0.12, "o", color="#4A5568", ms=8, label="Unconstrained optimum")
    ax2.plot(0.65, 0.0, "s", color="#9B2C2C", ms=8, label="Constrained KKT optimum")
    ax2.set_xlim(-0.35, 1.15)
    ax2.set_ylim(-0.35, 0.95)
    ax2.set_xlabel(r"$w_{\mathrm{cf30}}$ ($\alpha_{\mathrm{lin}} = 0.01$)")
    ax2.set_ylabel(r"$w_{\mathrm{rad}}$ ($\alpha_{\mathrm{meteo}} = 0.10$)")
    ax2.set_title(r"(b) Anisotropic $L_2$ contours and active bound")
    ax2.legend(frameon=True, loc="upper right", fontsize=9.8)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_huber_estructura.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig_validacion_anidada() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 5.2))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    ax.text(0.3, 4.55, "Outer cross-validation: causal temporal split + leave-one-group-out in ST (G = 7 spatial zones)",
            fontsize=12.0, fontweight="bold", color="#1A202C")

    _box(ax, 0.3, 3.4, 5.2, 0.85, "Outer training set $\\mathcal{D}_{\\mathrm{train}}$\n$t \\leq t_{\\mathrm{origin}}$ (and $g(j) \\neq g(i)$ in ST)", fc="#EBF8FF", ec="#2B6CB0", fs=11.0)
    _box(ax, 5.7, 3.4, 1.7, 0.85, "Causal embargo\n(30 d / $t_{\\mathrm{end}} \\leq t_0$)", fc="#FEFCBF", ec="#B7791F", fs=10.5, ls="--")
    _box(ax, 7.6, 3.4, 4.5, 0.85, "Outer OOF evaluation (unseen by HPO)\nD+1 blocks | Contiguous origins (7-28 d)", fc="#F0FFF4", ec="#276749", fs=10.8)

    ax.text(0.3, 2.65, "Inner nested validation (SM-CFHR-HPO only: 25 TPE trials inside $\\mathcal{D}_{\\mathrm{train}}$):",
            fontsize=11.5, fontweight="bold", color="#2D3748")

    for k, (y0, lbl) in enumerate([(1.75, "Inner fold 1"), (0.98, "Inner fold 2"), (0.2, "Inner fold 3")]):
        w_tr = 3.0 + 0.65 * (2 - k)
        _box(ax, 0.3, y0, w_tr, 0.58, f"{lbl}: Inner train (g != g_k in ST)", fc="#EDF2F7", ec="#4A5568", fs=10.0)
        _box(ax, 0.4 + w_tr, y0, 1.1, 0.58, "Gap >= 30d", fc="#FEFCBF", ec="#B7791F", fs=9.5, ls="--")
        _box(ax, 1.6 + w_tr, y0, 1.9, 0.58, "Inner val (exact)", fc="#FFF5F5", ec="#9B2C2C", fs=10.0)

    _box(ax, 8.0, 0.3, 4.1, 1.95,
         "Search budgets and fairness\nFixed: B = 0 (no inner CV)\nHPO: B = 25 (Optuna TPE)\nReferences: 4-config grid\nImputation & MAD refit per fold",
         fc="#FAF5FF", ec="#553C9A", fs=10.5)

    _arrow(ax, 7.2, 2.04, 8.0, 1.28)
    ax.set_title("Nested Causal Cross-Validation Protocol (Zero Temporal or Spatial Leakage)", pad=8, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_validacion_anidada.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig_despliegue_multihorizonte() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 5.2))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    _box(ax, 0.25, 3.45, 11.9, 1.25,
         "A. Direct Cumulative Scheme ($H \\in \\{7, 14, 21, 28\\}$ days, without meteorology)\n"
         "Trained on complete windows ($t_{\\mathrm{end}} = t_{\\mathrm{origin}} + H \\leq t_0$); scaled target $Y_{i,t}^{(H)} / (24000\\,H\\,C_{i,t}) \\in [0,1]$.\n"
         "Single-step forecast ($H_{\\mathrm{dir}}=H$) $\\longrightarrow \\hat{Y}_{i,t}^{(H)}$ in total kWh without feedback.",
         fc="#EBF8FF", ec="#2B6CB0", fs=11.0)

    _box(ax, 0.25, 0.35, 11.9, 2.7, "", fc="#F7FAFC", ec="#276749", lw=1.4)
    ax.text(6.2, 2.75, "B. Causal Recursive Scheme ($H \\in \\{7, 14, 21, 28\\}$ days, with or without GEFSv12)",
            ha="center", va="center", fontsize=11.8, fontweight="bold", color="#22543D")

    _box(ax, 0.5, 0.75, 2.55, 1.65,
         "State at origin $t_0$\nFixed $C_{i,t_0}$\n30d buffer $y_s$ ($s \\leq t_0$)\n30d NWP mean ($\\leq t_0$)",
         fc="#EDF2F7", ec="#4A5568", fs=10.2)

    _box(ax, 3.45, 0.75, 2.85, 1.65,
         "Step $h \\in \\{1,\\dots,H\\}$\nDamping $\\gamma_h = 0.85^{h-1}$\nfor $\\Delta_{7,30},\\,\\Delta_{1,7}$\n+ GEFSv12($t_0+h$)",
         fc="#E6FFFA", ec="#2C7A7B", fs=10.2)

    _box(ax, 6.7, 0.75, 2.45, 1.65,
         "SM-CFHR Predictor\n$\\widehat{\\mathrm{cf}}_{i,t_0+h\\mid t_0} \\in [0,1]$\n$\\hat{y}_{i,t_0+h\\mid t_0}$ (kWh/d)",
         fc="#FFF5F5", ec="#9B2C2C", fs=10.2)

    _box(ax, 9.55, 0.75, 2.35, 1.65,
         "Causal Buffer Update\nAppend $\\hat{y}_{i,t_0+h\\mid t_0}$\n(never future $y$)\nfor step $h+1$",
         fc="#FEFCBF", ec="#975A16", fs=10.2)

    _arrow(ax, 3.05, 1.57, 3.45, 1.57)
    _arrow(ax, 6.3, 1.57, 6.7, 1.57)
    _arrow(ax, 9.15, 1.57, 9.55, 1.57)
    _arrow(ax, 10.72, 0.75, 10.72, 0.48)
    _arrow(ax, 10.72, 0.48, 4.87, 0.48)
    _arrow(ax, 4.87, 0.48, 4.87, 0.75)

    ax.set_title("Multi-Horizon Deployment: Direct Cumulative vs. Causal Recursive Feedback", pad=8, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_despliegue_multihorizonte.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig_pipeline_reproduccion() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 4.2))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    _box(ax, 0.3, 2.05, 2.55, 1.45,
         "1. Data & Integrity\nSHA-256 verification\nPanel & 28 scenarios\n22 reference baselines", fc="#EDF2F7", ec="#2B4C7E", fs=10.5)
    _box(ax, 3.3, 2.05, 2.65, 1.45,
         "2. Causal OOF Engine\nFixed (B = 0) & HPO (B = 25)\nExact recursive rollout\nCheckpointed execution", fc="#EBF8FF", ec="#2B6CB0", fs=10.5)
    _box(ax, 6.4, 2.05, 2.65, 1.45,
         "3. Statistical Inference\nDM-HLN-HAC, MCS, SPA\nHolm / BH multiplicity\nAblations & Conformal", fc="#F0FFF4", ec="#276749", fs=10.5)
    _box(ax, 9.5, 2.05, 2.4, 1.45,
         "4. Reproducibility\nIndependent machine\ncross-check (Fixed/HPO)\nPaper artifact sync", fc="#FAF5FF", ec="#553C9A", fs=10.5)

    _box(ax, 1.2, 0.35, 9.8, 1.1,
         "Traceability invariants: Frozen core math (SHA-256 verified) | Formal Lean 4 proofs (v4.36.0)\n"
         "Strict key alignment without silent row drops | Zero synthetic fallbacks in confirmatory mode",
         fc="#FEFCBF", ec="#975A16", fs=10.8)

    _arrow(ax, 2.85, 2.77, 3.3, 2.77)
    _arrow(ax, 5.95, 2.77, 6.4, 2.77)
    _arrow(ax, 9.05, 2.77, 9.5, 2.77)

    ax.set_title("Reproducible Evaluation Pipeline and Integrity Verification Architecture", pad=8, fontweight="bold")
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
