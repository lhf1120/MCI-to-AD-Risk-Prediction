#!/usr/bin/env python3
"""Create a publication-style interface template figure for the web calculator."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = FIG_DIR / "web_calculator_interface_template.png"
OUT_PDF = FIG_DIR / "web_calculator_interface_template.pdf"


def add_text(ax, x, y, s, size=10.0, weight="normal", color="#0f172a", ha="left", va="center"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va)


def rounded(ax, x, y, w, h, fc="white", ec="#cbd5e1", lw=1.0, r=0.02, alpha=1.0):
    patch = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=patches.BoxStyle("Round", pad=0.008, rounding_size=r),
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        alpha=alpha,
    )
    ax.add_patch(patch)
    return patch


def add_panel_label(ax, label: str, title: str) -> None:
    ax.text(
        0.02,
        0.975,
        f"{label}  {title}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.0,
        fontweight="bold",
        color="#0f172a",
    )


def draw_browser_frame(ax, title: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("white")
    rounded(ax, 0.01, 0.02, 0.98, 0.94, fc="#f8fafc", ec="#cbd5e1", lw=1.0, r=0.02)
    rounded(ax, 0.01, 0.90, 0.98, 0.06, fc="#e2e8f0", ec="#cbd5e1", lw=0.8, r=0.02)
    for x, c in [(0.05, "#ef4444"), (0.08, "#f59e0b"), (0.11, "#22c55e")]:
        ax.add_patch(patches.Circle((x, 0.93), 0.012, facecolor=c, edgecolor="none"))
    add_text(ax, 0.16, 0.93, title, size=9.5, weight="bold", color="#334155")


def draw_input_field(ax, x, y, w, label, value, accent="#0ea5e9"):
    add_text(ax, x, y + 0.030, label, size=7.2, weight="bold", color="#475569")
    rounded(ax, x, y - 0.004, w, 0.028, fc="white", ec="#cbd5e1", lw=0.9, r=0.012)
    add_text(ax, x + 0.010, y + 0.009, value, size=8.0, color="#0f172a")
    ax.add_line(Line2D([x, x + w], [y - 0.004, y - 0.004], lw=1.8, color=accent, alpha=0.14))


def draw_section(ax, x, y, w, h, title, accent, entries):
    rounded(ax, x, y, w, h, fc="#ffffff", ec="#dbe4ee", lw=0.9, r=0.02)
    ax.add_patch(patches.Rectangle((x, y + h - 0.040), w, 0.040, facecolor=accent, edgecolor="none", alpha=0.12))
    add_text(ax, x + 0.02, y + h - 0.020, title, size=8.4, weight="bold", color="#0f172a")

    left = x + 0.02
    field_w = (w - 0.06) / 2
    current_y = y + h - 0.072
    for idx, (label, value) in enumerate(entries):
        field_x = left if idx % 2 == 0 else left + field_w + 0.02
        if idx % 2 == 0 and idx > 0:
            current_y -= 0.074
        draw_input_field(ax, field_x, current_y, field_w, label, value, accent=accent)
    return current_y


def draw_result_badge(ax, x, y, w, h, title, value, fc, ec, title_color="#334155", value_color="#0f172a"):
    rounded(ax, x, y, w, h, fc=fc, ec=ec, lw=1.0, r=0.025)
    add_text(ax, x + 0.02, y + h - 0.020, title, size=7.9, weight="bold", color=title_color, va="top")
    add_text(ax, x + 0.02, y + 0.024, value, size=13.0, weight="bold", color=value_color, va="bottom")


def draw_contrib_bar(ax, x, y, w, label, value, color, max_abs=0.28):
    add_text(ax, x, y + 0.012, label, size=7.5, color="#334155")
    track_x = x + 0.18
    track_w = w - 0.20
    rounded(ax, track_x, y, track_w, 0.022, fc="#edf2f7", ec="#e2e8f0", lw=0.6, r=0.010)
    zero = track_x + track_w * 0.50
    ax.add_line(Line2D([zero, zero], [y, y + 0.022], lw=1.0, color="#94a3b8"))
    bar_w = track_w * min(abs(value) / max_abs, 1.0) * 0.48
    x0 = zero if value >= 0 else zero - bar_w
    ax.add_patch(patches.Rectangle((x0, y), bar_w, 0.022, facecolor=color, edgecolor="none", alpha=0.92))
    add_text(ax, track_x + track_w + 0.01, y + 0.011, f"{value:+.2f}", size=7.5, color="#475569", ha="left")


def build_template() -> None:
    fig = plt.figure(figsize=(14.0, 8.2), dpi=300, facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[5.0, 0.95],
        width_ratios=[1.12, 0.88],
        hspace=0.07,
        wspace=0.06,
        left=0.04,
        right=0.98,
        top=0.905,
        bottom=0.07,
    )

    ax_in = fig.add_subplot(gs[0, 0])
    ax_out = fig.add_subplot(gs[0, 1])
    ax_strip = fig.add_subplot(gs[1, :])

    fig.text(
        0.04,
        0.962,
        "Online Risk Calculator Interface Template",
        fontsize=15.0,
        fontweight="bold",
        color="#0f172a",
        ha="left",
        va="top",
    )
    fig.text(
        0.04,
        0.928,
        "Publication-ready schematic for the 36-month ADNI risk calculator",
        fontsize=9.2,
        color="#475569",
        ha="left",
        va="top",
    )

    draw_browser_frame(ax_in, "ADNI 36-Month Risk Calculator")
    add_panel_label(ax_in, "A", "Input interface")
    rounded(ax_in, 0.05, 0.79, 0.42, 0.06, fc="#ecfeff", ec="#99f6e4", lw=0.9, r=0.018)
    add_text(ax_in, 0.07, 0.817, "Example case loaded from training defaults", size=8.0, weight="bold", color="#0f766e")

    draw_section(
        ax_in,
        0.05,
        0.57,
        0.90,
        0.18,
        "Demographics",
        "#0ea5e9",
        [
            ("Age at baseline", "72"),
            ("Sex", "Female"),
            ("Education (years)", "16"),
            ("Race category", "White"),
        ],
    )
    draw_section(
        ax_in,
        0.05,
        0.33,
        0.90,
        0.18,
        "Cognitive profile",
        "#8b5cf6",
        [
            ("MMSE score", "26"),
            ("ADAS-Cog13", "18"),
            ("FAQ total", "4"),
            ("Clinical note", "Stable memory complaints"),
        ],
    )
    draw_section(
        ax_in,
        0.05,
        0.10,
        0.90,
        0.16,
        "MRI summary",
        "#14b8a6",
        [
            ("Hippocampal volume", "0.58"),
            ("Ventricular volume", "0.31"),
            ("Cortical thickness", "2.41"),
        ],
    )
    add_text(
        ax_in,
        0.06,
        0.045,
        "All predictors follow the final 36-month model definition.",
        size=7.6,
        color="#64748b",
    )

    draw_browser_frame(ax_out, "Prediction output")
    add_panel_label(ax_out, "B", "Result summary")
    draw_result_badge(
        ax_out,
        0.06,
        0.72,
        0.88,
        0.13,
        "Predicted 36-month conversion probability",
        "36.4%",
        fc="#ecfdf5",
        ec="#bbf7d0",
        title_color="#047857",
        value_color="#065f46",
    )
    draw_result_badge(
        ax_out,
        0.06,
        0.57,
        0.42,
        0.10,
        "Risk group",
        "Intermediate",
        fc="#fffbeb",
        ec="#fde68a",
        title_color="#92400e",
        value_color="#78350f",
    )
    draw_result_badge(
        ax_out,
        0.52,
        0.57,
        0.42,
        0.10,
        "Model quality",
        "AUC 0.83",
        fc="#eff6ff",
        ec="#bfdbfe",
        title_color="#1d4ed8",
        value_color="#1e3a8a",
    )

    rounded(ax_out, 0.06, 0.13, 0.88, 0.34, fc="#ffffff", ec="#dbe4ee", lw=0.9, r=0.02)
    ax_out.add_patch(patches.Rectangle((0.06, 0.42), 0.88, 0.035, facecolor="#f8fafc", edgecolor="none"))
    add_text(ax_out, 0.08, 0.438, "Top feature contributions", size=8.8, weight="bold", color="#0f172a")
    draw_contrib_bar(ax_out, 0.08, 0.34, 0.80, "ADAS-Cog13", +0.23, "#0f766e")
    draw_contrib_bar(ax_out, 0.08, 0.29, 0.80, "MMSE score", -0.15, "#ef4444")
    draw_contrib_bar(ax_out, 0.08, 0.24, 0.80, "Hippocampal volume", +0.11, "#0ea5e9")
    draw_contrib_bar(ax_out, 0.08, 0.19, 0.80, "Age at baseline", +0.07, "#f59e0b")
    draw_contrib_bar(ax_out, 0.08, 0.14, 0.80, "FAQ total", +0.05, "#8b5cf6")
    ax_out.add_line(Line2D([0.11, 0.91], [0.11, 0.11], lw=1.0, color="#cbd5e1"))
    add_text(
        ax_out,
        0.08,
        0.08,
        "The calculator returns probability, risk group, and feature-level contributions.",
        size=7.6,
        color="#64748b",
    )

    ax_strip.axis("off")
    rounded(ax_strip, 0.01, 0.18, 0.98, 0.64, fc="#f8fafc", ec="#dbe4ee", lw=0.9, r=0.02)
    add_text(ax_strip, 0.03, 0.68, "Suggested manuscript caption text", size=9.5, weight="bold", color="#0f172a")
    ax_strip.text(
        0.03,
        0.38,
        "Panel A shows the input workflow with grouped clinical predictors. "
        "Panel B shows the predicted probability, assigned risk group, and feature contributions.",
        fontsize=8.6,
        color="#475569",
        ha="left",
        va="center",
        wrap=True,
    )
    ax_strip.text(
        0.03,
        0.18,
        "Research use only. Not intended for autonomous clinical decision-making.",
        fontsize=7.9,
        color="#64748b",
        ha="left",
        va="center",
        style="italic",
    )

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")


if __name__ == "__main__":
    build_template()
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")
