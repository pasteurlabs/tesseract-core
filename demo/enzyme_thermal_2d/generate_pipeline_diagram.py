#!/usr/bin/env python3
# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the compilation pipeline diagram for the Enzyme AD blog post."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIGURE_DIR = Path(__file__).parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

# Colors
C_FORT_BG, C_FORT = "#dbeafe", "#2563eb"
C_IR_BG, C_IR = "#fef9c4", "#ca8a04"
C_ENZ_BG, C_ENZ = "#ede9fe", "#7c3aed"
C_SO_BG, C_SO = "#ffedd5", "#ea580c"
C_WRAP_BG, C_WRAP = "#dcfce7", "#16a34a"
C_ARROW = "#9ca3af"
C_TEXT = "#374151"


def draw_box(ax, cx, cy, w, h, line1, line2, bg, edge, lw=1.5, fs=10):
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.04",
        facecolor=bg,
        edgecolor=edge,
        linewidth=lw,
        zorder=2,
    )
    ax.add_patch(p)
    kw = dict(ha="center", va="center", zorder=3, parse_math=False)
    if line2:
        ax.text(
            cx, cy + 0.13, line1, fontsize=fs, fontweight="bold", color="#1f2937", **kw
        )
        ax.text(
            cx, cy - 0.15, line2, fontsize=fs - 2, color="#6b7280", style="italic", **kw
        )
    else:
        ax.text(cx, cy, line1, fontsize=fs, fontweight="bold", color="#1f2937", **kw)


def draw_arrow(ax, x0, y0, x1, y1, rad=0.0):
    a = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=14,
        color=C_ARROW,
        linewidth=1.8,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=0,
        shrinkB=0,
        zorder=1,
    )
    ax.add_patch(a)


def label(ax, x, y, text):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=8,
        color=C_TEXT,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#d1d5db", lw=0.6),
        zorder=4,
    )


def badge(ax, x, y, num, color):
    ax.text(
        x,
        y,
        str(num),
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=color,
        bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec=color, lw=1.3),
        zorder=5,
    )


def main():
    fig, ax = plt.subplots(figsize=(16, 6))

    # Y lanes
    Y_TOP = 4.6
    Y_BOT = 2.4
    Y_MID = 3.5

    # X centers — each column gets >= box width + gap of clearance.
    # Box width is W; column pitch is 3.4 so there is always a ~1.2 gap.
    X = {
        "f90": 0.0,
        "ll": 3.4,
        "opt": 6.8,
        "combined": 10.2,
        "ad": 13.6,
        "so": 17.4,
    }

    W, H = 2.2, 0.85

    ax.set_xlim(X["f90"] - W / 2 - 0.6, X["so"] + (W + 0.6) / 2 + 0.6)
    ax.set_ylim(0.4, 6.0)
    ax.axis("off")

    # ── TOP TRACK: Fortran source path ──
    draw_box(
        ax, X["f90"], Y_TOP, W, H, "thermal_2d.f90", "Fortran source", C_FORT_BG, C_FORT
    )
    draw_box(ax, X["ll"], Y_TOP, W, H, "thermal_2d.ll", "LLVM IR", C_IR_BG, C_IR)
    draw_box(
        ax, X["opt"], Y_TOP, W, H, "thermal_2d_opt.ll", "optimized IR", C_IR_BG, C_IR
    )

    draw_arrow(ax, X["f90"] + W / 2, Y_TOP, X["ll"] - W / 2, Y_TOP)
    label(ax, (X["f90"] + X["ll"]) / 2, Y_TOP - 0.62, "lfortran\n--show-llvm")

    draw_arrow(ax, X["ll"] + W / 2, Y_TOP, X["opt"] - W / 2, Y_TOP)
    label(ax, (X["ll"] + X["opt"]) / 2, Y_TOP + 0.62, "opt -O3")

    # ── BOTTOM TRACK: C wrapper path ──
    draw_box(
        ax, X["f90"], Y_BOT, W, H, "wrapper.c", "Enzyme annotations", C_WRAP_BG, C_WRAP
    )
    draw_box(ax, X["ll"], Y_BOT, W, H, "wrapper.ll", "LLVM IR", C_IR_BG, C_IR)

    draw_arrow(ax, X["f90"] + W / 2, Y_BOT, X["ll"] - W / 2, Y_BOT)
    label(ax, (X["f90"] + X["ll"]) / 2, Y_BOT + 0.62, "clang\n-emit-llvm")

    # ── MERGE into combined.ll ──
    draw_box(ax, X["combined"], Y_MID, W, H, "combined.ll", "linked IR", C_IR_BG, C_IR)

    # both tracks feed llvm-link; arrows enter the left face of combined.ll
    draw_arrow(
        ax, X["opt"] + W / 2, Y_TOP, X["combined"] - W / 2, Y_MID + 0.18, rad=-0.18
    )
    draw_arrow(
        ax, X["ll"] + W / 2, Y_BOT, X["combined"] - W / 2, Y_MID - 0.18, rad=0.18
    )
    # place the merge label in the open wedge to the left of combined.ll
    label(ax, X["combined"] - W / 2 - 1.4, Y_MID, "llvm-link")

    # ── LINEAR: combined -> ad -> .so ──
    draw_box(
        ax,
        X["ad"],
        Y_MID,
        W,
        H,
        "ad.ll",
        "differentiated IR",
        C_ENZ_BG,
        C_ENZ,
        lw=2.5,
    )

    W_SO = W + 0.6
    draw_box(
        ax,
        X["so"],
        Y_MID,
        W_SO,
        H + 0.15,
        "libthermal_2d_ad.so",
        "forward / JVP / VJP",
        C_SO_BG,
        C_SO,
        lw=2.5,
        fs=10,
    )

    draw_arrow(ax, X["combined"] + W / 2, Y_MID, X["ad"] - W / 2, Y_MID)
    label(ax, (X["combined"] + X["ad"]) / 2, Y_MID + 0.62, "opt\n-passes=enzyme")

    draw_arrow(ax, X["ad"] + W / 2, Y_MID, X["so"] - W_SO / 2, Y_MID)
    label(ax, (X["ad"] + X["so"]) / 2, Y_MID + 0.62, "opt +\nclang -shared")

    # ── Step badges, placed just above/below the box they number ──
    badge(ax, X["f90"] - W / 2 + 0.1, Y_TOP + H / 2 + 0.35, 1, C_FORT)
    badge(ax, X["ll"] - W / 2 + 0.1, Y_TOP + H / 2 + 0.35, 2, C_IR)
    badge(ax, X["f90"] - W / 2 + 0.1, Y_BOT - H / 2 - 0.35, 3, C_WRAP)
    badge(ax, X["combined"] - W / 2 + 0.1, Y_MID + H / 2 + 0.35, 4, C_IR)
    badge(ax, X["ad"] - W / 2 + 0.1, Y_MID + H / 2 + 0.35, 5, C_ENZ)
    badge(ax, X["so"] - W_SO / 2 + 0.1, Y_MID + (H + 0.15) / 2 + 0.35, 6, C_SO)

    # ── Tool labels at bottom, centered under the columns they cover ──
    by = 0.9
    for bx, txt, bg, ec in [
        ((X["f90"] + X["ll"]) / 2, "LFortran 0.61", C_FORT_BG, C_FORT),
        ((X["ll"] + X["combined"]) / 2, "LLVM 19", C_IR_BG, C_IR),
        ((X["ad"] + X["so"]) / 2, "Enzyme (LLVM pass)", C_ENZ_BG, C_ENZ),
    ]:
        ax.text(
            bx,
            by,
            txt,
            ha="center",
            va="center",
            fontsize=9,
            color=C_TEXT,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=bg, ec=ec, lw=1),
        )

    # Title
    ax.text(
        (X["f90"] + X["so"]) / 2,
        5.7,
        "Compilation pipeline: Fortran → Enzyme AD → shared library",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#111827",
    )

    fig.savefig(
        FIGURE_DIR / "pipeline.png",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.3,
        facecolor="white",
    )
    plt.close(fig)
    print(f"Saved {FIGURE_DIR / 'pipeline.png'}")


if __name__ == "__main__":
    main()
