#!/usr/bin/env python3
# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the compilation pipeline diagram for the Enzyme AD blog post.

Vertical recipe: six numbered steps flow top to bottom, each a single stage of
the Fortran -> Enzyme AD -> shared library pipeline. Artifact boxes are
color-coded by what produced them; the command for each step sits on the arrow
that produces the next artifact.
"""

from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

FIGURE_DIR = Path(__file__).parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

# Colors keyed to the tool that produces each artifact.
C_FORT_BG, C_FORT = "#dbeafe", "#2563eb"  # Fortran source
C_WRAP_BG, C_WRAP = "#dcfce7", "#16a34a"  # C wrapper source
C_IR_BG, C_IR = "#fef9c4", "#ca8a04"  # LLVM IR (LFortran / LLVM)
C_ENZ_BG, C_ENZ = "#ede9fe", "#7c3aed"  # Enzyme-differentiated IR
C_SO_BG, C_SO = "#ffedd5", "#ea580c"  # final shared library
# Arrows carry the mechanism (the flow of compilation), so they lead: dark and
# solid. Box borders recede to a thin, muted line — the fill already encodes the
# stage, so a heavy border would be redundant data-ink.
C_ARROW = "#4b5563"
C_TEXT = "#374151"


def draw_box(ax, cx, cy, w, h, line1, line2, bg, edge, fs=11):
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.04",
        facecolor=bg,
        edgecolor=edge,
        linewidth=0.8,
        zorder=2,
    )
    ax.add_patch(p)
    kw = dict(ha="center", va="center", zorder=3, parse_math=False)
    if line2:
        ax.text(
            cx, cy + 0.16, line1, fontsize=fs, fontweight="bold", color="#1f2937", **kw
        )
        ax.text(
            cx, cy - 0.18, line2, fontsize=fs - 3, color="#4a4f5a", style="italic", **kw
        )
    else:
        ax.text(cx, cy, line1, fontsize=fs, fontweight="bold", color="#1f2937", **kw)


def draw_arrow(ax, x0, y0, x1, y1, rad=0.0):
    a = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=15,
        color=C_ARROW,
        linewidth=2.0,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=0,
        shrinkB=0,
        zorder=1,
    )
    ax.add_patch(a)


def cmd_label(ax, x, y, text, ha="left"):
    # Monospace command annotation sitting beside the arrow it drives. A white
    # halo keeps it legible where it crosses an arrow; no enclosing chip (that
    # border was non-data-ink).
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va="center",
        fontsize=9,
        color=C_TEXT,
        fontfamily="monospace",
        zorder=4,
        linespacing=1.3,
        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
    )


def step_badge(ax, x, y, n, color):
    ax.add_patch(
        Circle(
            (x, y), 0.16, facecolor="white", edgecolor=color, linewidth=1.6, zorder=4
        )
    )
    ax.text(
        x,
        y,
        str(n),
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=color,
        zorder=5,
    )


def main():
    fig, ax = plt.subplots(figsize=(9, 11))

    # One row per artifact, descending. Boxes are centered on the main column;
    # the C-wrapper source sits in a parallel column and merges at the link step.
    X_MAIN = 0.0
    X_SIDE = 3.0
    W, H = 2.6, 0.9

    # Row y-coordinates (top -> bottom).
    Y = {
        "f90": 10.0,
        "ll": 8.3,
        "opt": 6.6,
        "combined": 4.9,
        "ad": 3.2,
        "so": 1.5,
    }
    # The C wrapper lives beside the main column on the rows it spans, then
    # feeds the link step.
    Y_WRAPC = 8.3
    Y_WRAPLL = 6.6

    ax.set_xlim(X_MAIN - W / 2 - 2.2, X_SIDE + W / 2 + 3.4)
    ax.set_ylim(0.5, 11.4)
    ax.axis("off")

    # Command labels sit in the empty left margin, right-aligned toward the
    # main column so each reads as an annotation on the arrow beside it.
    LX = X_MAIN - W / 2 - 0.3

    # ── Main vertical track ──
    draw_box(
        ax,
        X_MAIN,
        Y["f90"],
        W,
        H,
        "thermal_2d.f90",
        "Fortran source",
        C_FORT_BG,
        C_FORT,
    )
    draw_box(ax, X_MAIN, Y["ll"], W, H, "thermal_2d.ll", "raw LLVM IR", C_IR_BG, C_IR)
    draw_box(
        ax, X_MAIN, Y["opt"], W, H, "thermal_2d_opt.ll", "optimized IR", C_IR_BG, C_IR
    )
    draw_box(ax, X_MAIN, Y["combined"], W, H, "combined.ll", "linked IR", C_IR_BG, C_IR)
    draw_box(ax, X_MAIN, Y["ad"], W, H, "ad.ll", "differentiated IR", C_ENZ_BG, C_ENZ)
    draw_box(
        ax,
        X_MAIN,
        Y["so"],
        W + 0.5,
        H + 0.1,
        "libthermal_2d_ad.so",
        "forward / JVP / VJP",
        C_SO_BG,
        C_SO,
    )

    # ── C-wrapper side track ──
    draw_box(
        ax, X_SIDE, Y_WRAPC, W, H, "wrapper.c", "Enzyme annotations", C_WRAP_BG, C_WRAP
    )
    draw_box(ax, X_SIDE, Y_WRAPLL, W, H, "wrapper.ll", "wrapper IR", C_IR_BG, C_IR)

    # ── Vertical arrows + step badges + command labels ──
    # Main-track commands sit in the left margin, right-aligned to the column.
    # Step 1: f90 -> ll
    draw_arrow(ax, X_MAIN, Y["f90"] - H / 2, X_MAIN, Y["ll"] + H / 2)
    step_badge(ax, X_MAIN, (Y["f90"] + Y["ll"]) / 2, 1, C_FORT)
    cmd_label(ax, LX, (Y["f90"] + Y["ll"]) / 2, "lfortran\n--show-llvm", ha="right")

    # Step 2: ll -> opt  (mild -O1 BEFORE Enzyme — the whole point of the post)
    draw_arrow(ax, X_MAIN, Y["ll"] - H / 2, X_MAIN, Y["opt"] + H / 2)
    step_badge(ax, X_MAIN, (Y["ll"] + Y["opt"]) / 2, 2, C_IR)
    cmd_label(ax, LX, (Y["ll"] + Y["opt"]) / 2, "opt -O1", ha="right")

    # Step 3: wrapper.c -> wrapper.ll (side track) — label to the right of it.
    draw_arrow(ax, X_SIDE, Y_WRAPC - H / 2, X_SIDE, Y_WRAPLL + H / 2)
    step_badge(ax, X_SIDE, (Y_WRAPC + Y_WRAPLL) / 2, 3, C_WRAP)
    cmd_label(
        ax, X_SIDE + W / 2, (Y_WRAPC + Y_WRAPLL) / 2, "clang\n-emit-llvm -O1", ha="left"
    )

    # Step 4: opt.ll + wrapper.ll -> combined.ll
    draw_arrow(ax, X_MAIN, Y["opt"] - H / 2, X_MAIN, Y["combined"] + H / 2)
    # wrapper.ll merges in from the side
    draw_arrow(
        ax, X_SIDE - W / 2, Y_WRAPLL, X_MAIN + W / 2, Y["combined"] + 0.05, rad=-0.2
    )
    step_badge(ax, X_MAIN, (Y["opt"] + Y["combined"]) / 2, 4, C_IR)
    cmd_label(ax, LX, (Y["opt"] + Y["combined"]) / 2, "llvm-link", ha="right")

    # Step 5: combined.ll -> ad.ll  (Enzyme)
    draw_arrow(ax, X_MAIN, Y["combined"] - H / 2, X_MAIN, Y["ad"] + H / 2)
    step_badge(ax, X_MAIN, (Y["combined"] + Y["ad"]) / 2, 5, C_ENZ)
    cmd_label(ax, LX, (Y["combined"] + Y["ad"]) / 2, "opt\n-passes=enzyme", ha="right")

    # Step 6: ad.ll -> .so  (post-Enzyme -O3 is where -O3 belongs)
    draw_arrow(ax, X_MAIN, Y["ad"] - H / 2, X_MAIN, Y["so"] + (H + 0.1) / 2)
    step_badge(ax, X_MAIN, (Y["ad"] + Y["so"]) / 2, 6, C_SO)
    cmd_label(ax, LX, (Y["ad"] + Y["so"]) / 2, "opt -O3\nclang -shared", ha="right")

    # ── Tool brackets down the right margin, spanning the rows each tool owns ──
    # The box fills already encode the tool; the brackets add version context and
    # group the stages without a heavy enclosure.
    band_x = X_SIDE + W / 2 + 1.9
    tick = 0.18
    for y_hi, y_lo, txt, color in [
        (Y["f90"], Y["f90"], "LFortran 0.61", C_FORT),
        (Y["ll"], Y["combined"], "LLVM 19", C_IR),
        (Y["ad"], Y["ad"], "Enzyme\n(LLVM pass)", C_ENZ),
        (Y["so"], Y["so"], "LLVM 19 +\nclang", C_SO),
    ]:
        top, bot = y_hi + H / 2, y_lo - H / 2
        ax.plot(
            [band_x, band_x],
            [bot, top],
            color=color,
            lw=1.4,
            zorder=1,
            solid_capstyle="round",
        )
        # short inward ticks close the bracket at top and bottom
        ax.plot([band_x - tick, band_x], [top, top], color=color, lw=1.4, zorder=1)
        ax.plot([band_x - tick, band_x], [bot, bot], color=color, lw=1.4, zorder=1)
        ax.text(
            band_x + 0.15,
            (y_hi + y_lo) / 2,
            txt,
            ha="left",
            va="center",
            fontsize=9.5,
            color=color,
            fontweight="bold",
            linespacing=1.3,
        )

    # Title
    ax.text(
        (X_MAIN + X_SIDE) / 2,
        11.1,
        "Compilation pipeline:\nFortran → Enzyme AD → shared library",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#111827",
        linespacing=1.3,
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
