"""Palette, logos, and SVG helpers shared by the figure scripts in this directory."""

# ruff: noqa: E501  (SVG markup in long f-strings)

import base64
from pathlib import Path

import numpy as np
from matplotlib import colormaps

HERE = Path(__file__).resolve().parent
LOGOS = HERE / "logos"

INK = "#0f172a"
INK2 = "#334155"
MUTED = "#64748b"
FAINT = "#94a3b8"
TRAIN = "#2563eb"
TRAIN_T = "#eff6ff"
SOLVER = "#16a34a"
SOLVER_T = "#f0fdf4"
GPU = "#7c3aed"
GPU_T = "#f5f3ff"
GPU_B = "#c4b5fd"
HOST_T = "#f8fafc"
HOST_B = "#cbd5e1"
HTTP = "#ea580c"

SANS = "Helvetica Neue, Helvetica, Arial, sans-serif"
MONO = "Menlo, DejaVu Sans Mono, monospace"


def b64file(path):
    return base64.b64encode(open(path, "rb").read()).decode()


def logo_symbol(sid, path, vb):
    mime = "image/png" if path.endswith(".png") else "image/svg+xml"
    w, h = vb
    return (
        f'<symbol id="{sid}" viewBox="0 0 {w} {h}">'
        f'<image width="{w}" height="{h}" href="data:{mime};base64,{b64file(path)}"/></symbol>'
    )


def logo_defs():
    return "\n".join(
        [
            logo_symbol("logo-jax", f"{LOGOS}/jax_logo.svg", (451, 260.81)),
            logo_symbol("logo-tesseract", f"{LOGOS}/tesseract-120.png", (120, 120)),
            logo_symbol("logo-python", f"{LOGOS}/Python-logo-notext.svg", (110, 110)),
            logo_symbol("logo-cpp", f"{LOGOS}/ISO_C++_Logo.svg", (306, 344.35)),
            '<symbol id="logo-docker" viewBox="0 0 24 24"><path fill="#2496ED" d="'
            + open(f"{LOGOS}/si-docker.svg").read().split(' d="')[1].split('"')[0]
            + '"/></symbol>',
        ]
    )


def use(sid, x, y, w, h):
    return f'<use href="#{sid}" x="{x}" y="{y}" width="{w}" height="{h}"/>'


def heat_field(nt=300, nx=100):
    """A 1D temperature field driven by moving heat sources, like Multi-Agent-DPC's rollouts."""
    x = np.linspace(0, 1, nx)
    u = np.zeros(nx)
    out = np.empty((nt + 60, nx))
    for t in range(nt + 60):
        pos = [
            0.25 + 0.12 * np.sin(t / 45),
            0.55 + 0.15 * np.sin(t / 60 + 2.0),
            0.8 + 0.08 * np.cos(t / 35),
        ]
        src = sum(np.exp(-(((x - p) / 0.04) ** 2)) for p in pos)
        lap = np.zeros_like(u)
        lap[1:-1] = u[2:] - 2 * u[1:-1] + u[:-2]
        u = u + 0.25 * lap + 0.06 * src - 0.02 * u
        out[t] = u
    return out[60:].astype(np.float32)


def tile_symbol(sid, field, rows=15, cols=24, cmap="inferno", lo=0.08, hi=0.95):
    """Pixelated heatmap tile, one rect per cell, in a 0..cols x 0..rows viewBox."""
    nt, nx = field.shape
    f = field[: nt - nt % rows, : nx - nx % cols]
    f = f.reshape(rows, f.shape[0] // rows, cols, f.shape[1] // cols).mean(axis=(1, 3))
    f = (f - f.min()) / (f.max() - f.min())
    cm = colormaps[cmap]
    cells = []
    for r in range(rows):
        for c in range(cols):
            rgba = cm(lo + (hi - lo) * f[r, c])
            col = "#{:02x}{:02x}{:02x}".format(*tuple(int(255 * v) for v in rgba[:3]))
            cells.append(
                f'<rect x="{c}" y="{r}" width="1.04" height="1.04" fill="{col}"/>'
            )
    return (
        f'<symbol id="{sid}" viewBox="0 0 {cols} {rows}" preserveAspectRatio="none" shape-rendering="crispEdges">'
        + "".join(cells)
        + "</symbol>"
    )


def tile(sid, x, y, w, h, stroke=None):
    s = use(sid, x, y, w, h)
    if stroke:
        s += f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{stroke}" stroke-width="2.5"/>'
    return s


def markers():
    out = []
    for name, col in [("ink", INK2), ("gpu", GPU), ("http", HTTP), ("muted", MUTED)]:
        out.append(
            f'<marker id="arr-{name}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>'
        )
    return "\n".join(out)


def text(
    x,
    y,
    s,
    size=18,
    fill=INK,
    anchor="start",
    weight="normal",
    family=None,
    style="normal",
    extra="",
):
    fam = f' font-family="{family}"' if family else ""
    w = f' font-weight="{weight}"' if weight != "normal" else ""
    st = f' font-style="{style}"' if style != "normal" else ""
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}"{w}{st}{fam} {extra}>'
        f"{s}</text>"
    )


def gpu_chip(x, y, s=34, col=GPU):
    """Small chip glyph: square with pins."""
    p = [
        f'<g stroke="{col}" stroke-width="2.2" fill="none">',
        f'<rect x="{x + s * 0.18}" y="{y + s * 0.18}" width="{s * 0.64}" height="{s * 0.64}" rx="3" fill="#ffffff"/>',
        f'<rect x="{x + s * 0.36}" y="{y + s * 0.36}" width="{s * 0.28}" height="{s * 0.28}" rx="1.5" fill="{col}" stroke="none"/>',
    ]
    for k in (0.32, 0.5, 0.68):
        p += [
            f'<line x1="{x + s * k}" y1="{y}" x2="{x + s * k}" y2="{y + s * 0.18}"/>',
            f'<line x1="{x + s * k}" y1="{y + s * 0.82}" x2="{x + s * k}" y2="{y + s}"/>',
            f'<line x1="{x}" y1="{y + s * k}" x2="{x + s * 0.18}" y2="{y + s * k}"/>',
            f'<line x1="{x + s * 0.82}" y1="{y + s * k}" x2="{x + s}" y2="{y + s * k}"/>',
        ]
    p.append("</g>")
    return "".join(p)


def ram_stick(x, y, w=44, h=22, col=MUTED):
    p = [
        f'<g stroke="{col}" stroke-width="2" fill="none">',
        f'<rect x="{x}" y="{y}" width="{w}" height="{h * 0.7}" rx="2" fill="#ffffff"/>',
    ]
    n = 4
    for i in range(n):
        cx = x + w * (0.12 + 0.76 * i / (n - 1)) - 3.5
        p.append(
            f'<rect x="{cx}" y="{y + h * 0.17}" width="7" height="{h * 0.36}" fill="{col}" stroke="none"/>'
        )
    for i in range(7):
        lx = x + 5 + i * (w - 10) / 6
        p.append(f'<line x1="{lx}" y1="{y + h * 0.7}" x2="{lx}" y2="{y + h}"/>')
    p.append("</g>")
    return "".join(p)
