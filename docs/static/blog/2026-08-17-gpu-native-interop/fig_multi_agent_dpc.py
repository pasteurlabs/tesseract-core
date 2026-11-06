"""Draws multi-agent-dpc.svg: the Multi-Agent-DPC task, and how a training step splits across the Tesseract boundary."""

# ruff: noqa: E501, RUF001  (SVG markup in long f-strings; typographic minus, dash, and times signs in labels)

import itertools

import numpy as np
from common import (
    FAINT,
    HERE,
    HOST_B,
    INK,
    INK2,
    MONO,
    MUTED,
    SANS,
    SOLVER,
    SOLVER_T,
    TRAIN_T,
    logo_defs,
    markers,
    text,
    use,
)
from matplotlib import colormaps

W, H = 1600, 760
d = np.load(HERE / "rollouts.npz")  # from rollouts.py
z0, zt, z, xi0, xiT, u0 = d["z0"], d["zt"], d["z"], d["xi0"], d["xi_end"], d["u0"]
x = np.linspace(0, 1, z0.size)
HEAT, COOL, FINAL = "#dc2626", "#2563eb", "#ea580c"
GRAD = "#db2777"
s = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{SANS}">',
    "<defs>",
    markers(),
    logo_defs(),
    f'<marker id="arr-grad" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{GRAD}"/></marker>',
    f'<marker id="arr-solver" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{SOLVER}"/></marker>',
    "</defs>",
    f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
]

# ---------------------------------------------------------------- the problem
PX0, PX1, PY0, PY1 = 70, 680, 190, 470
zmin, zmax = -0.8, 2.0


def px(xx):
    return PX0 + (PX1 - PX0) * xx


def py(zz):
    return PY1 - (PY1 - PY0) * (zz - zmin) / (zmax - zmin)


def curve(zz, col, w, dash="", op=1.0):
    pts = " ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in zip(x, zz, strict=True))
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{w}"{da} opacity="{op}" stroke-linejoin="round"/>'


s += [
    text(40, 52, "THE PROBLEM", 15, MUTED, extra='letter-spacing="1.5"'),
    text(40, 88, "Shape the temperature along a rod", 26, INK, weight="bold"),
    text(40, 118, "by learning how 8 heaters should heat, cool, and move", 17, INK2),
]
s.append(
    f'<line x1="{PX0}" y1="{py(0)}" x2="{PX1}" y2="{py(0)}" stroke="{HOST_B}" stroke-width="1.5"/>'
)
s.append(text(PX0 - 10, py(0) + 5, "0", 14, MUTED, anchor="end"))
s.append(
    text(
        0,
        0,
        "temperature",
        15,
        MUTED,
        anchor="middle",
        extra=f'transform="translate({PX0 - 28} {(PY0 + PY1) / 2}) rotate(-90)"',
    )
)
s.append(curve(z0, FAINT, 3.5))
for t in (3, 8, 16):
    s.append(curve(z[t], FINAL, 2, op=0.28))
s.append(curve(zt, INK, 3, dash="9 7"))
s.append(curve(z[-1], FINAL, 4))
i0 = int(np.argmax(z0))
s.append(text(px(x[i0]) - 70, py(z0[i0]) - 2, "start", 17, MUTED, anchor="end"))
s.append(text(px(0.80) + 4, py(zt.min()) + 34, "target, and the rod", 17, INK2))
s.append(
    text(px(0.80) + 4, py(zt.min()) + 55, "after 300 steps", 17, FINAL, weight="bold")
)

# the rod, coloured by its starting temperature, with the heaters on it
RY, RH = 528, 26
cm = colormaps["RdBu_r"]
lim = 1.9
cells = []
for k in range(z0.size):
    c = cm(0.5 + 0.5 * np.clip(z0[k] / lim, -1, 1))
    cells.append(
        f'<rect x="{px(x[k]) - 3.2:.1f}" y="{RY}" width="6.6" height="{RH}" fill="#%02x%02x%02x"/>'
        % tuple(int(255 * v) for v in c[:3])
    )
s.append(
    f'<clipPath id="rod"><rect x="{PX0}" y="{RY}" width="{PX1 - PX0}" height="{RH}" rx="{RH / 2}"/></clipPath>'
)
s.append('<g clip-path="url(#rod)">' + "".join(cells) + "</g>")
s.append(
    f'<rect x="{PX0}" y="{RY}" width="{PX1 - PX0}" height="{RH}" rx="{RH / 2}" fill="none" stroke="{INK2}" stroke-width="1.5"/>'
)


def heater(cx, cy, heating, r=13):
    col = HEAT if heating else COOL
    sign = "+" if heating else "−"
    return (
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{col}" stroke="#ffffff" stroke-width="2.5"/>'
        + text(cx, cy + 6, sign, 19, "#ffffff", anchor="middle", weight="bold")
    )


hy = RY + RH + 30
# The heaters' actions are what the policy learns: frame them in the gradient colour.
s.append(
    f'<rect x="{PX0 - 14}" y="{hy - 24}" width="{PX1 - PX0 + 28}" height="76" rx="10" fill="#fdf2f8" '
    f'stroke="{GRAD}" stroke-width="2" stroke-dasharray="7 5"/>'
)
for k, (a, b) in enumerate(zip(xi0, xiT, strict=True)):
    if abs(px(b) - px(a)) > 20:
        ay = hy + 24 + 12 * (k % 2)
        s.append(
            f'<line x1="{px(a)}" y1="{hy + 14}" x2="{px(a)}" y2="{ay}" stroke="{INK2}" stroke-width="1.5"/>'
        )
        s.append(
            f'<line x1="{px(a)}" y1="{ay}" x2="{px(b)}" y2="{ay}" stroke="{INK2}" stroke-width="2.5" marker-end="url(#arr-ink)"/>'
        )
for a, uu in zip(xi0, u0, strict=True):
    s.append(
        f'<line x1="{px(a)}" y1="{RY + RH}" x2="{px(a)}" y2="{hy - 13}" stroke="{INK2}" stroke-width="1.5"/>'
    )
    s.append(heater(px(a), hy, uu > 0))
s.append(
    text(
        PX0 - 14,
        hy + 80,
        "LEARNED",
        15,
        GRAD,
        weight="bold",
        extra='letter-spacing="1.5"',
    )
)
s.append(
    text(
        PX0 + 78,
        hy + 80,
        "how hard each heater heats (+) or cools (−), and where it moves",
        16,
        INK2,
    )
)

# ---------------------------------------------------------------- the two parts
BX0, BX1 = 770, 1560
TY0, TY1 = 40, 300
SY0, SY1 = 440, 720


def box(y0, y1, tint, stroke, logo, logo_w, logo_h, label, sub):
    return [
        f'<rect x="{BX0}" y="{y0}" width="{BX1 - BX0}" height="{y1 - y0}" rx="12" fill="{tint}" stroke="{stroke}" stroke-width="2"/>',
        use(logo, BX0 + 22, y0 + 18, logo_w, logo_h),
        text(BX0 + 30 + logo_w, y0 + 44, label, 17, INK, extra='letter-spacing="1.5"'),
        text(BX1 - 22, y0 + 44, sub, 16, INK2, anchor="end", family=MONO),
    ]


s += box(
    TY0, TY1, TRAIN_T, "#93c5fd", "logo-jax", 56, 32, "TRAINING PROCESS", "train.py"
)
s += box(
    SY0,
    SY1,
    SOLVER_T,
    "#86efac",
    "logo-tesseract",
    34,
    34,
    "SOLVER TESSERACT",
    "solverHeat_decentralized",
)


def mini_axes(x0, y0, w, h):
    return f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" rx="6" fill="#ffffff" stroke="{HOST_B}" stroke-width="1.2"/>'


def mini_curve(zz, x0, y0, w, h, col, sw, dash="", lo=-0.8, hi=2.0, op=1.0):
    xs = np.linspace(0, 1, zz.size)
    pts = " ".join(
        f"{x0 + 6 + (w - 12) * a:.1f},{y0 + h - 6 - (h - 12) * (b - lo) / (hi - lo):.1f}"
        for a, b in zip(xs, zz, strict=True)
    )
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{sw}"{da} opacity="{op}"/>'


def step(cx, y, title, sub, code=False):
    fam = MONO if code else None
    return [
        text(cx, y, title, 17, INK, anchor="middle", weight="bold", family=fam),
        text(cx, y + 22, sub, 15, INK2, anchor="middle"),
    ]


# Training process: sample, loss, update
cy0 = TY0 + 76
cw, ch = 170, 96
cols = [BX0 + 140, BX0 + 395, BX0 + 650]
rng = np.random.default_rng(4)
others = [{"z0": d[f"z0_{k}"], "zt": d[f"zt_{k}"]} for k in (1, 42, 99)]
for j, o in enumerate(others):
    ox, oy = cols[0] - cw / 2 + 12 - 6 * j, cy0 + 12 - 6 * j
    s += [
        mini_axes(ox, oy, cw - 24, ch - 24),
        mini_curve(o["z0"], ox, oy, cw - 24, ch - 24, FAINT, 2, lo=-2, hi=1.6),
        mini_curve(
            o["zt"], ox, oy, cw - 24, ch - 24, INK, 2, dash="5 4", lo=-2, hi=1.6
        ),
    ]
s += step(cols[0], cy0 + ch + 34, "32 random rods", "a start and a target each")

lx, ly = cols[1] - cw / 2, cy0
pts_f = [
    (lx + 6 + (cw - 12) * a, ly + ch - 6 - (ch - 12) * (b + 0.8) / 2.8)
    for a, b in zip(x, z[3], strict=True)
]
pts_t = [
    (lx + 6 + (cw - 12) * a, ly + ch - 6 - (ch - 12) * (b + 0.8) / 2.8)
    for a, b in zip(x, zt, strict=True)
]
poly = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts_f + pts_t[::-1])
s += [
    mini_axes(lx, ly, cw, ch),
    f'<polygon points="{poly}" fill="{FINAL}" opacity="0.18"/>',
    mini_curve(zt, lx, ly, cw, ch, INK, 2, dash="5 4"),
    mini_curve(z[3], lx, ly, cw, ch, FINAL, 2.5),
]
s += step(cols[1], cy0 + ch + 34, "loss", "distance to the target, plus penalties")

gx, gy = cols[2], cy0 + ch / 2
s += [
    text(gx, gy - 6, "jax.grad", 22, INK, anchor="middle", weight="bold", family=MONO),
    text(gx, gy + 24, "→ Adam step on θ", 18, INK2, anchor="middle"),
]
s += step(cols[2], cy0 + ch + 34, "update the policy", "11,298 weights θ")
for a, b in (
    (cols[0] + cw / 2 + 6, cols[1] - cw / 2 - 10),
    (cols[1] + cw / 2 + 6, cols[2] - 90),
):
    s.append(
        f'<line x1="{a}" y1="{cy0 + ch / 2}" x2="{b}" y2="{cy0 + ch / 2}" stroke="{INK2}" stroke-width="3" marker-end="url(#arr-ink)"/>'
    )

# Tesseract: policy -> heaters -> heat equation, 300 times
sy = SY0 + 78
pc = [BX0 + 140, BX0 + 395, BX0 + 650]
# policy network glyph
nx0 = pc[0] - 60
layers = [3, 4, 4, 2]
nodes = []
for li, n in enumerate(layers):
    xs_ = nx0 + li * 40
    nodes.append([(xs_, sy + 48 + (k - (n - 1) / 2) * 22) for k in range(n)])
for la, lb in itertools.pairwise(nodes):
    for a in la:
        for b in lb:
            s.append(
                f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="{FAINT}" stroke-width="1"/>'
            )
for la in nodes:
    for a in la:
        s.append(
            f'<circle cx="{a[0]}" cy="{a[1]}" r="7" fill="#ffffff" stroke="{SOLVER}" stroke-width="2.2"/>'
        )
s += step(pc[0], sy + 130, "policy(θ)", "each heater sees the error near it", code=True)
# heaters glyph
for k, (heat, dx) in enumerate([(True, -1), (False, 0), (False, 1), (False, 1)]):
    hx = pc[1] - 66 + k * 44
    s.append(heater(hx, sy + 48, heat, 14))
    if dx:
        s.append(
            f'<line x1="{hx}" y1="{sy + 76}" x2="{hx + 18 * dx}" y2="{sy + 76}" stroke="{INK2}" stroke-width="2.2" marker-end="url(#arr-ink)"/>'
        )
s += step(pc[1], sy + 130, "heat and move", "u and v for every heater")
# heat-equation step glyph: two rod states
rx0, rw = pc[2] - 80, 160
for j, zz in enumerate((z[3], z[4])):
    yy = sy + 22 + j * 40
    cells = []
    for k in range(0, zz.size, 2):
        c = cm(0.5 + 0.5 * np.clip(zz[k] / lim, -1, 1))
        cells.append(
            f'<rect x="{rx0 + rw * x[k]:.1f}" y="{yy}" width="{rw / 49:.2f}" height="16" fill="#%02x%02x%02x"/>'
            % tuple(int(255 * v) for v in c[:3])
        )
    s.append(
        f'<clipPath id="mr{j}"><rect x="{rx0}" y="{yy}" width="{rw}" height="16" rx="8"/></clipPath>'
    )
    s.append(f'<g clip-path="url(#mr{j})">' + "".join(cells) + "</g>")
    s.append(
        f'<rect x="{rx0}" y="{yy}" width="{rw}" height="16" rx="8" fill="none" stroke="{INK2}" stroke-width="1.2"/>'
    )
s.append(
    f'<line x1="{pc[2]}" y1="{sy + 40}" x2="{pc[2]}" y2="{sy + 60}" stroke="{INK2}" stroke-width="2.2" marker-end="url(#arr-ink)"/>'
)
s += step(pc[2], sy + 130, "heat equation", "one Crank–Nicolson step")
for a, b in ((pc[0] + 80, pc[1] - 92), (pc[1] + 92, pc[2] - 92)):
    s.append(
        f'<line x1="{a}" y1="{sy + 48}" x2="{b}" y2="{sy + 48}" stroke="{INK2}" stroke-width="3" marker-end="url(#arr-ink)"/>'
    )
# loop back, x300
ly_ = sy + 178
s.append(
    f'<path d="M{pc[2]},{sy + 160} V{ly_} H{pc[0]} V{sy + 162}" fill="none" stroke="{SOLVER}" stroke-width="3" marker-end="url(#arr-solver)" />'
)
s.append(
    text(
        (pc[0] + pc[2]) / 2,
        ly_ + 6,
        "× 300 steps",
        18,
        SOLVER,
        anchor="middle",
        weight="bold",
        extra=f'paint-order="stroke" stroke="{SOLVER_T}" stroke-width="10"',
    )
)

# Arrows between the two: forward (apply) and backward (VJP)
ay0, ay1 = TY1 + 6, SY0 - 8


def arrow_pair(x0, col, kind, endpoint, down, up, extra=None):
    out = [
        f'<line x1="{x0}" y1="{ay0}" x2="{x0}" y2="{ay1}" stroke="{col}" stroke-width="3" marker-end="url(#arr-{kind})"/>',
        f'<line x1="{x0 + 30}" y1="{ay1}" x2="{x0 + 30}" y2="{ay0}" stroke="{col}" stroke-width="3" marker-end="url(#arr-{kind})"/>',
        text(x0 + 50, ay0 + 36, endpoint, 16, col, weight="bold", family=MONO),
        text(x0 + 50, ay0 + 62, "↓ " + down, 16, INK),
        text(x0 + 50, ay0 + 86, "↑ " + up, 16, INK),
    ]
    if extra:
        out.append(text(x0 + 50, ay0 + 108, extra, 14, INK2, family=MONO))
    return out


s += arrow_pair(
    BX0 + 70,
    INK2,
    "ink",
    "apply",
    "θ, starts, targets",
    "trajectories",
    "z_trajectory: [32, 300, 100]",
)
s += arrow_pair(
    BX0 + 450,
    GRAD,
    "grad",
    "vector_jacobian_product",
    "∂loss / ∂trajectories",
    "∂loss / ∂θ",
)

s.append("</svg>")
(HERE / "multi-agent-dpc.svg").write_text("\n".join(s) + "\n")
