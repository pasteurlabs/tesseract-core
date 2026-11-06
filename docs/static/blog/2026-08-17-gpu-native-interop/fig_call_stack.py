"""Draws call-stack.svg: one VJP call through Tesseract-JAX, the solver, and the GPU they share."""

# ruff: noqa: E501  (SVG markup in long f-strings)

import numpy as np
from common import (
    FAINT,
    GPU,
    GPU_B,
    GPU_T,
    HERE,
    HTTP,
    INK,
    INK2,
    MONO,
    MUTED,
    SANS,
    SOLVER,
    SOLVER_T,
    TRAIN,
    TRAIN_T,
    gpu_chip,
    heat_field,
    logo_defs,
    markers,
    text,
    tile,
    tile_symbol,
    use,
)

W, H = 1600, 750
field = heat_field()
grad = np.gradient(field, axis=1)

# Rows, top to bottom: training native, training Python | boundary | solver Python, solver native | GPU
ROW_H = 112
TRAIN_Y = 36
R = {
    "t_cpp": TRAIN_Y,
    "t_py": TRAIN_Y + ROW_H,
    "s_py": TRAIN_Y + 2 * ROW_H + 64,
    "s_cpp": TRAIN_Y + 3 * ROW_H + 64,
}
SOLVER_Y = R["s_py"]
GPU_Y = R["s_cpp"] + ROW_H + 30
GPU_H = 150
WALL_Y = (R["t_py"] + ROW_H + SOLVER_Y) / 2
NODE_DY = 34
X0 = 300
GAPS = [92, 104, 92, 162, 124, 120, 100, 150, 96, 104, 92]
XS = [X0 + sum(GAPS[:i]) for i in range(len(GAPS) + 1)]

# (row, title, subtitle, title is code)
STOPS = [
    ("t_py", "jax.grad(loss)", "your training step", True),
    ("t_cpp", "XLA", "runs the step", False),
    ("t_cpp", "C++ shim", "via XLA FFI", False),
    ("t_py", "HTTP client", "exports handles", False),
    ("s_py", "runtime", "maps handles", False),
    ("s_py", "tesseract_api.py", "your solver", True),
    ("s_cpp", "solver's XLA", "computes the VJP", False),
    ("s_py", "runtime", "exports outputs", False),
    ("t_py", "HTTP client", "maps outputs", False),
    ("t_cpp", "C++ shim", "copies results in", False),
    ("t_cpp", "XLA", "carries on", False),
    ("t_py", "grads", "", True),
]
HTTP_AFTER = {3, 7}


def elbow_x(i):
    """Where the path from stop i to stop i+1 changes rows: away from the labels it would cross."""
    (x1, y1), (x2, y2) = node_xy(i), node_xy(i + 1)
    return x2 - 40 if y2 > y1 else x1 + 24


def node_xy(i):
    row = STOPS[i][0]
    return XS[i], R[row] + NODE_DY


def path_markers():
    return "".join(
        f'<marker id="path-{n}" viewBox="0 0 10 10" refX="8" refY="5" markerUnits="userSpaceOnUse" markerWidth="15" '
        f'markerHeight="15" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
        for n, c in [("ink", INK), ("http", HTTP)]
    )


s = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{SANS}">',
    "<defs>",
    markers(),
    path_markers(),
    logo_defs(),
    tile_symbol("field", field),
    tile_symbol("grad", grad, cmap="viridis", lo=0.0, hi=1.0),
    "</defs>",
    f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
]

# Process bands
s += [
    f'<rect x="40" y="{TRAIN_Y}" width="1520" height="{2 * ROW_H}" rx="10" fill="{TRAIN_T}" stroke="#bfdbfe" stroke-width="1.5"/>',
    f'<line x1="40" y1="{TRAIN_Y + ROW_H}" x2="1560" y2="{TRAIN_Y + ROW_H}" stroke="#bfdbfe" stroke-width="1.5"/>',
    f'<rect x="40" y="{SOLVER_Y}" width="1520" height="{2 * ROW_H}" rx="10" fill="{SOLVER_T}" stroke="#bbf7d0" stroke-width="1.5"/>',
    f'<line x1="40" y1="{SOLVER_Y + ROW_H}" x2="1560" y2="{SOLVER_Y + ROW_H}" stroke="#bbf7d0" stroke-width="1.5"/>',
    f'<line x1="40" y1="{WALL_Y}" x2="1560" y2="{WALL_Y}" stroke="{FAINT}" stroke-width="2.5" stroke-dasharray="10 8"/>',
]


# Gutter: group and row labels
def group_label(y0, h, label, col):
    cy = y0 + h / 2
    return text(
        0,
        0,
        label,
        14,
        col,
        anchor="middle",
        extra=f'letter-spacing="1.5" transform="translate(64 {cy}) rotate(-90)"',
    )


s += [
    group_label(TRAIN_Y, 2 * ROW_H, "TRAINING PROCESS", TRAIN),
    group_label(SOLVER_Y, 2 * ROW_H, "SOLVER CONTAINER", SOLVER),
    f'<line x1="82" y1="{TRAIN_Y + 14}" x2="82" y2="{TRAIN_Y + 2 * ROW_H - 14}" stroke="#bfdbfe" stroke-width="1.5"/>',
    f'<line x1="82" y1="{SOLVER_Y + 14}" x2="82" y2="{SOLVER_Y + 2 * ROW_H - 14}" stroke="#bbf7d0" stroke-width="1.5"/>',
]
for key, row_y in R.items():
    y = row_y + NODE_DY
    if key.endswith("py"):
        s += [
            use("logo-python", 100, y - 17, 34, 34),
            text(146, y + 7, "Python", 20, INK, weight="bold"),
        ]
    else:
        s += [
            use("logo-cpp", 102, y - 17, 30, 34),
            text(146, y + 7, "C++", 20, INK, weight="bold"),
        ]
s += [use("logo-docker", 1506, SOLVER_Y + 2 * ROW_H - 42, 40, 40)]
s.append(
    text(
        250,
        WALL_Y - 9,
        "process or container boundary",
        14,
        MUTED,
        anchor="middle",
        extra='paint-order="stroke" stroke="#ffffff" stroke-width="6"',
    )
)

# The call path
path = []
for i in range(len(STOPS) - 1):
    x1, y1 = node_xy(i)
    x2, y2 = node_xy(i + 1)
    if i in HTTP_AFTER:
        col, kind, dash, w = HTTP, "http", ' stroke-dasharray="9 6"', 4
    else:
        col, kind, dash, w = INK, "ink", "", 4
    # elbow: horizontal, then vertical, ending just short of the next node
    if y1 == y2:
        d = f"M{x1 + 12},{y1} H{x2 - 13}"
    else:
        sgn = 1 if y2 > y1 else -1
        mx = elbow_x(i)
        d = f"M{x1 + 12},{y1} H{mx - 10} Q{mx},{y1} {mx},{y1 + sgn * 10} V{y2 - sgn * 10} Q{mx},{y2} {mx + 10},{y2} H{x2 - 13}"
    path.append(
        f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{w}"{dash} stroke-linejoin="round" '
        f'marker-end="url(#path-{kind})"/>'
    )
s += path

# HTTP labels on the boundary
HTTP_LABELS = {3: "POST /vector_jacobian_product", 7: "200 OK"}
for i, label in HTTP_LABELS.items():
    s.append(
        text(
            elbow_x(i) + 20,
            WALL_Y + 5,
            label,
            15,
            HTTP,
            weight="bold",
            family=MONO,
            extra='paint-order="stroke" stroke="#ffffff" stroke-width="6"',
        )
    )

# Count the switches: badges where the path crosses between Python and C++ (dark) or between processes (orange)
DIVIDERS = [TRAIN_Y + ROW_H, SOLVER_Y + ROW_H]
lang, proc = 0, 0
for i in range(len(STOPS) - 1):
    (x1, y1), (x2, y2) = node_xy(i), node_xy(i + 1)
    mx = elbow_x(i)
    if i in HTTP_AFTER:
        proc += 1
        by, col, n = WALL_Y, HTTP, proc
    elif y1 != y2:
        lang += 1
        by, col, n = (
            next(d for d in DIVIDERS if min(y1, y2) < d < max(y1, y2)),
            INK,
            lang,
        )
    else:
        continue
    s.append(
        f'<circle cx="{mx}" cy="{by}" r="12" fill="{col}" stroke="#ffffff" stroke-width="2.5"/>'
    )
    s.append(text(mx, by + 5, str(n), 14, "#ffffff", anchor="middle", weight="bold"))

# Nodes and labels
for i, (row, title, sub, code) in enumerate(STOPS):
    x, y = node_xy(i)
    s.append(
        f'<circle cx="{x}" cy="{y}" r="10" fill="#ffffff" stroke="{INK}" stroke-width="3.5"/>'
    )
    fam = MONO if code else None
    size = 15 if code else 16
    halo = f'paint-order="stroke" stroke="{TRAIN_T if row.startswith("t") else SOLVER_T}" stroke-width="7"'
    s.append(
        text(
            x,
            y + 34,
            title,
            size,
            INK,
            anchor="middle",
            weight="bold",
            family=fam,
            extra=halo,
        )
    )
    if sub:
        s.append(
            text(x, y + 54, sub, 14, INK2, anchor="middle", style="italic", extra=halo)
        )
s.append(use("logo-jax", node_xy(0)[0] - 72, node_xy(0)[1] - 15, 52, 30))
for i in (4, 7):
    x, y = node_xy(i)
    s.append(use("logo-tesseract", x - 15, y - 44, 30, 30))

# GPU memory, aligned in time with the stops that touch it
s += [
    f'<rect x="40" y="{GPU_Y}" width="1520" height="{GPU_H}" rx="10" fill="{GPU_T}" stroke="{GPU_B}" stroke-width="1.5"/>',
    gpu_chip(56, GPU_Y + 12, 26),
    text(92, GPU_Y + 31, "ONE GPU", 15, GPU, extra='letter-spacing="1"'),
]
TW, TH = 84, 52
ty = GPU_Y + 58
# (stop index that touches it, symbol, owner colour, label)
BUFS = [
    (1, "field", TRAIN, "XLA's input"),
    (3, "field", TRAIN, "staging copy"),
    (4, "field", SOLVER, "solver's copy"),
    (6, "grad", SOLVER, "VJP result"),
    (9, "grad", TRAIN, "XLA's output"),
]
for k, (i, sym, owner, label) in enumerate(BUFS):
    x = node_xy(i)[0]
    ny = node_xy(i)[1]
    s.append(
        f'<line x1="{x}" y1="{ny + 62}" x2="{x}" y2="{ty - 4}" stroke="{GPU}" stroke-width="1.8" '
        f'stroke-dasharray="3 5" opacity="0.7"/>'
    )
    s.append(tile(sym, x - TW / 2, ty, TW, TH, owner))
    s.append(text(x, ty + TH + 22, label, 14, GPU, anchor="middle"))
    if k:
        px = node_xy(BUFS[k - 1][0])[0]
        s.append(
            f'<line x1="{px + TW / 2 + 4}" y1="{ty + TH / 2}" x2="{x - TW / 2 - 6}" y2="{ty + TH / 2}" '
            f'stroke="{GPU}" stroke-width="3.5" marker-end="url(#arr-gpu)"/>'
        )
        if owner != BUFS[k - 1][2]:
            s.append(
                text(
                    (px + x) / 2,
                    ty + TH / 2 - 10,
                    "IPC",
                    13,
                    GPU,
                    anchor="middle",
                    weight="bold",
                )
            )

s.append("</svg>")
(HERE / "call-stack.svg").write_text("\n".join(s) + "\n")
