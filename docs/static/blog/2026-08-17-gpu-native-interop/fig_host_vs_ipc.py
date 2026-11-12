"""Draws host-vs-ipc.svg: a GPU array reaching another process through host memory, and with cuda_ipc."""

# ruff: noqa: E501  (SVG markup in long f-strings)

import base64

from common import (
    FAINT,
    GPU,
    GPU_B,
    GPU_T,
    HERE,
    HOST_B,
    HOST_T,
    HTTP,
    INK,
    INK2,
    MONO,
    MUTED,
    SANS,
    SOLVER,
    TRAIN,
    gpu_chip,
    heat_field,
    logo_defs,
    markers,
    ram_stick,
    text,
    tile,
    tile_symbol,
    use,
)

W, H = 1600, 950
field = heat_field()
b64 = base64.b64encode(field.tobytes()).decode()

TILE_W, TILE_H = 120, 72
LX, RX = 200, W - 200 - TILE_W
LC, RC = LX + TILE_W / 2, RX + TILE_W / 2
RIVER_X0, RIVER_X1 = LX + TILE_W + 46, RX - 46
HOST_H, GAP, GPU_H = 170, 34, 136
WHITE_HALO = 'paint-order="stroke" stroke="#ffffff" stroke-width="8"'

s = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{SANS}">',
    "<defs>",
    markers(),
    logo_defs(),
    tile_symbol("field", field),
    f'<clipPath id="river-clip"><rect x="{RIVER_X0}" y="0" width="{RIVER_X1 - RIVER_X0}" height="{H}"/></clipPath>',
    "</defs>",
    f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
]

# Column headers
s += [
    use("logo-jax", 222, 26, 74, 43),
    text(310, 58, "TRAINING PROCESS", 22, INK, extra='letter-spacing="1"'),
    use("logo-tesseract", 1086, 22, 50, 50),
    text(1148, 58, "SOLVER PROCESS", 22, INK, extra='letter-spacing="1"'),
]


def lanes(top, gpu_label):
    hy = top + 50
    gy = hy + HOST_H + GAP
    out = [
        # The process boundary splits host memory, not the GPU.
        f'<line x1="800" y1="{top + 4}" x2="800" y2="{gy - 4}" stroke="{FAINT}" stroke-width="2.5" stroke-dasharray="10 8"/>',
        f'<rect x="40" y="{hy}" width="730" height="{HOST_H}" rx="10" fill="{HOST_T}" stroke="{HOST_B}" stroke-width="1.5"/>',
        f'<rect x="830" y="{hy}" width="730" height="{HOST_H}" rx="10" fill="{HOST_T}" stroke="{HOST_B}" stroke-width="1.5"/>',
        ram_stick(58, hy + 15, 32, 18),
        text(100, hy + 30, "HOST MEMORY", 15, MUTED, extra='letter-spacing="1"'),
        ram_stick(848, hy + 15, 32, 18),
        text(890, hy + 30, "HOST MEMORY", 15, MUTED, extra='letter-spacing="1"'),
        f'<rect x="40" y="{gy}" width="1520" height="{GPU_H}" rx="10" fill="{GPU_T}" stroke="{GPU_B}" stroke-width="1.5"/>',
        gpu_chip(56, gy + 11, 26),
        text(92, gy + 30, gpu_label, 15, GPU, extra='letter-spacing="1"'),
    ]
    return out, hy, gy


def arrow(x1, y1, x2, y2, kind="ink", width=3.5, col=INK2):
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{width}" '
        f'marker-end="url(#arr-{kind})"/>'
    )


# Panel A: through host memory
top = 100
s.append(text(40, top + 30, "Default: through host memory", 24, INK, weight="bold"))
out, hy, gy = lanes(top, "GPU MEMORY")
s += out
ry, rh = hy + 52, 92
mid = ry + rh / 2
h_tile_y = mid - TILE_H / 2
g_tile_y = gy + 46
gap_mid = gy - 11
s += [
    tile("field", LX, g_tile_y, TILE_W, TILE_H, TRAIN),
    tile("field", RX, g_tile_y, TILE_W, TILE_H, SOLVER),
    tile("field", LX, h_tile_y, TILE_W, TILE_H, FAINT),
    tile("field", RX, h_tile_y, TILE_W, TILE_H, FAINT),
    arrow(LC, g_tile_y - 4, LC, h_tile_y + TILE_H + 6),
    text(LC - 14, gap_mid, "copy to host", 17, INK2, anchor="end"),
    arrow(RC, h_tile_y + TILE_H + 4, RC, g_tile_y - 6),
    text(RC + 14, gap_mid, "copy to GPU", 17, INK2),
]

# A 3.84 MB array as base64 text, crossing the boundary
s.append(
    f'<rect x="{RIVER_X0}" y="{ry}" width="{RIVER_X1 - RIVER_X0}" height="{rh}" rx="6" fill="#fff7ed" stroke="#fdba74" stroke-width="1.5"/>'
)
s.append('<g clip-path="url(#river-clip)">')
nchar = 130
for i in range(6):
    s.append(
        text(
            RIVER_X0 + 6,
            ry + 19 + i * 14.5,
            b64[i * nchar : (i + 1) * nchar],
            12,
            "#c2410c",
            family=MONO,
            extra='opacity="0.8"',
        )
    )
s.append("</g>")
s += [
    arrow(LX + TILE_W + 4, mid, RIVER_X0 - 4, mid),
    arrow(RIVER_X1, mid, RX - 6, mid),
    text(RIVER_X0, ry - 9, "encode", 16, INK2),
    text(RIVER_X1, ry - 9, "decode", 16, INK2, anchor="end"),
    text(
        800, ry - 9, "HTTP", 17, HTTP, anchor="middle", weight="bold", extra=WHITE_HALO
    ),
    text(
        800,
        gy - 11,
        "a 3.84 MB array becomes 5.1 million characters of base64",
        17,
        "#9a3412",
        anchor="middle",
        extra=WHITE_HALO,
    ),
]

# Panel B: cuda_ipc
top = 530
s.append(
    text(40, top + 30, 'gpu_transport="cuda_ipc"', 24, INK, weight="bold", family=MONO)
)
out, hy, gy = lanes(top, "GPU MEMORY")
s += out
g_tile_y = gy + 46
gm = g_tile_y + TILE_H / 2
hm = hy + 52 + 46
gap_mid = gy - 11
s += [
    tile("field", LX, g_tile_y, TILE_W, TILE_H, TRAIN),
    tile("field", RX, g_tile_y, TILE_W, TILE_H, SOLVER),
    arrow(LX + TILE_W + 4, gm, RX - 6, gm, "gpu", 5, GPU),
    text(
        800,
        gm - 14,
        "device-to-device copy, on the one GPU both processes share",
        17,
        GPU,
        anchor="middle",
        extra=f'paint-order="stroke" stroke="{GPU_T}" stroke-width="8"',
    ),
]

# The handle takes the host path, but it is tiny
handle = "80c3660b…:xAz7Rvl/rnXS…:0:3840000"
pw = 340
s += [
    (
        f'<path d="M{LC},{g_tile_y - 4} V{hm} H{800 - pw / 2 - 4}" fill="none" stroke="{MUTED}" stroke-width="2.2" '
        f'stroke-dasharray="5 5" marker-end="url(#arr-muted)"/>'
    ),
    (
        f'<path d="M{800 + pw / 2},{hm} H{RC} V{g_tile_y - 6}" fill="none" stroke="{MUTED}" stroke-width="2.2" '
        f'stroke-dasharray="5 5" marker-end="url(#arr-muted)"/>'
    ),
    text(LC - 14, gap_mid, "export handle", 17, INK2, anchor="end"),
    text(RC + 14, gap_mid, "map handle", 17, INK2),
    f'<rect x="{800 - pw / 2}" y="{hm - 18}" width="{pw}" height="36" rx="18" fill="#fff7ed" stroke="#fdba74" stroke-width="1.5"/>',
    text(800, hm + 5, handle, 14, "#c2410c", anchor="middle", family=MONO),
    text(
        800,
        hy + 52 - 9,
        "HTTP",
        17,
        HTTP,
        anchor="middle",
        weight="bold",
        extra=WHITE_HALO,
    ),
    text(
        800,
        gy - 11,
        "64-byte handle to GPU memory",
        17,
        "#9a3412",
        anchor="middle",
        extra=WHITE_HALO,
    ),
]

s.append("</svg>")
(HERE / "host-vs-ipc.svg").write_text("\n".join(s) + "\n")
