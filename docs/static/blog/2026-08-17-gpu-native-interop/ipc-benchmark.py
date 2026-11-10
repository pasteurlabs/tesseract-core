"""Times one jitted apply_tesseract call on arrays of increasing size, through the host and with cuda_ipc.

Draws the array-size figure in the blog post "Tesseract is now GPU-native".
Needs a CUDA GPU and

    pip install tesseract-core[runtime] tesseract-jax "jax[cuda13]" equinox matplotlib

    python ipc-benchmark.py --json ipc-benchmark.json --plot ipc-benchmark.png

To redraw the figure from the saved medians without a GPU:

    python ipc-benchmark.py --from-json ipc-benchmark.json --plot ipc-benchmark.png

The Tesseract below only doubles its input, so the times are the cost of the call
itself. Each setup runs in its own fresh processes, which time every array size in
turn after a few warm-up calls. The table reports the median over processes of each
process's median call time.

Setups:
  in-process   the same doubling, jitted in this process (no Tesseract)
  host         Tesseract served by Tesseract.from_source, arrays through host memory
               as base64 inside JSON (the default)
  binref       same, arrays through host memory as json+binref files on /dev/shm
  cuda_ipc     same, arrays passed as CUDA IPC handles
"""

import os

# The client and the Tesseract share one GPU, so neither JAX process should
# claim 75% of its memory at startup (from_source passes this on to the Tesseract).
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

SETUPS = ("in-process", "host", "binref", "cuda_ipc")
SIZES = [int(m * 10**e) for e in range(3, 8) for m in (1, 3)] + [10**8]

TESSERACT_API = """
import equinox as eqx
from pydantic import BaseModel

from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.jax_recipes import jax_abstract_eval, jax_apply


class InputSchema(BaseModel):
    x: Differentiable[Array[(None,), Float32]]


class OutputSchema(BaseModel):
    y: Differentiable[Array[(None,), Float32]]


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    return {"y": 2 * inputs["x"]}


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(**jax_apply(apply_jit, inputs))


def abstract_eval(abstract_inputs):
    return jax_abstract_eval(apply_jit, abstract_inputs)
"""


def time_calls(f, x, warmup=3, min_calls=15, min_seconds=0.5, max_calls=300):
    """Median wall time of f(x) in ms, over at least min_calls calls and min_seconds."""
    for _ in range(warmup):
        f(x).block_until_ready()
    times, start = [], time.perf_counter()
    while len(times) < max_calls and (
        len(times) < min_calls or time.perf_counter() - start < min_seconds
    ):
        t0 = time.perf_counter()
        f(x).block_until_ready()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3


def run_one(setup, api_path):
    """Time every size for one setup in this process; print a JSON line."""
    import jax
    import jax.numpy as jnp
    from tesseract_jax import apply_tesseract

    from tesseract_core import Tesseract

    def sweep(f):
        return {
            n: time_calls(
                f, jax.random.uniform(jax.random.PRNGKey(n), (n,), dtype=jnp.float32)
            )
            for n in SIZES
        }

    def sweep_tesseract(**kwargs):
        with Tesseract.from_source(
            api_path, python_executable=sys.executable, **kwargs
        ) as tess:
            return sweep(jax.jit(lambda x: apply_tesseract(tess, {"x": x})["y"]))

    if setup == "in-process":
        ms = sweep(jax.jit(lambda x: 2 * x))
    elif setup == "host":
        ms = sweep_tesseract()
    elif setup == "binref":
        # Without a GPU transport, GPU arrays go through the host, and json+binref
        # exchanges them as files in these directories instead of inside the JSON.
        with (
            tempfile.TemporaryDirectory(prefix="tess_in_", dir="/dev/shm") as inp,
            tempfile.TemporaryDirectory(prefix="tess_out_", dir="/dev/shm") as out,
        ):
            ms = sweep_tesseract(
                output_format="json+binref", input_path=inp, output_path=out
            )
    else:
        ms = sweep_tesseract(gpu_transport="cuda_ipc")
    print(
        json.dumps(
            {
                "ms": ms,
                "gpu": jax.devices()[0].device_kind,
                "versions": {
                    p: version(p) for p in ("jax", "tesseract-core", "tesseract-jax")
                },
            }
        )
    )


def plot(medians, path):
    """Draw median call time against array size, one line per setup."""
    import matplotlib.pyplot as plt

    # The palette and type of the other figures in the post: purple for data that
    # stays on the GPU, orange for data that goes through host memory, and code
    # names in a monospaced font.
    ink, ink_muted, grid = "#0f172a", "#64748b", "#e2e8f0"
    sans = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    mono = ["Menlo", "DejaVu Sans Mono"]
    style = {
        "host": ("#ea580c", "-", "host (base64)", sans),
        "binref": ("#ea580c", "--", "host (shmem)", sans),
        "cuda_ipc": ("#7c3aed", "-", "cuda_ipc", mono),
        "in-process": ("#94a3b8", "-", "in-process", sans),
    }
    plt.rcParams.update(
        {
            "font.family": sans,
            "font.size": 8,
            "text.color": ink,
            "axes.labelcolor": ink_muted,
            "xtick.color": ink_muted,
            "ytick.color": ink_muted,
        }
    )
    # Same width in pixels as the post's other figures, so text renders at the same size.
    fig, ax = plt.subplots(figsize=(9, 4), dpi=220)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.grid(True, which="major", color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(which="both", length=0)

    med = {s: [medians[s][n] for n in SIZES] for s in medians}
    for setup in ("in-process", "host", "binref", "cuda_ipc"):
        color, linestyle, label, family = style[setup]
        ax.plot(
            SIZES,
            med[setup],
            color=color,
            linestyle=linestyle,
            linewidth=1.6,
            marker="o",
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            label,
            (SIZES[-1], med[setup][-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            fontweight="bold",
            family=family,
            color=color if setup != "in-process" else ink_muted,
        )

    # A compute reference: one Multi-Agent-DPC training step with the solver
    # imported directly, on the same A100 (benchmark.py in the Multi-Agent-DPC fork).
    step_ms = 106
    ax.axhline(step_ms, color=ink_muted, linewidth=0.8, zorder=2)
    ax.annotate(
        f"one Multi-Agent-DPC training step without Tesseract ({step_ms} ms)",
        (SIZES[0], step_ms),
        xytext=(0, 4),
        textcoords="offset points",
        va="bottom",
        color=ink_muted,
    )

    # A call that moves as many bytes as one Multi-Agent-DPC call (about 4.8 MB),
    # counting both directions: each element here is 4 bytes in and 4 bytes out.
    madpc = 4.8e6 / 8
    ax.axvline(madpc, color=ink_muted, linewidth=0.8, linestyle=":", zorder=1)
    ax.annotate(
        "as many bytes as one\nMulti-Agent-DPC call (4.8 MB)",
        (madpc, 0.9),
        xytext=(-6, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        color=ink_muted,
    )

    # How much faster cuda_ipc is than the host round trip at the largest size.
    ratio = med["host"][-1] / med["cuda_ipc"][-1]
    ax.plot(
        [SIZES[-1]] * 2,
        [med["cuda_ipc"][-1] * 1.4, med["host"][-1] / 1.4],
        color=ink_muted,
        linewidth=0.8,
        zorder=2,
    )
    ax.annotate(
        f"{ratio:.0f}\N{MULTIPLICATION SIGN} faster than default\n(host round trip + base64)",
        (SIZES[-1], math.sqrt(med["cuda_ipc"][-1] * step_ms)),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        color=ink_muted,
    )

    ax.set_xlabel("array size (float32 elements)")
    ax.set_ylabel("median time per call (ms)")
    ax.set_xlim(SIZES[0] / 1.5, SIZES[-1] * 1.5)
    fig.tight_layout()
    fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.1)


def measure(processes):
    """Run every setup in `processes` fresh processes each; return per-size medians and metadata."""
    runs = {}
    with tempfile.TemporaryDirectory() as tmp:
        api_path = Path(tmp) / "tesseract_api.py"
        api_path.write_text(TESSERACT_API)
        # Interleave the setups, so slow drift on the machine affects them all alike.
        for _ in range(processes):
            for setup in SETUPS:
                out = subprocess.run(
                    [sys.executable, __file__, "--run", setup, "--api", str(api_path)],
                    capture_output=True,
                    check=False,
                    text=True,
                )
                if out.returncode != 0:
                    sys.exit(f"{setup} failed:\n{out.stderr}")
                runs.setdefault(setup, []).append(
                    json.loads(out.stdout.strip().splitlines()[-1])
                )
    medians = {
        s: {n: statistics.median(r["ms"][str(n)] for r in rs) for n in SIZES}
        for s, rs in runs.items()
    }
    first = runs[SETUPS[0]][0]
    return medians, {
        "gpu": first["gpu"],
        "versions": first["versions"],
        "processes": processes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--json", type=Path, help="write the medians here")
    parser.add_argument("--plot", type=Path, help="draw the figure here")
    parser.add_argument(
        "--from-json", type=Path, help="plot saved medians instead of measuring"
    )
    parser.add_argument("--run", choices=SETUPS, help=argparse.SUPPRESS)
    parser.add_argument("--api", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run:
        run_one(args.run, args.api)
        return

    if args.from_json:
        saved = json.loads(args.from_json.read_text())
        medians = {
            s: {int(n): v for n, v in d.items()} for s, d in saved["medians"].items()
        }
        meta = {k: v for k, v in saved.items() if k != "medians"}
    else:
        medians, meta = measure(args.processes)
        if args.json:
            args.json.write_text(json.dumps({**meta, "medians": medians}, indent=1))

    print(
        f"GPU: {meta['gpu']}, "
        + ", ".join(f"{p} {v}" for p, v in meta["versions"].items())
    )
    print(f"{meta['processes']} processes per setup\n")
    print(
        f"{'elements':>10s}"
        + "".join(f"{s:>12s}" for s in SETUPS)
        + "   host/cuda_ipc  binref/cuda_ipc"
    )
    for n in SIZES:
        row = "".join(f"{medians[s][n]:12.3f}" for s in SETUPS)
        ratios = [medians[s][n] / medians["cuda_ipc"][n] for s in ("host", "binref")]
        print(f"{n:>10d}{row}   {ratios[0]:13.1f}  {ratios[1]:15.1f}")
    if args.plot:
        plot(medians, args.plot)


if __name__ == "__main__":
    main()
