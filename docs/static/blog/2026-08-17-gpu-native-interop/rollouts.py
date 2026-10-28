"""Computes the Multi-Agent-DPC rollouts drawn in multi-agent-dpc.svg and saves them to rollouts.npz.

Runs the team's trained decentralized policy through their 1D heat solver. Needs a
checkout of https://github.com/SOLARIS-JHU/Multi-Agent-DPC and its dependencies:

    python rollouts.py /path/to/Multi-Agent-DPC
"""

import sys
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
MADPC = Path(sys.argv[1]).resolve()
EXAMPLE = MADPC / "examples/heat1d/decentralized"
sys.path[:0] = [str(MADPC / "tesseracts/solverHeat_decentralized"), str(EXAMPLE)]

import data_utils  # noqa: E402
import solver  # noqa: E402
from models.policy import DecentralizedControlNet  # noqa: E402

N_PDE, N_AGENTS, T = 100, 8, 300
MAIN_SEED, OTHER_SEEDS = 2024, (1, 42, 99)

model = DecentralizedControlNet(features=(64, 64))
template = model.init(
    jax.random.PRNGKey(0),
    jnp.zeros((N_PDE,)),
    jnp.zeros((N_PDE,)),
    jnp.zeros((N_AGENTS,)),
)
params = flax.serialization.from_bytes(
    template, (EXAMPLE / "decentralized_params.msgpack").read_bytes()
)


def rollout(seed):
    """One random start and target, and the trained policy's rollout between them."""
    _, k1, k2 = jax.random.split(jax.random.PRNGKey(seed), 3)
    _, z0 = data_utils.generate_grf(k1, n_points=N_PDE, length_scale=0.2)
    _, zt = data_utils.generate_grf(k2, n_points=N_PDE, length_scale=0.4)
    xi0 = jnp.linspace(0.2, 0.8, N_AGENTS)
    z, xi, u, _ = solver.solve_with_policy(z0, xi0, zt, params, model.apply, T)
    rms = [float(jnp.sqrt(jnp.mean((a - zt) ** 2))) for a in (z0, z[-1])]
    print(f"seed {seed}: rms error {rms[0]:.3f} at the start, {rms[1]:.3f} at the end")
    return z0, zt, z, xi0, xi, u


# The main example is drawn in full; the others only as start and target thumbnails.
z0, zt, z, xi0, xi, u = rollout(MAIN_SEED)
out = {"z0": z0, "zt": zt, "z": z, "xi0": xi0, "xi_end": xi[-1], "u0": u[0]}
for seed in OTHER_SEEDS:
    z0, zt, *_ = rollout(seed)
    out[f"z0_{seed}"], out[f"zt_{seed}"] = z0, zt
np.savez_compressed(HERE / "rollouts.npz", **{k: np.asarray(v) for k, v in out.items()})
