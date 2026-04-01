#!/usr/bin/env python
"""End-to-end test of the Bayesian inference demo notebook logic."""

import time

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SA
from tesseract_jax import apply_tesseract

from tesseract_core import Tesseract

# Reduce sample counts for a faster test
NUM_WARMUP = 100
NUM_SAMPLES = 200

# ── Step 1: Serve the JAX Lorenz Tesseract ──────────────────────────────
print("=== Step 1: Serving JAX Lorenz Tesseract ===")
lorenz = Tesseract.from_image("lorenz-bayesian")
lorenz.serve()
print(f"Available endpoints: {lorenz.available_endpoints}")

# ── Step 2: Generate synthetic observations ─────────────────────────────
print("\n=== Step 2: Generating synthetic observations ===")
data = np.load("lorenz96_two_scale_F_18_sample_0_small.npz")
X_states = data["X_states"]
true_trajectory = X_states[500:]
X0 = true_trajectory[0]
x0_jax = jnp.array(X0, dtype=jnp.float32)

OBS_GAP = 10
N_OBS = 3
STD_OBS = 0.5
TRUE_F = 18.0
N_STEPS = OBS_GAP * N_OBS

# Generate self-consistent observations from the model itself
true_result = apply_tesseract(
    lorenz,
    {"state": x0_jax, "F": jnp.float32(TRUE_F), "dt": 0.005, "n_steps": N_STEPS},
)
true_traj = true_result["result"]
obs_indices = jnp.arange(OBS_GAP - 1, N_STEPS, OBS_GAP)
true_obs = true_traj[obs_indices]

key = jax.random.PRNGKey(42)
observations = true_obs + STD_OBS * jax.random.normal(key, true_obs.shape)
print(f"Observations shape: {observations.shape}")

# ── Step 3: Test apply_tesseract works ──────────────────────────────────
print("\n=== Step 3: Testing apply_tesseract ===")
test_result = apply_tesseract(
    lorenz,
    {"state": x0_jax, "F": jnp.float32(TRUE_F), "dt": 0.005, "n_steps": N_STEPS},
)
print(f"apply_tesseract output keys: {list(test_result.keys())}")
print(f"result shape: {test_result['result'].shape}")

# ── Step 4: Test jax.grad flows through ─────────────────────────────────
print("\n=== Step 4: Testing jax.grad through Tesseract ===")


def loss_fn(F_val):
    result = apply_tesseract(
        lorenz,
        {"state": x0_jax, "F": F_val, "dt": 0.005, "n_steps": OBS_GAP},
    )
    return jnp.sum(result["result"] ** 2)


grad_F = jax.grad(loss_fn)(jnp.float32(18.0))
print(f"grad w.r.t. F: {grad_F}")
assert not jnp.isnan(grad_F), "Gradient is NaN!"
print("Gradient OK")

# ── Step 5: Define NumPyro model and run NUTS ───────────────────────────
print("\n=== Step 5: Running NUTS ===")


def bayesian_lorenz_model(observations, x0, obs_gap, n_obs, std_obs):
    F = numpyro.sample("F", dist.Normal(15.0, 5.0))
    result = apply_tesseract(
        lorenz,
        {"state": x0, "F": F, "dt": 0.005, "n_steps": obs_gap * n_obs},
    )
    trajectory = result["result"]
    obs_idx = jnp.arange(obs_gap - 1, obs_gap * n_obs, obs_gap)
    predicted_obs = trajectory[obs_idx]
    numpyro.sample("obs", dist.Normal(predicted_obs, std_obs), obs=observations)


nuts_kernel = NUTS(bayesian_lorenz_model)
mcmc_nuts = MCMC(
    nuts_kernel, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES, num_chains=1
)

start = time.time()
mcmc_nuts.run(
    jax.random.PRNGKey(0),
    observations=observations,
    x0=x0_jax,
    obs_gap=OBS_GAP,
    n_obs=N_OBS,
    std_obs=STD_OBS,
)
nuts_time = time.time() - start
mcmc_nuts.print_summary()

nuts_samples = mcmc_nuts.get_samples()
F_mean = float(nuts_samples["F"].mean())
F_std = float(nuts_samples["F"].std())
print(
    f"\nNUTS: F = {F_mean:.2f} ± {F_std:.2f} (true: {TRUE_F}), time: {nuts_time:.1f}s"
)
assert abs(F_mean - TRUE_F) < 5.0, f"NUTS posterior mean too far from truth: {F_mean}"
print("NUTS posterior check PASSED")

# ── Step 6: Test finite-diff Tesseract ──────────────────────────────────
print("\n=== Step 6: Serving finite-diff Lorenz Tesseract ===")
lorenz_fd = Tesseract.from_image("lorenz-finitediff")
lorenz_fd.serve()
print(f"Available endpoints: {lorenz_fd.available_endpoints}")


def bayesian_lorenz_model_fd(observations, x0, obs_gap, n_obs, std_obs):
    F = numpyro.sample("F", dist.Normal(15.0, 5.0))
    result = apply_tesseract(
        lorenz_fd,
        {"state": x0, "F": F, "dt": 0.005, "n_steps": obs_gap * n_obs},
    )
    trajectory = result["result"]
    obs_idx = jnp.arange(obs_gap - 1, obs_gap * n_obs, obs_gap)
    predicted_obs = trajectory[obs_idx]
    numpyro.sample("obs", dist.Normal(predicted_obs, std_obs), obs=observations)


nuts_fd_kernel = NUTS(bayesian_lorenz_model_fd)
mcmc_nuts_fd = MCMC(
    nuts_fd_kernel, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES, num_chains=1
)

start = time.time()
mcmc_nuts_fd.run(
    jax.random.PRNGKey(0),
    observations=observations,
    x0=x0_jax,
    obs_gap=OBS_GAP,
    n_obs=N_OBS,
    std_obs=STD_OBS,
)
nuts_fd_time = time.time() - start
mcmc_nuts_fd.print_summary()

fd_samples = mcmc_nuts_fd.get_samples()
F_mean_fd = float(fd_samples["F"].mean())
print(
    f"\nNUTS (FD): F = {F_mean_fd:.2f} ± {float(fd_samples['F'].std()):.2f} (true: {TRUE_F}), time: {nuts_fd_time:.1f}s"
)
assert abs(F_mean_fd - TRUE_F) < 5.0, (
    f"NUTS-FD posterior mean too far from truth: {F_mean_fd}"
)
print("NUTS-FD posterior check PASSED")

# ── Step 7: Gradient-free baseline ──────────────────────────────────────
print("\n=== Step 7: Running SA (gradient-free) ===")
sa_kernel = SA(bayesian_lorenz_model)
mcmc_sa = MCMC(
    sa_kernel, num_warmup=NUM_WARMUP * 5, num_samples=NUM_SAMPLES, num_chains=1
)

start = time.time()
mcmc_sa.run(
    jax.random.PRNGKey(0),
    observations=observations,
    x0=x0_jax,
    obs_gap=OBS_GAP,
    n_obs=N_OBS,
    std_obs=STD_OBS,
)
sa_time = time.time() - start
mcmc_sa.print_summary()

sa_samples = mcmc_sa.get_samples()
F_mean_sa = float(sa_samples["F"].mean())
print(
    f"\nSA: F = {F_mean_sa:.2f} ± {float(sa_samples['F'].std()):.2f} (true: {TRUE_F}), time: {sa_time:.1f}s"
)
# SA is expected to perform poorly — no assertion on accuracy

# ── Cleanup ─────────────────────────────────────────────────────────────
print("\n=== Cleanup ===")
lorenz.teardown()
lorenz_fd.teardown()

print("\n=== ALL STEPS PASSED ===")
