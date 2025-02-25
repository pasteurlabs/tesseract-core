import equinox as eqx
import numpy as np
import yaml
from flatten_dict import flatten, unflatten
from jax import config as jax_config
from jax import jacrev, jit
from pydantic import BaseModel, Field
from tsadar import ThomsonParams, ThomsonScatteringDiagnostic, get_scattering_angles
from tsadar.core.modules import get_filter_spec

from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

jax_config.update("jax_enable_x64", False)

with open("1d-defaults.yaml") as fi:
    defaults = yaml.safe_load(fi)

with open("1d-inputs.yaml") as fi:
    inputs = yaml.safe_load(fi)

defaults = flatten(defaults)
defaults.update(flatten(inputs))
config = unflatten(defaults)

# get scattering angles and weights
config["other"]["lamrangE"] = [
    config["data"]["fit_rng"]["forward_epw_start"],
    config["data"]["fit_rng"]["forward_epw_end"],
]
config["other"]["lamrangI"] = [
    config["data"]["fit_rng"]["forward_iaw_start"],
    config["data"]["fit_rng"]["forward_iaw_end"],
]
config["other"]["npts"] = int(
    config["other"]["CCDsize"][1] * config["other"]["points_per_pixel"]
)
sas = get_scattering_angles(config)
dummy_batch = {
    "i_data": np.array([1]),
    "e_data": np.array([1]),
    "noise_e": np.array([0]),
    "noise_i": np.array([0]),
    "e_amps": np.array([1]),
    "i_amps": np.array([1]),
}
ts_params = ThomsonParams(config["parameters"], num_params=1, batch=True, activate=True)
ts_diag = ThomsonScatteringDiagnostic(config, sas)

# filter all static parameters from the python object required to rebuild the model
_, static_params = eqx.partition(
    ts_params,
    filter_spec=get_filter_spec(cfg_params=config["parameters"], ts_params=ts_params),
)


def initialize_thomson_params(inputs):
    diff_params = inputs.model_dump()

    config["parameters"]["electron"]["ne"]["val"] = diff_params["ne"]
    config["parameters"]["electron"]["Te"]["val"] = diff_params["Te"]
    config["parameters"]["general"]["amp1"]["val"] = diff_params["amp1"]
    config["parameters"]["general"]["amp2"]["val"] = diff_params["amp2"]
    config["parameters"]["general"]["lam"]["val"] = diff_params["lam"]

    _ts_params = ThomsonParams(
        config["parameters"], num_params=1, batch=True, activate=True
    )
    # filter all differentiable parameters from the python object
    diff_params, _ = eqx.partition(
        _ts_params,
        filter_spec=get_filter_spec(
            cfg_params=config["parameters"], ts_params=_ts_params
        ),
    )

    return diff_params


def _ts_diag_wrapper(_diff_params):
    # combine the differentiable parameters with the static parameters
    _all_params = eqx.combine(_diff_params, static_params)
    e_spec, _, _, _ = ts_diag(_all_params, dummy_batch)
    return e_spec


# precompile the jacobian function to speed up the computation
jac_fn = jit(jacrev(_ts_diag_wrapper))
ts_diag_wrapper = jit(_ts_diag_wrapper)


class InputSchema(BaseModel):
    ne: Differentiable[Float64] = Field(description="electron density")
    Te: Differentiable[Float64] = Field(description="electron temperature")
    amp1: Differentiable[Float64] = Field(description="amplitude 1")
    amp2: Differentiable[Float64] = Field(description="amplitude 2")
    lam: Differentiable[Float64] = Field(description="central wavelength")


class OutputSchema(BaseModel):
    electron_spectrum: Differentiable[Array[(None,), Float64]] = Field(
        description="electron spectrum"
    )


def apply(inputs: InputSchema) -> OutputSchema:
    """Apply the model to the inputs and return the electron spectrum.

    Args:
        inputs: InputSchema

    Returns:
        OutputSchema
    """
    diff_params = initialize_thomson_params(inputs)
    e_spec = ts_diag_wrapper(diff_params)
    return OutputSchema(electron_spectrum=np.squeeze(e_spec))


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    _diff_params = initialize_thomson_params(inputs)
    # Compute the Jacobian using the Thomson Scattering Diagnostic model as implemented in tsadar
    jac = jac_fn(_diff_params)

    jac_dict = {"electron_spectrum": {}}
    for jax_input in jac_inputs:
        if jax_input in ["ne", "Te"]:
            jac_dict["electron_spectrum"][jax_input] = np.squeeze(
                getattr(jac.electron, f"normed_{jax_input}")
            )

        elif jax_input in ["amp1", "amp2", "lam"]:
            jac_dict["electron_spectrum"][jax_input] = np.squeeze(
                getattr(jac.general, f"normed_{jax_input}")
            )

        else:
            raise ValueError(f"Unknown input: {jax_input}")

    return jac_dict


def abstract_eval(abstract_inputs):
    return {"electron_spectrum": ShapeDType(shape=(1024), dtype="float64")}
