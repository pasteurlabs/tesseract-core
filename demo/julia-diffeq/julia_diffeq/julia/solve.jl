"""
Lotka-Volterra ODE solver and adjoint-based gradient computation.

Solves the classic predator-prey system:
    du/dt = αu - βuv
    dv/dt = δuv - γv

Gradient endpoints use SciMLSensitivity's adjoint methods so that
Julia's native AD machinery is what Python consumers get — not finite
differences.

This module is loaded once at import time via JuliaCall and stays
resident for the lifetime of the Tesseract process.
"""
module LotkaVolterraSolver

using DifferentialEquations
using SciMLSensitivity

function lotka_volterra!(du, u, p, t)
    α, β, δ, γ = p
    du[1] = α * u[1] - β * u[1] * u[2]
    du[2] = δ * u[1] * u[2] - γ * u[2]
end

"""
    solve_lotka_volterra(params, u0, tspan, saveat)

Solve the Lotka-Volterra system. Returns (t, u) where u is (n_times, 2).
"""
function solve_lotka_volterra(params, u0, tspan, saveat)
    p = Float64.(collect(params))
    u0_ = Float64.(collect(u0))
    tspan_ = (Float64(tspan[1]), Float64(tspan[2]))
    saveat_ = Float64.(collect(saveat))

    prob = ODEProblem(lotka_volterra!, u0_, tspan_, p)
    sol = solve(prob, Tsit5(); saveat=saveat_)

    t = collect(sol.t)
    u = reduce(hcat, sol.u)'  # (n_times, 2)
    return t, Matrix(u)
end

"""
    vjp_lotka_volterra(params, u0, tspan, saveat, cotangent_trajectory, grad_inputs)

Compute vector-Jacobian product of the Lotka-Volterra solve w.r.t. params
and/or u0, using SciMLSensitivity's adjoint method.

cotangent_trajectory: cotangent for the trajectory output, shape (n_times, 2)
    passed as a list-of-lists from Python
grad_inputs: list of strings ("params", "u0") indicating which gradients to compute

Returns a Dict with gradient arrays for requested inputs.
"""
function vjp_lotka_volterra(params, u0, tspan, saveat, cotangent_trajectory, grad_inputs)
    p = Float64.(collect(params))
    u0_ = Float64.(collect(u0))
    tspan_ = (Float64(tspan[1]), Float64(tspan[2]))
    saveat_ = Float64.(collect(saveat))

    # Convert cotangent from list-of-lists to Matrix{Float64}
    n_times = length(saveat_)
    n_states = 2
    cotangent = zeros(Float64, n_times, n_states)
    for i in 1:n_times
        row = cotangent_trajectory[i]
        for j in 1:n_states
            cotangent[i, j] = Float64(row[j])
        end
    end

    prob = ODEProblem(lotka_volterra!, u0_, tspan_, p)
    sol = solve(prob, Tsit5(); saveat=saveat_)

    # SciMLSensitivity wants dgdu_discrete as a function (out, u, p, t, i)
    # where i is the timepoint index and out is filled with the cotangent.
    function dgdu_discrete(out, u, p, t, i)
        out .= cotangent[i, :]
    end

    function dgdp_discrete(out, u, p, t, i)
        out .= zeros(length(p))
    end

    # adjoint_sensitivities returns (du0, dp)
    du0_adj, dp_adj = adjoint_sensitivities(
        sol, Tsit5();
        dgdu_discrete=dgdu_discrete,
        dgdp_discrete=dgdp_discrete,
        sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
        t=sol.t,
    )

    result = Dict{String, Any}()
    if "params" in grad_inputs
        result["params"] = collect(Float64, vec(dp_adj))
    end
    if "u0" in grad_inputs
        result["u0"] = collect(Float64, vec(du0_adj))
    end
    return result
end

end  # module
