# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Julia half of tesseract_core.runtime.julia_recipes, which calls the user's
# `apply_jl` and differentiates it with Enzyme.
#
# All Python objects (PyArray, PyList, PyString) are converted to native Julia
# types before Enzyme sees them. Enzyme traces at the LLVM level and cannot
# differentiate through PythonCall's conversion internals.

module TesseractJuliaRecipes

using Enzyme
using LinearAlgebra: dot
using PythonCall: pyconvert

_vecs(pyargs) = [Vector{Float64}(a) for a in pyargs]
_strings(pyargs) = String[pyconvert(String, s) for s in pyargs]

function _to_julia(diff_args, non_diff_args, diff_paths, non_diff_paths)
    return (
        diff_args = _vecs(diff_args),
        non_diff_args = Any[pyconvert(Any, a) for a in non_diff_args],
        diff_paths = _strings(diff_paths),
        non_diff_paths = _strings(non_diff_paths),
    )
end

_to_python(outputs::NamedTuple) = Dict(String(k) => v for (k, v) in pairs(outputs))

# All differentiable inputs go to Enzyme as a single Duplicated vector, with zero
# shadows for inputs that were not requested. Annotating each input separately
# would compile a new derivative for every distinct set of requested inputs.
_zero_shadows(inputs) = [zero(a) for a in inputs.diff_args]

# apply_fn with the differentiable arrays as its only argument
_closure(apply_fn, inputs) = let rest = (inputs.non_diff_args, inputs.diff_paths, inputs.non_diff_paths)
    d -> apply_fn(d, rest...)
end

function apply(apply_fn, diff_args, non_diff_args, diff_paths, non_diff_paths)
    return _to_python(apply_fn(_to_julia(diff_args, non_diff_args, diff_paths, non_diff_paths)...))
end

"""
    jvp(apply_fn, diff_args, non_diff_args, diff_paths, non_diff_paths, jvp_paths, tangents)

Forward-mode AD. `tangents[i]` is the tangent of the input at `jvp_paths[i]`.
Returns the tangent of every output, keyed by output name.
"""
function jvp(apply_fn, diff_args, non_diff_args, diff_paths, non_diff_paths, jvp_paths, tangents)
    inputs = _to_julia(diff_args, non_diff_args, diff_paths, non_diff_paths)
    shadows = _zero_shadows(inputs)
    for (p, t) in zip(_strings(jvp_paths), _vecs(tangents))
        shadows[findfirst(==(p), inputs.diff_paths)] = t
    end
    f = _closure(apply_fn, inputs)
    out = autodiff(set_runtime_activity(Forward), Const(f), Duplicated(inputs.diff_args, shadows))
    return _to_python(out[1])
end

"""
    vjp(apply_fn, diff_args, non_diff_args, diff_paths, non_diff_paths, vjp_paths, output_names, cotangents)

Reverse-mode AD. `cotangents[i]` is the cotangent of the output named
`output_names[i]`. Returns the gradient of every input in `vjp_paths`.
"""
function vjp(apply_fn, diff_args, non_diff_args, diff_paths, non_diff_paths, vjp_paths, output_names, cotangents)
    inputs = _to_julia(diff_args, non_diff_args, diff_paths, non_diff_paths)
    shadows = _zero_shadows(inputs)
    names = Symbol.(_strings(output_names))
    cts = _vecs(cotangents)
    apply_d = _closure(apply_fn, inputs)
    function f(d)
        out = apply_d(d)
        s = 0.0
        for (name, ct) in zip(names, cts)
            s += dot(ct, getproperty(out, name))
        end
        return s
    end
    autodiff(set_runtime_activity(Reverse), Const(f), Active, Duplicated(inputs.diff_args, shadows))
    requested = Set(_strings(vjp_paths))
    return Dict(p => g for (p, g) in zip(inputs.diff_paths, shadows) if p in requested)
end

end
