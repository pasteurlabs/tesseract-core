# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Julia side of tesseract_core.runtime.julia_recipes: calls the user's
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
        _vecs(diff_args),
        Any[pyconvert(Any, a) for a in non_diff_args],
        _strings(diff_paths),
        _strings(non_diff_paths),
    )
end

_to_python(outputs::NamedTuple) = Dict(String(k) => v for (k, v) in pairs(outputs))

# Duplicated for the inputs being differentiated, Const for the rest, so Enzyme
# does no work for inputs that were not requested.
function _annotate(args, paths, shadow_of)
    return [haskey(shadow_of, p) ? Duplicated(a, shadow_of[p]) : Const(a) for (a, p) in zip(args, paths)]
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
    args, statics, dpaths, spaths = _to_julia(diff_args, non_diff_args, diff_paths, non_diff_paths)
    shadow_of = Dict(zip(_strings(jvp_paths), _vecs(tangents)))
    f(d...) = apply_fn(collect(d), statics, dpaths, spaths)
    out = autodiff(set_runtime_activity(Forward), Const(f), _annotate(args, dpaths, shadow_of)...)
    return _to_python(out[1])
end

"""
    vjp(apply_fn, diff_args, non_diff_args, diff_paths, non_diff_paths, vjp_paths, output_names, cotangents)

Reverse-mode AD. `cotangents[i]` is the cotangent of the output named
`output_names[i]`. Returns the gradient of every input in `vjp_paths`.
"""
function vjp(apply_fn, diff_args, non_diff_args, diff_paths, non_diff_paths, vjp_paths, output_names, cotangents)
    args, statics, dpaths, spaths = _to_julia(diff_args, non_diff_args, diff_paths, non_diff_paths)
    active = Set(_strings(vjp_paths))
    shadow_of = Dict(p => zero(a) for (a, p) in zip(args, dpaths) if p in active)
    names = Symbol.(_strings(output_names))
    cts = _vecs(cotangents)
    function f(d...)
        out = apply_fn(collect(d), statics, dpaths, spaths)
        s = 0.0
        for (name, ct) in zip(names, cts)
            s += dot(ct, getproperty(out, name))
        end
        return s
    end
    autodiff(set_runtime_activity(Reverse), Const(f), Active, _annotate(args, dpaths, shadow_of)...)
    return shadow_of
end

end
