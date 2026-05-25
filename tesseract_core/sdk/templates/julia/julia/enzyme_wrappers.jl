# Generic Enzyme AD wrappers for Tesseract Julia recipes.
#
# These work with any apply_jl that follows the contract:
#   apply_jl(diff_args, non_diff_args, diff_paths, non_diff_paths) -> Vector{Float64}
#
# IMPORTANT: All Python objects (PyArray, PyList, PyString) must be converted
# to native Julia types before Enzyme sees them. Enzyme traces at the LLVM
# level and cannot differentiate through PythonCall's conversion internals.

using LinearAlgebra
using Enzyme
using PythonCall: pyconvert

function _to_jl_vecs(pyargs)
    return [Vector{Float64}(a) for a in pyargs]
end

function _to_jl_strings(pyargs)
    return String[pyconvert(String, s) for s in pyargs]
end

function _to_jl_any(pyargs)
    return Any[pyconvert(Any, a) for a in pyargs]
end

"""
    enzyme_jvp(apply_fn, diff_args, non_diff_args, diff_paths, non_diff_paths, tangents)

Forward-mode AD. Returns the JVP output vector.
"""
function enzyme_jvp(apply_fn, diff_args, non_diff_args, diff_paths, non_diff_paths, tangents)
    jl_args = _to_jl_vecs(diff_args)
    jl_tangents = _to_jl_vecs(tangents)
    jl_non_diff = _to_jl_any(non_diff_args)
    jl_diff_paths = _to_jl_strings(diff_paths)
    jl_non_diff_paths = _to_jl_strings(non_diff_paths)
    closure(d...) = apply_fn(collect(d), jl_non_diff, jl_diff_paths, jl_non_diff_paths)
    dups = [Enzyme.Duplicated(jl_args[i], jl_tangents[i]) for i in eachindex(jl_args)]
    return Enzyme.autodiff(set_runtime_activity(Enzyme.Forward), Enzyme.Const(closure), dups...)[1]
end

"""
    enzyme_vjp(apply_fn, diff_args, non_diff_args, diff_paths, non_diff_paths, cotangent)

Reverse-mode AD. Returns a list of gradients, one per element of diff_args.
"""
function enzyme_vjp(apply_fn, diff_args, non_diff_args, diff_paths, non_diff_paths, cotangent)
    jl_args = _to_jl_vecs(diff_args)
    jl_cotangent = Vector{Float64}(cotangent)
    jl_non_diff = _to_jl_any(non_diff_args)
    jl_diff_paths = _to_jl_strings(diff_paths)
    jl_non_diff_paths = _to_jl_strings(non_diff_paths)
    closure(d...) = apply_fn(collect(d), jl_non_diff, jl_diff_paths, jl_non_diff_paths)
    shadows = [zero(a) for a in jl_args]
    dups = [Enzyme.Duplicated(jl_args[i], shadows[i]) for i in eachindex(jl_args)]
    scalar_f(d...) = dot(jl_cotangent, closure(d...))
    Enzyme.autodiff(set_runtime_activity(Enzyme.Reverse), Enzyme.Const(scalar_f), Enzyme.Active, dups...)
    return shadows
end
