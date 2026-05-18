# Core computation for {{name}}.
#
# Contract: apply_jl receives all inputs split into diff and non-diff,
# with paths describing each value's role. Replace the body with your solver.
#
# Arguments:
#   diff_args::Vector{Vector{Float64}}  — differentiable arrays
#   non_diff_args::Vector{Any}          — static values (ints, strings, etc.)
#   diff_paths::Vector{String}          — Tesseract path for each diff arg
#   non_diff_paths::Vector{String}      — Tesseract path for each non-diff arg

function apply_jl(
    diff_args::Vector{Vector{Float64}},
    non_diff_args::Vector{Any},
    diff_paths::Vector{String},
    non_diff_paths::Vector{String},
)::Vector{Float64}
    # Example: square the first (and only) differentiable input.
    # Replace with your solver.
    return diff_args[1] .^ 2
end
