# Core computation for {{name}}.
#
# Implements the apply_jl contract of tesseract_core.runtime.julia_recipes.
# Returns a NamedTuple with one entry per field of OutputSchema.

function apply_jl(
    diff_args::Vector{Vector{Float64}},
    non_diff_args::Vector{Any},
    diff_paths::Vector{String},
    non_diff_paths::Vector{String},
)::NamedTuple
    # Example: square the first (and only) differentiable input.
    # Replace the body with your solver.
    return (; y = diff_args[1] .^ 2)
end
