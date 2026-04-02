# Sparse CHOLMOD solver for SPD block systems with tridiagonal blocks.
#
# Assembles a sparse SparseMatrixCSC from tridiagonal block diagonals
# (identified by their Tesseract paths) and solves via CHOLMOD.
#
# Contract:
#   apply_jl(diff_args, non_diff_args, diff_paths, non_diff_paths) -> Vector{Float64}

using LinearAlgebra, SparseArrays, LinearSolve

function apply_jl(diff_args, non_diff_args, diff_paths, non_diff_paths)
    # Parse block_sizes from non_diff inputs
    block_sizes = Int[]
    for (i, path) in enumerate(non_diff_paths)
        if startswith(path, "block_sizes.")
            push!(block_sizes, Int(non_diff_args[i]))
        end
    end

    N = sum(block_sizes)
    offsets = cumsum([0; block_sizes[1:end-1]]) .+ 1

    # Build sparse matrix from tridiagonal block diagonals
    I_idx = Int[]; J_idx = Int[]; V = Float64[]
    b = zeros(N)

    for (k, path) in enumerate(diff_paths)
        arr = diff_args[k]
        path == "b" && (b .= arr; continue)

        m = match(r"^blocks\.\[(\d+)\]\.\[(\d+)\]\.(sub|main|sup)$", path)
        m === nothing && continue

        bi = parse(Int, m[1]) + 1  # 1-indexed
        bj = parse(Int, m[2]) + 1
        comp = m[3]
        r0 = offsets[bi]
        c0 = offsets[bj]

        if comp == "main"
            for idx in 1:length(arr)
                push!(I_idx, r0+idx-1); push!(J_idx, c0+idx-1); push!(V, arr[idx])
            end
        elseif comp == "sub"
            for idx in 1:length(arr)
                push!(I_idx, r0+idx); push!(J_idx, c0+idx-1); push!(V, arr[idx])
            end
        elseif comp == "sup"
            for idx in 1:length(arr)
                push!(I_idx, r0+idx-1); push!(J_idx, c0+idx); push!(V, arr[idx])
            end
        end
    end

    A_raw = sparse(I_idx, J_idx, V, N, N)
    # Materialize symmetry into a plain SparseMatrixCSC (avoids Symmetric wrapper
    # which has a known bug with Enzyme reverse-mode in LinearSolve)
    A_sym = sparse(Symmetric(A_raw))

    prob = LinearProblem(A_sym, b)
    sol = solve(prob, CHOLMODFactorization())
    return copy(sol.u)
end
