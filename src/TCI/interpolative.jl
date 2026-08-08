using LinearAlgebra: I, Diagonal, UnitUpperTriangular
using ITensors: ITensor, dim, inds, hastags, combiner, combinedind, matrix, space, Index, dag

"""
Returns `C`, `Z`, `piv_cols`, `inf_error` where `C` and 
`Z` are matrices such that `C*Z ≈ M`. The matrix `C` consists 
of columns of `M`, and which column is given by the integer 
entries of the array `piv_cols`. The number of columns of `C` 
is controlled by the approximate rank of `M`, which is 
controlled by the parameters `cutoff` and `maxdim`.
"""
function interpolative(
    M::Matrix;
    pivotsearch::AbstractPivotSearch = FullPivot(),
    kwargs...
)
    # Compute interpolative decomposition (ID) from PRRLU
    L, d, U, pr, pc, inf_error = prrldu(M; pivotsearch, kwargs...)
    U11 = UnitUpperTriangular(U[:, 1:length(d)])
    ZjJ = U11 \ U
    CIj = L * Diagonal(d) * U11
    C = CIj[pr, :]
    Z = ZjJ[:, pc]
    # Compute mapping of pivot columns to column indices
    piv_cols = invperm(pc)[1:length(d)]
    return C, Z, piv_cols, inf_error
end

"""
Compute the interpolative decomposition of an ITensor `T`, treated as a matrix whose
column indices are `col_inds` and whose row indices are the remaining indices of `T`.
"""
function interpolative(
    T::ITensor,
    col_inds::Vector{<:Index};
    pivotsearch::AbstractPivotSearch = FullPivot(),
    cutoff = 0.0,
    maxdim = typemax(Int),
    mindim = 1,
    tags = "Link"
)
    for i in col_inds
        (
            haspivots(i) ||
            hastags(i, "Digit") ||
            error("interpolative requires all indices to have pivots or else \"Digit\" tag")
        )
    end
    # Matricize T
    row_inds = setdiff(inds(T), col_inds)
    Cmb_row, Cmb_col = combiner(row_inds), combiner(col_inds)
    cr, cc = combinedind(Cmb_row), combinedind(Cmb_col)
    t = matrix(Cmb_row * T * Cmb_col, cr, cc)

    # Interpolative decomp of t matrix
    c, z, piv_cols, inf_error = interpolative(t; pivotsearch, cutoff, maxdim, mindim)
    rank = length(piv_cols)
    # Compute mapping of pivot columns to column indices
    col_ranges = [1:dim(i) for i in col_inds]
    col_pivs = [zeros(Int, length(col_ranges)) for c in 1:rank]
    for (col, vals) in enumerate(Iterators.product(col_ranges...))
        loc = findfirst(==(col), piv_cols)
        if !isnothing(loc)
            @assert length(vals) == length(col_inds)
            col_pivs[loc] = collect(vals)
        end
    end

    is_site = [hastags(i, "Digit") for i in col_inds]
    ncols = length(col_inds)

    # Make connecting index with pivot info
    function get_pivs(r, c)
        i = col_inds[c]
        ip = col_pivs[r][c]
        is_site[c] && (return [i => ip])
        return space(i)[ip]
    end
    pivs = [vcat([get_pivs(r, c) for c in 1:ncols]...) for r in 1:rank]
    b = Index(pivs; tags)
    @assert dim(b) == rank

    # Make ITensors from C and Z matrices
    C = ITensor(c, cr, b) * dag(Cmb_row)
    Z = ITensor(z, b, cc) * dag(Cmb_col)

    return C, Z, inf_error
end

"""
Compute the pivoted, rank-revealing LDU decomposition of an
arbitrary matrix `T`.

The vector of pivots `d` has length `k` such that
`norm(L[pr,:] * Diagonal(d) * U[:,pc] - T, Inf) <= abs(d[k])`.

The value of `k` is determined so that both `mindim <= k <= maxdim`
and `abs(d[k]) >= cutoff`.

Returns arrays `L`, `d`, `U`, and permutations `pr` and `pc` such that
`L` and `U` are lower- and upper-triangular matrices with diagonal
values equal to 1 and `L[pr,:] * Diagonal(d) * U[:,pc] ≈ T`, and 
the truncated factorization error `inf_error`.
"""
function prrldu(
    T::Matrix; 
    pivotsearch::AbstractPivotSearch = FullPivot(),
    cutoff::Real = 0.0, 
    maxdim::Int = typemax(Int), 
    mindim::Int = 1
)
    mindim <= maxdim || throw(ArgumentError("mindim ($mindim) must not exceed maxdim ($maxdim)"))
    Elt = eltype(T)
    M = copy(T)
    Nr, Nc = size(M)
    k = min(Nr, Nc)
    rps = collect(1:Nr)
    cps = collect(1:Nc)
    inf_error = zero(real(Elt))
    @inbounds for s in 1:min(k, maxdim)
        Mabs_max, piv = searchpivot(M, pivotsearch)
        if (Mabs_max < cutoff && s > mindim) || iszero(Mabs_max)
            # a non-exhaustive search may have missed a larger entry; confirm before stopping
            isexhaustive(pivotsearch) ||
                ((Mabs_max, piv) = searchpivot(M, FullPivot()))
            (Mabs_max < cutoff || iszero(Mabs_max)) && break
        end
        Base.swaprows!(M, 1, piv[1])
        Base.swapcols!(M, 1, piv[2])
        M = @view(M[2:end, 2:end]) - @view(M[2:end, 1]) * transpose(@view(M[1, 2:end])) / M[1, 1]
        rps[s], rps[piv[1] + s - 1] = rps[piv[1] + s - 1], rps[s]
        cps[s], cps[piv[2] + s - 1] = cps[piv[2] + s - 1], cps[s]
    end
    M = T[rps, cps]
    L = Matrix{Elt}(I, Nr, k)
    d = zeros(Elt, k)
    U = Matrix{Elt}(I, k, Nc)
    rank = 0
    @inbounds for s in 1:min(k, maxdim)
        P = M[s, s]
        d[s] = P
        if rank >= mindim && (iszero(P) || abs(P) < cutoff)
            break
        end
        iszero(P) && (P = one(Elt))
        rank += 1
        piv_col = @view(M[(s + 1):end, s])
        L[(s + 1):end, s] = piv_col / P
        piv_row = @view(M[s, (s + 1):end])
        U[s, (s + 1):end] = piv_row / P
        if s < k
            M[(s + 1):end, (s + 1):end] =
            @view(M[(s + 1):end, (s + 1):end]) - piv_col * transpose(piv_row) / P
        end
    end
    inf_error = maximum(
        abs, @view(M[(rank + 1):end, (rank + 1):end]); init = zero(real(Elt))
    )
    L = @view(L[:, 1:rank])
    d = @view(d[1:rank])
    U = @view(U[1:rank, :])
    return L, d, U, invperm(rps), invperm(cps), inf_error
end