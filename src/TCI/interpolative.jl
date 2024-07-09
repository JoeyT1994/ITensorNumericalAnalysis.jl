using ITensors: ITensor, Index, dag, combinedind, combiner
using ITensors.NDTensors: matrix
using LinearAlgebra: LinearAlgebra

"""
    interpolative(M::Matrix; cutoff, maxdim)

Returns `C, Z, piv_cols, inf_error` where
C and Z are matrices such that `C*Z ≈ M`.
The matrix C consists of columns of M, and
which column is given by the integer entries of
the array `piv_cols`. The number of columns of
C is controlled by the approximate (SVD) rank of M,
which is controlled by the parameters `cutoff`
and `maxdim`.

"""
function interpolative(M::Matrix; kws...)
  # Compute interpolative decomposition (ID) from PRRLU
  L, d, U, pr, pc, inf_error = prrldu(M; kws...)
  U11 = U[:, 1:length(d)]
  iU11 = backsolveU(U11)
  ZjJ = iU11 * U
  CIj = L * LinearAlgebra.Diagonal(d) * U11
  C = CIj[pr, :]
  Z = ZjJ[:, pc]
  # Compute mapping of pivot columns to column indices
  piv_cols = invperm(pc)[1:length(d)]
  return C, Z, piv_cols, inf_error
end

"""
    interpolative(M::ITensor, col_inds; kws...)

Compute the interpolative decomposition of an ITensor, treated as a 
matrix with column indices given by the collection `col_inds`. 

Return a tuple of the following:
* C - ITensor containing specific columns of `M` and having 
*     indices `col_inds` plus an index connecting to Z
* Z - ITensor such that `C*Z ≈ M`
* pivs - array of arrays specifying the settings of the column indices of `M`
  corresponding to the columns contained in C
* inf_error - maximum elementwise (infinity norm) error between `C*Z` and `M`

Internally uses the pivoted, rank-revealing LDU matrix decomposition.

Optional keyword arguments:
* maxdim::Int - maximum number of columns to keep in factorization
* mindim::Int - minimum number of columns to keep in factorization
* cutoff::Float64 - keep only as many columns such that the value of the infinity (max) norm difference from the original tensor is below this value
* tags="Link" - tags to use for the Index connecting `C` to `Z`
"""
function interpolative(
  T::ITensor, col_inds; col_vertex, cutoff=0.0, maxdim=typemax(Int), mindim=0, tags="Link"
)
  for i in col_inds
    (haspivots(i) || hastags(i, "Digit") ||
      error("interpolative requires all indices to have pivots or else \"Digit\" tag"))
  end
  # Matricize T
  row_inds = setdiff(inds(T), col_inds)
  Cmb_row, Cmb_col = combiner(row_inds), combiner(col_inds)
  cr, cc = combinedind(Cmb_row), combinedind(Cmb_col)
  t = matrix(Cmb_row * T * Cmb_col, cr, cc)

  # Interpolative decomp of t matrix
  c, z, piv_cols, inf_error = interpolative(t; cutoff, maxdim, mindim)
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
  pivot_eltype = Pair{typeof(col_vertex),Int}
  pivs = Vector{Vector{pivot_eltype}}(undef, rank)
  function get_pivs(r, c)
    i = col_inds[c]
    ip = col_pivs[r][c]
    is_site[c] && (return [col_vertex => ip])
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
