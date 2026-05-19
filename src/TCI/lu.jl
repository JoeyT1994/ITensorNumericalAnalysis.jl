
function eye(Elt, R::Int, C::Int)
  M = zeros(Elt, R, C)
  for j in 1:min(R, C)
    M[j, j] = 1.0
  end
  return M
end

eye(R::Int, C::Int) = eye(Float64, R, C)

"""
    backsolveL(L::AbstractMatrix)

Compute the inverse of a lower-triangular matrix L whose
diagonal entries are all equal to 1.0 using a stable
back-solving algorithm.
"""
function backsolveL(L::AbstractMatrix)
  N = size(L, 1)
  (size(L, 2) == N) || error("backsolveL and backsolveU only supported for square matrices")
  iL = eye(eltype(L), N, N)
  for j in 1:(N - 1), i in (j + 1):N
    iL[i, j] = -L[i, j]
    for m in (j + 1):(i - 1)
      iL[i, j] -= L[i, m] * iL[m, j]
    end
  end
  return iL
end

"""
    backsolveU(U::AbstractMatrix)

Compute the inverse of an upper-triangular matrix U whose
diagonal entries are all equal to 1.0 using a stable
back-solving algorithm.
"""
backsolveU(U::Matrix) = transpose(backsolveL(transpose(U)))

"""
    prrldu(M::Matrix; cutoff, maxdim)

Compute the pivoted, rank-revealing LDU decomposition of an
arbitrary matrix M (M can be non-invertible and/or rectangular).
Returns matrices L,D,U and permutations pr and pc such that
L and U are lower- and upper-triangular matrices with diagonal
values equal to 1 and L[pr,:]*D*U[:,pc] ≈ M. The diagonal matrix
D will have size (k,k) with diagonal entries of decreasing
absolute value such that norm(L[pr,:]*D*U[:,pc]-M,Inf) <= abs(D[k,k]).
(Note that this inequality uses the infinity norm.)
The value of k is determined dynamically such that both `k <= maxdim`
and `abs(D[k,k]) < cutoff`.
"""
function prrldu(M_::Matrix; cutoff::Real=0.0, maxdim::Int=typemax(Int), mindim::Int=1)
  mindim = max(maxdim, 1)
  mindim = min(maxdim, mindim)
  Elt = eltype(M_)
  M = copy(M_)
  Nr, Nc = size(M)
  k = min(Nr, Nc)

  # Determine pivots
  rps = collect(1:Nr)
  cps = collect(1:Nc)

  inf_error = 0.0
  for s in 1:k
    Mabs_max, piv = findmax(abs, M)
    if Mabs_max < cutoff
      inf_error = Mabs_max
      break
    end
    Base.swaprows!(M, 1, piv[1])
    Base.swapcols!(M, 1, piv[2])
    M = M[2:end, 2:end] - M[2:end, 1] * transpose(M[1, 2:end]) / M[1, 1]
    rps[s], rps[piv[1] + s - 1] = rps[piv[1] + s - 1], rps[s]
    cps[s], cps[piv[2] + s - 1] = cps[piv[2] + s - 1], cps[s]
  end
  M = M_[rps, cps]

  L = eye(Elt, Nr, k)
  d = zeros(Elt, k)
  U = eye(Elt, k, Nc)
  rank = 0
  for s in 1:min(k, maxdim)
    P = M[s, s]
    d[s] = P

    if rank < mindim
      # then proceed
    elseif (iszero(P) || (abs(P) < cutoff && rank + 1 > mindim))
      break
    end
    iszero(P) && (P = one(Elt))
    rank += 1

    piv_col = M[(s + 1):end, s]
    L[(s + 1):end, s] = piv_col / P

    piv_row = M[s, (s + 1):end]
    U[s, (s + 1):end] = piv_row / P

    if s < k
      M[(s + 1):end, (s + 1):end] =
        M[(s + 1):end, (s + 1):end] - piv_col * transpose(piv_row) / P
    end
  end
  L = L[:, 1:rank]
  d = d[1:rank]
  U = U[1:rank, :]

  return L, d, U, invperm(rps), invperm(cps), inf_error
end
