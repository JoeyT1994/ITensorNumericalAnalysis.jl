using ITensors: Index, ITensor, dim

"""Exponent on x_i for the tensor Q(x_i) on the tree"""
function f_alpha_beta(α::Vector{Int}, beta::Int)
  return !isempty(α) ? max(0, beta - sum(α)) : max(0, beta)
end

"""Coefficient on x_i for the tensor Q(x_i) on the tree"""
function _coeff(N::Int, α::Vector{Int}, beta)
  @assert length(α) == N - 1
  return if N == 1
    1
  else
    prod([binomial(f_alpha_beta(α[1:(N - 1 - i)], beta), α[N - i]) for i in 1:(N - 1)])
  end
end

"""Constructor for the tensor that sits on a vertex of degree N"""
function Q_N_tensor(
  N::Int, siteind::Index, αind::Vector{Index}, betaind::Index, xivals::Vector
)
  @assert length(αind) == N - 1
  @assert length(xivals) == dim(siteind)
  n = dim(betaind) - 1
  @assert all(x -> x == n + 1, dim.(αind))

  link_dims = [n + 1 for i in 1:N]
  dims = vcat([dim(siteind)], link_dims)
  Q_N_array = zeros(Tuple(dims))
  for (i, xi) in enumerate(xivals)
    for j in 0:((n + 1)^(N) - 1)
      is = Base.digits(j; base=n + 1, pad=N)
      f = f_alpha_beta(is[1:(N - 1)], last(is))
      Q_N_array[(i, Tuple(is + ones(Int, (N)))...)...] =
        _coeff(N, is[1:(N - 1)], last(is)) * (xi^f)
    end
  end

  return ITensor(Q_N_array, siteind, αind, betaind)
end

"""Tensor for transferring the alpha index (beta ind here) of a QN tensor (defined above) across multiple inds (alpha_inds)"""
#Needed for building multi-dimensional polynomials
function transfer_tensor(phys_ind::Index, beta_ind::Index, alpha_inds::Vector)
  virt_inds = vcat(beta_ind, alpha_inds)
  inds = vcat(phys_ind, virt_inds)
  virt_dims = dim.(virt_inds)
  @assert allequal(virt_dims)
  dims = vcat(dim(phys_ind), virt_dims)
  N = length(alpha_inds)
  T_array = zeros(Tuple(dims))
  for i in 0:(dim(phys_ind) - 1)
    for j in 0:(dim(beta_ind) - 1)
      if !isempty(alpha_inds)
        for k in 0:((first(virt_dims))^(N) - 1)
          is = Base.digits(k; base=first(virt_dims), pad=N)
          if sum(is) == j
            T_array[(i + 1, j + 1, Tuple(is + ones(Int, (N)))...)...] = 1.0
          end
        end
      else
        T_array[i + 1, j + 1] = j == 0 ? 1 : 0
      end
    end
  end
  return ITensor(T_array, phys_ind, beta_ind, alpha_inds)
end

"""Given a tree find the edge coming from the vertex v which is directed towards `source_vertex`"""
function get_edge_toward_vertex(g::AbstractGraph, v, source_vertex)
  for vn in neighbors(g, v)
    if length(a_star(g, vn, source_vertex)) < length(a_star(g, v, source_vertex))
      return NamedEdge(v => vn)
    end
  end

  return error("Couldn't find edge. Perhaps graph is not a tree or not connected.")
end
