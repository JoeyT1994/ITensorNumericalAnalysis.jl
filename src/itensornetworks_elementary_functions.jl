using NamedGraphs: random_bfs_tree, rem_edges

include("itensornetworksutils.jl")

"""Construct the product state representation of the function f(x) = const."""
function const_itensornetwork(s::IndsNetwork; c::Union{Float64,ComplexF64}=1.0)
  ψ = delta_network(s; link_space=1)
  for v in vertices(ψ)
    ψ[v] = ITensor([1.0, 1.0], inds(ψ[v]))
  end
  ψ[first(vertices(ψ))] *= c

  ψ = is_tree(underlying_graph(s)) ? TTN(ψ) : ψ

  return ψ
end

"""Construct the representation of the function f(x) = kx"""
function x_itensornetwork(
  s::IndsNetwork, vertex_map::Dict; k::Union{Float64,ComplexF64}=1.0, cutoff=1e-16
)
  ψ = const_itensornetwork(s; c=0.0)
  g = underlying_graph(s)

  for v in vertices(s)
    ψxi = const_itensornetwork(s)
    ψxi[v] = ITensor([0.0, 1.0 / 2^vertex_map[v]], inds(ψxi[v]))
    ψ = ψ + ψxi

    if isa(ψ, TreeTensorNetwork)
      ψ = truncate(ψ; cutoff)
    end
  end
  ψ[first(vertices(ψ))] *= k

  return ψ
end

"""Construct the representation of the function f(x) = kx*x"""
function xsq_itensornetwork(
  s::IndsNetwork, vertex_map::Dict; k::Union{Float64,ComplexF64}=1.0, cutoff=1e-16
)
  ψ = const_itensornetwork(s; c=0.0)

  for v in vertices(s)
    for vp in vertices(s)
      ψxi = const_itensornetwork(s)
      if v != vp
        ψxi[v] = ITensor([0.0, 1.0 / 2^vertex_map[v]], inds(ψxi[v]))
        ψxi[vp] = ITensor([0.0, 1.0 / 2^vertex_map[vp]], inds(ψxi[vp]))
      else
        ψxi[v] = ITensor([0.0, 1.0 / (2^(2 * vertex_map[v]))], inds(ψxi[v]))
      end

      ψ = ψ + ψxi

      if isa(ψ, TreeTensorNetwork)
        ψ = truncate(ψ; cutoff)
      end
    end
  end
  ψ[first(vertices(ψ))] *= k

  return ψ
end

"""Construct the product state representation of the exp(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function exp_itensornetwork(
  s::IndsNetwork,
  vertex_map::Dict;
  k::Union{Float64,ComplexF64}=Float64(1.0),
  a::Union{Float64,ComplexF64}=Float64(1.0),
)
  ψ = delta_network(s; link_space=1)
  for v in vertices(ψ)
    ψ[v] = ITensor([1.0, exp(k / (2^vertex_map[v]))], inds(ψ[v]))
  end

  ψ = is_tree(underlying_graph(s)) ? TTN(ψ) : ψ
  ψ[first(vertices(ψ))] *= exp(a)

  return ψ
end

"""Construct the bond dim 2 representation of the cosh(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function cosh_itensornetwork(
  s::IndsNetwork,
  vertex_map::Dict;
  k::Union{Float64,ComplexF64}=Float64(1.0),
  a::Float64=1.0,
)
  ψ1 = exp_itensornetwork(s, vertex_map; a, k)
  ψ2 = exp_itensornetwork(s, vertex_map; a=-a, k=-k)

  ψ1[first(vertices(ψ1))] *= 0.5
  ψ2[first(vertices(ψ1))] *= 0.5

  return ψ1 + ψ2
end

"""Construct the bond dim 2 representation of the sinh(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function sinh_itensornetwork(
  s::IndsNetwork,
  vertex_map::Dict;
  k::Union{Float64,ComplexF64}=Float64(1.0),
  a::Float64=1.0,
)
  ψ1 = exp_itensornetwork(s, vertex_map; a, k)
  ψ2 = exp_itensornetwork(s, vertex_map; a=-a, k=-k)

  ψ1[first(vertices(ψ1))] *= 0.5
  ψ2[first(vertices(ψ1))] *= -0.5

  return ψ1 + ψ2
end

"""Construct the bond dim n representation of the tanh(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function tanh_itensornetwork(
  s::IndsNetwork,
  vertex_map::Dict,
  nterms::Int64;
  k::Union{Float64,ComplexF64}=Float64(1.0),
  a::Float64=1.0,
)
  ψ = const_itensornetwork(s)
  for n in 1:nterms
    ψt = exp_itensornetwork(s, vertex_map; a=-2 * n * a, k=-2 * k * n)
    ψt[first(vertices(ψt))] *= 2 * ((-1)^n)
    ψ = ψ + ψt
  end

  return ψ
end

"""Construct the bond dim 2 representation of the cos(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function cos_itensornetwork(
  s::IndsNetwork, vertex_map::Dict; k::Float64=1.0, a::Float64=1.0
)
  ψ1 = exp_itensornetwork(s, vertex_map; a=a * im, k=k * im)
  ψ2 = exp_itensornetwork(s, vertex_map; a=-a * im, k=-k * im)

  ψ1[first(vertices(ψ1))] *= 0.5
  ψ2[first(vertices(ψ1))] *= 0.5

  return ψ1 + ψ2
end

"""Construct the bond dim 2 representation of the sin(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function sin_itensornetwork(
  s::IndsNetwork, vertex_map::Dict; k::Float64=1.0, a::Float64=1.0
)
  ψ1 = exp_itensornetwork(s, vertex_map; a=a * im, k=k * im)
  ψ2 = exp_itensornetwork(s, vertex_map; a=-a * im, k=-k * im)

  ψ1[first(vertices(ψ1))] *= -0.5 * im
  ψ2[first(vertices(ψ1))] *= 0.5 * im

  return ψ1 + ψ2
end

function f_alpha_beta(α::Vector{Int64}, beta::Int64)
  if !isempty(α)
    return max(0, beta - 1 - sum(α) + length(α))
  else
    return max(0, beta - 1)
  end
end

function _coeff(N::Int64, α::Vector{Int64}, beta)
  @assert length(α) == N - 1
  if f_alpha_beta(α, beta) >= 0
    if N == 1
      return 1
    else
      coeffs = [binomial(f_alpha_beta(α[1:N-1-i], beta), α[N-i] - 1) for i in 1:N-1]
      return prod(coeffs)
    end
  else
    return 0
  end
end

function Q_N_tensor(N::Int64, siteind::Index, αind::Vector{Index}, betaind::Index, xivals::Vector{Float64})
  @assert length(αind) == N - 1
  @assert length(xivals) == dim(siteind)
  n = dim(betaind) - 1
  @assert all(x -> x== n +1, dim.(αind))

  link_dims = [n+1 for i in 1:N] 
  dims = vcat([dim(siteind)], link_dims)
  Q_N_array = zeros(Tuple(dims))
  for (i, xi) in enumerate(xivals)
    for j = 0:(n+1)^(N) - 1
      is = digits(j, base = n+1, pad = N) + ones(Int64, (N))
      f = f_alpha_beta(is[1:(N-1)], last(is))
      Q_N_array[(i, Tuple(is)...)...] = _coeff(N, is[1:(N-1)], last(is))*(xi^f)
    end
  end
    
  Q_N = ITensor(Q_N_array, siteind, αind, betaind)
end

function get_edge_toward_root(g::AbstractGraph, v, root_vertex)

  @assert is_tree(g)
  @assert v != root_vertex

  for vn in neighbors(g, v)
      if length(a_star(g, vn, root_vertex)) < length(a_star(g, v, root_vertex))
          return NamedEdge(v=>vn)
      end
  end
end

function polynomial_itensornetwork(s::IndsNetwork, vertex_map::Dict, coeffs::Vector{Float64})
  n = length(coeffs) - 1
  g = underlying_graph(s)
  g_d_tree = random_bfs_tree(g, first(vertices(g)))
  g_tree = NamedGraph(vertices(g_d_tree))
  g_tree = add_edges(g_tree, edges(g_d_tree))
  s_tree = add_edges(rem_edges(s, edges(g)), edges(g_tree))
  root_vertex = first(leaf_vertices(s_tree))
  ψ = delta_network(s_tree; link_space = n + 1)
  for v in vertices(ψ)
    siteindex = s_tree[v][]
    if v!= root_vertex
      e = get_edge_toward_root(g_tree, v, root_vertex)
      betaindex = first(commoninds(ψ, e))
      alphas = setdiff(inds(ψ[v]), Index[siteindex, betaindex])
      N = length(neighbors(g_tree, v))
      ψ[v] = Q_N_tensor(N, siteindex, alphas, betaindex, [0.0, (1.0/(2^vertex_map[v]))])
    else
      betaindex = Index(n+1, "DummyInd")
      alphas = setdiff(inds(ψ[v]), Index[siteindex])
      ψv = Q_N_tensor(2, siteindex, alphas, betaindex, [0.0, (1.0/(2^vertex_map[v]))])
      C_tensor = ITensor(reverse(coeffs), betaindex)
      ψ[v] = ψv * C_tensor
    end
  end

  return ψ
end

const const_itn = const_itensornetwork
const poly_itn = polynomial_itensornetwork
const cosh_itn = cosh_itensornetwork
const sinh_itn = sinh_itensornetwork
const tanh_itn = tanh_itensornetwork
const exp_itn = exp_itensornetwork
const sin_itn = sin_itensornetwork
const cos_itn = cos_itensornetwork
const x_itn = x_itensornetwork