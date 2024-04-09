using Graphs: nv, vertices, edges, neighbors
using NamedGraphs:
  random_bfs_tree,
  rem_edges,
  add_edges,
  undirected_graph,
  NamedEdge,
  AbstractGraph,
  leaf_vertices,
  a_star
using ITensors: dim, commoninds
using ITensorNetworks: IndsNetwork, underlying_graph

default_c_value() = 1.0
default_a_value() = 0.0
default_k_value() = 1.0
default_nterms() = 20
default_dimension() = 1

"""Build a representation of the function f(x,y,z,...) = c, with flexible choice of linkdim"""
function const_itensornetwork(
  s::IndsNetwork, bit_map; c::Union{Float64,ComplexF64}=default_c_value(), linkdim::Int64=1
)
  ψ = random_tensornetwork(s; link_space=linkdim)
  inv_L = Float64(1.0 / nv(s))
  for v in vertices(ψ)
    virt_inds = setdiff(inds(ψ[v]), Index[only(s[v])])
    ψ[v] = c^inv_L * c_tensor(only(s[v]), virt_inds)
  end

  return ITensorNetworkFunction(ψ, bit_map)
end

"""Construct the product state representation of the exp(kx+a) 
function for x ∈ [0,1] as an ITensorNetworkFunction, along the specified dim"""
function exp_itensornetwork(
  s::IndsNetwork,
  bit_map;
  k::Union{Float64,ComplexF64}=default_k_value(),
  a::Union{Float64,ComplexF64}=default_a_value(),
  dimension::Int64=default_dimension(),
)
  ψ = const_itensornetwork(s, bit_map)
  Lx = length(vertices(bit_map, dimension))
  for v in vertices(bit_map, dimension)
    ψ[v] = ITensor(
      [
        exp(a / Lx) * exp(k * bit_value_to_scalar(bit_map, v, i)) for i in 0:(dim(s[v]) - 1)
      ],
      inds(ψ[v]),
    )
  end

  return ψ
end

"""Construct the bond dim 2 representation of the cosh(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function cosh_itensornetwork(
  s::IndsNetwork,
  bit_map;
  k::Union{Float64,ComplexF64}=default_k_value(),
  a::Union{Float64,ComplexF64}=default_a_value(),
  dimension::Int64=default_dimension(),
)
  ψ1 = exp_itensornetwork(s, bit_map; a, k, dimension)
  ψ2 = exp_itensornetwork(s, bit_map; a=-a, k=-k, dimension)

  ψ1[first(vertices(ψ1))] *= 0.5
  ψ2[first(vertices(ψ1))] *= 0.5

  return ψ1 + ψ2
end

"""Construct the bond dim 2 representation of the sinh(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function sinh_itensornetwork(
  s::IndsNetwork,
  bit_map;
  k::Union{Float64,ComplexF64}=default_k_value(),
  a::Union{Float64,ComplexF64}=default_a_value(),
  dimension::Int64=default_dimension(),
)
  ψ1 = exp_itensornetwork(s, bit_map; a, k, dimension)
  ψ2 = exp_itensornetwork(s, bit_map; a=-a, k=-k, dimension)

  ψ1[first(vertices(ψ1))] *= 0.5
  ψ2[first(vertices(ψ1))] *= -0.5

  return ψ1 + ψ2
end

"""Construct the bond dim n representation of the tanh(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function tanh_itensornetwork(
  s::IndsNetwork,
  bit_map;
  k::Union{Float64,ComplexF64}=default_k_value(),
  a::Union{Float64,ComplexF64}=default_a_value(),
  nterms::Int64=default_nterms(),
  dimension::Int64=default_dimension(),
)
  ψ = const_itensornetwork(s, bit_map)
  for n in 1:nterms
    ψt = exp_itensornetwork(s, bit_map; a=-2 * n * a, k=-2 * k * n, dimension)
    ψt[first(vertices(ψt))] *= 2 * ((-1)^n)
    ψ = ψ + ψt
  end

  return ψ
end

"""Construct the bond dim 2 representation of the cos(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function cos_itensornetwork(
  s::IndsNetwork,
  bit_map;
  k::Union{Float64,ComplexF64}=default_k_value(),
  a::Union{Float64,ComplexF64}=default_a_value(),
  dimension::Int64=default_dimension(),
)
  ψ1 = exp_itensornetwork(s, bit_map; a=a * im, k=k * im, dimension)
  ψ2 = exp_itensornetwork(s, bit_map; a=-a * im, k=-k * im, dimension)

  ψ1[first(vertices(ψ1))] *= 0.5
  ψ2[first(vertices(ψ1))] *= 0.5

  return ψ1 + ψ2
end

"""Construct the bond dim 2 representation of the sin(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function sin_itensornetwork(
  s::IndsNetwork,
  bit_map;
  k::Union{Float64,ComplexF64}=default_k_value(),
  a::Union{Float64,ComplexF64}=default_a_value(),
  dimension::Int64=default_dimension(),
)
  ψ1 = exp_itensornetwork(s, bit_map; a=a * im, k=k * im, dimension)
  ψ2 = exp_itensornetwork(s, bit_map; a=-a * im, k=-k * im, dimension)

  ψ1[first(vertices(ψ1))] *= -0.5 * im
  ψ2[first(vertices(ψ1))] *= 0.5 * im

  return ψ1 + ψ2
end

# #FUNCTIONS NEEDED TO IMPLEMENT POLYNOMIALS
# """Exponent on x_i for the tensor Q(x_i) on the tree"""
function f_alpha_beta(α::Vector{Int64}, beta::Int64)
  return !isempty(α) ? max(0, beta - 1 - sum(α) + length(α)) : max(0, beta - 1)
end

# """Coefficient on x_i for the tensor Q(x_i) on the tree"""
function _coeff(N::Int64, α::Vector{Int64}, beta)
  @assert length(α) == N - 1
  return if N == 1
    1
  else
    prod([binomial(f_alpha_beta(α[1:(N - 1 - i)], beta), α[N - i] - 1) for i in 1:(N - 1)])
  end
end

"""Constructor for the tensor that sits on a vertex of degree N"""
function Q_N_tensor(
  N::Int64, siteind::Index, αind::Vector{Index}, betaind::Index, xivals::Vector{Float64}
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
      is = digits(j; base=n + 1, pad=N) + ones(Int64, (N))
      f = f_alpha_beta(is[1:(N - 1)], last(is))
      Q_N_array[(i, Tuple(is)...)...] = _coeff(N, is[1:(N - 1)], last(is)) * (xi^f)
    end
  end

  return ITensor(Q_N_array, siteind, αind, betaind)
end

# """Given a tree find the edge coming from the vertex v which is directed towards `root_vertex`"""
function get_edge_toward_root(g::AbstractGraph, v, root_vertex)
  @assert is_tree(g)
  @assert v != root_vertex

  for vn in neighbors(g, v)
    if length(a_star(g, vn, root_vertex)) < length(a_star(g, v, root_vertex))
      return NamedEdge(v => vn)
    end
  end
end

"""Build a representation of the function f(x) = sum_{i=0}^{n}coeffs[i+1]*(x)^{i} on the graph structure specified
by indsnetwork"""
function polynomial_itensornetwork(
  s::IndsNetwork, bit_map, coeffs::Vector{Float64}; dimension::Int64=default_dimension()
)
  n = length(coeffs) - 1

  #First treeify the index network (ignore edges that form loops)
  g = underlying_graph(s)
  g_tree = undirected_graph(random_bfs_tree(g, first(vertices(g))))
  s_tree = add_edges(rem_edges(s, edges(g)), edges(g_tree))

  dimension_vertices = vertices(bit_map, dimension)

  #Pick a root

  #Need the root vertex to be in the dimension vertices at the moment, should be a way around
  root_vertices_dim = filter(v -> v ∈ dimension_vertices, leaf_vertices(s_tree))
  @assert !isempty(root_vertices_dim)
  root_vertex = first(root_vertices_dim)
  ψ = const_itensornetwork(s_tree, bit_map; linkdim=n + 1)
  #Place the Q_n tensors, making sure we get the right index pointing towards the root
  for v in dimension_vertices
    siteindex = s_tree[v][]
    if v != root_vertex
      e = get_edge_toward_root(g_tree, v, root_vertex)
      betaindex = first(commoninds(ψ, e))
      alphas = setdiff(inds(ψ[v]), Index[siteindex, betaindex])
      ψ[v] = Q_N_tensor(
        length(neighbors(g_tree, v)),
        siteindex,
        alphas,
        betaindex,
        [bit_value_to_scalar(bit_map, v, i) for i in 0:(dim(siteindex) - 1)],
      )
    elseif v == root_vertex
      betaindex = Index(n + 1, "DummyInd")
      alphas = setdiff(inds(ψ[v]), Index[siteindex])
      ψv = Q_N_tensor(
        2,
        siteindex,
        alphas,
        betaindex,
        [bit_value_to_scalar(bit_map, v, i) for i in 0:(dim(siteindex) - 1)],
      )
      ψ[v] = ψv * ITensor(reverse(coeffs), betaindex)
    end
  end

  return ψ
end

function random_itensornetwork(s::IndsNetwork, bit_map; kwargs...)
  return ITensorNetworkFunction(random_tensornetwork(s; kwargs...), bit_map)
end

const const_itn = const_itensornetwork
const poly_itn = polynomial_itensornetwork
const cosh_itn = cosh_itensornetwork
const sinh_itn = sinh_itensornetwork
const tanh_itn = tanh_itensornetwork
const exp_itn = exp_itensornetwork
const sin_itn = sin_itensornetwork
const cos_itn = cos_itensornetwork
const rand_itn = random_itensornetwork
