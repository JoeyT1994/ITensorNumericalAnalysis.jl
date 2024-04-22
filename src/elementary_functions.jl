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
default_nterms() = 30
default_dimension() = 1

"""Build a representation of the function f(x,y,z,...) = c, with flexible choice of linkdim"""
function const_itensornetwork(s::IndsNetworkMap; c=default_c_value(), linkdim::Int=1)
  ψ = random_itensornetwork(s; link_space=linkdim)
  inv_L = Number(1.0 / nv(s))
  for v in vertices(ψ)
    sinds = inds(s, v)
    virt_inds = setdiff(inds(ψ[v]), sinds)
    ψ[v] = c^inv_L * c_tensor(only(sinds), virt_inds)
  end

  return ψ
end

"""Construct the product state representation of the exp(kx+a) 
function for x ∈ [0,1] as an ITensorNetworkFunction, along the specified dim"""
function exp_itensornetwork(
  s::IndsNetworkMap;
  k=default_k_value(),
  a=default_a_value(),
  c=default_c_value(),
  dimension::Int=default_dimension(),
)
  ψ = const_itensornetwork(s)
  Lx = length(dimension_vertices(ψ, dimension))
  c_inv = c^Number(1.0 / Lx)
  for v in dimension_vertices(ψ, dimension)
    sind = only(inds(s, v))
    ψ[v] = ITensor(
      exp(a / Lx) * c_inv * exp.(k * index_values_to_scalars(s, sind)), inds(ψ[v])
    )
  end

  return ψ
end

"""Construct the bond dim 2 representation of the cosh(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function cosh_itensornetwork(
  s::IndsNetworkMap;
  k=default_k_value(),
  a=default_a_value(),
  c=default_c_value(),
  dimension::Int=default_dimension(),
)
  ψ1 = exp_itensornetwork(s; a, k, c, dimension)
  ψ2 = exp_itensornetwork(s; a=-a, k=-k, c, dimension)

  ψ1[first(vertices(ψ1))] *= 0.5
  ψ2[first(vertices(ψ1))] *= 0.5

  return ψ1 + ψ2
end

"""Construct the bond dim 2 representation of the sinh(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function sinh_itensornetwork(
  s::IndsNetworkMap;
  k=default_k_value(),
  a=default_a_value(),
  c=default_c_value(),
  dimension::Int=default_dimension(),
)
  ψ1 = exp_itensornetwork(s; a, k, c, dimension)
  ψ2 = exp_itensornetwork(s; a=-a, k=-k, c, dimension)

  ψ1[first(vertices(ψ1))] *= 0.5
  ψ2[first(vertices(ψ1))] *= -0.5

  return ψ1 + ψ2
end

"""Construct the bond dim n representation of the tanh(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function tanh_itensornetwork(
  s::IndsNetworkMap;
  k=default_k_value(),
  a=default_a_value(),
  c=default_c_value(),
  nterms::Int=default_nterms(),
  dimension::Int=default_dimension(),
)
  ψ = const_itensornetwork(s)
  for n in 1:nterms
    ψt = exp_itensornetwork(s; a=-2 * n * a, c, k=-2 * k * n, dimension)
    ψt[first(vertices(ψt))] *= 2 * ((-1)^n)
    ψ = ψ + ψt
  end

  return ψ
end

"""Construct the bond dim 2 representation of the cos(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function cos_itensornetwork(
  s::IndsNetworkMap;
  k=default_k_value(),
  a=default_a_value(),
  c=default_c_value(),
  dimension::Int=default_dimension(),
)
  ψ1 = exp_itensornetwork(s; a=a * im, k=k * im, c=c, dimension)
  ψ2 = exp_itensornetwork(s; a=-a * im, k=-k * im, c=c, dimension)

  ψ1[first(vertices(ψ1))] *= 0.5
  ψ2[first(vertices(ψ1))] *= 0.5

  return ψ1 + ψ2
end

"""Construct the bond dim 2 representation of the sin(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function sin_itensornetwork(
  s::IndsNetworkMap;
  k=default_k_value(),
  a=default_a_value(),
  c = default_c_value(),
  dimension::Int=default_dimension(),
)
  ψ1 = exp_itensornetwork(s; a=a * im, k=k * im, c =c,  dimension)
  ψ2 = exp_itensornetwork(s; a=-a * im, k=-k * im, c = c, dimension)

  ψ1[first(vertices(ψ1))] *= -0.5 * im
  ψ2[first(vertices(ψ1))] *= 0.5 * im

  return ψ1 + ψ2
end

"""Build a representation of the function f(x) = sum_{i=0}^{n}coeffs[i+1]*(x)^{i} on the graph structure specified
by indsnetwork"""
function polynomial_itensornetwork(
  s::IndsNetworkMap, coeffs::Vector; dimension::Int=default_dimension()
)
  n = length(coeffs)
  #First treeify the index network (ignore edges that form loops)
  _s = indsnetwork(s)
  g = underlying_graph(_s)
  g_tree = undirected_graph(random_bfs_tree(g, first(vertices(g))))
  s_tree = add_edges(rem_edges(_s, edges(g)), edges(g_tree))
  s_tree = IndsNetworkMap(s_tree, indexmap(s))

  ψ = const_itensornetwork(s_tree; linkdim=n)
  dim_vertices = dimension_vertices(ψ, dimension)
  source_vertex = first(dim_vertices)

  for v in dim_vertices
    siteindex = only(inds(s_tree, v))
    if v != source_vertex
      e = get_edge_toward_vertex(g_tree, v, source_vertex)
      betaindex = only(commoninds(ψ, e))
      alphas = setdiff(inds(ψ[v]), Index[siteindex, betaindex])
      ψ[v] = Q_N_tensor(
        length(neighbors(g_tree, v)),
        siteindex,
        alphas,
        betaindex,
        index_values_to_scalars(s_tree, siteindex),
      )
    elseif v == source_vertex
      betaindex = Index(n, "DummyInd")
      alphas = setdiff(inds(ψ[v]), Index[siteindex])
      ψv = Q_N_tensor(
        length(neighbors(g_tree, v)) + 1,
        siteindex,
        alphas,
        betaindex,
        index_values_to_scalars(s_tree, siteindex),
      )
      ψ[v] = ψv * ITensor(coeffs, betaindex)
    end
  end

  #Put the transfer tensors in, these are special tensors that
  # go on the digits (sites) that don't correspond to the desired dimension
  for v in setdiff(vertices(ψ), dim_vertices)
    siteindex = only(inds(s_tree, v))
    e = get_edge_toward_vertex(g_tree, v, source_vertex)
    betaindex = only(commoninds(ψ, e))
    alphas = setdiff(inds(ψ[v]), Index[siteindex, betaindex])
    ψ[v] = transfer_tensor(siteindex, betaindex, alphas)
  end

  return ψ
end

function random_itensornetwork(s::IndsNetworkMap; kwargs...)
  return ITensorNetworkFunction(random_tensornetwork(indsnetwork(s); kwargs...), s)
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
