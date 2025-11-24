using Graphs: nv, vertices, edges, neighbors
using NamedGraphs: NamedEdge, AbstractGraph, a_star
using NamedGraphs.GraphsExtensions:
  random_bfs_tree, rem_edges, add_edges, leaf_vertices, undirected_graph
using TensorNetworkQuantumSimulator: setindex_preserve!, virtualinds, insert_virtualinds!
using ITensors: dim, commoninds, delta

default_c_value() = 1
default_a_value() = 0
default_k_value() = 1
default_nterms() = 20
default_dim() = 1

function random_tensornetworkfunction(eltype::Type, g::NamedGraph, I::AbstractIndexMap; kwargs...)
  return TensorNetworkFunction(
    tensornetwork(random_tensornetworkstate(eltype, g, siteinds(I); kwargs...)), I
  )
end

function random_tensornetworkfunction(g::NamedGraph, I::AbstractIndexMap; kwargs...)
  eltype = I isa RealIndexMap ? Float64 : ComplexF64
  return random_tensornetworkfunction(eltype, g, I; kwargs...)
end

"""Build a representation of the function f(x,y,z,...) = c, with flexible choice of bond dimension"""
function const_tensornetworkfunction(g::NamedGraph, I::AbstractIndexMap; c=default_c_value(), bond_dimension::Int=1)
  ψ = random_tensornetworkfunction(g, I; bond_dimension)
  c = c < 0 ? (Complex(c) / bond_dimension)^Number(1.0 / nv(g)) : (c / bond_dimension)^Number(1.0 / nv(g))
  for v in vertices(ψ)
    sinds = siteinds(I, v)
    virt_inds = setdiff(inds(ψ[v]), sinds)
    setindex_preserve!(ψ, c * c_tensor(sinds, virt_inds), v)
  end
  return ψ
end

"""Construct the product state representation of the exp(kx+a) 
function for x ∈ [0,1] as an ITensorNetworkFunction, along the specified dim"""
function exp_tensornetworkfunction(
  g::NamedGraph, 
  I::AbstractIndexMap;
  k=default_k_value(),
  a=default_a_value(),
  c=default_c_value(),
  dim::Int=default_dim(),
)
  ψ = const_tensornetworkfunction(g, I)
  Lx = length(dimension_vertices(ψ, dim))
  for v in dimension_vertices(ψ, dim)
    sinds = siteinds(I, v)
    sinds_dim = filter(i -> dimension(I, i) == dim, sinds)
    sinds_not_dim = filter(i -> dimension(I, i) != dim, sinds)
    linds = setdiff(inds(ψ[v]), sinds)
    t = prod([
      ITensor(exp.(k * index_values_to_scalars(I, sind)), sind) for sind in sinds_dim
    ])
    t *= exp(a / Lx) * delta(linds) * ITensor(1, sinds_not_dim)
    setindex_preserve!(ψ, t, v)
  end

  v0 = first(dimension_vertices(ψ, dim))
  setindex_preserve!(ψ, ψ[v0]*c, v0)

  return ψ
end

"""Construct the bond dim 2 representation of the cosh(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function cosh_tensornetworkfunction(
  g::NamedGraph,
  s::AbstractIndexMap;
  k=default_k_value(),
  a=default_a_value(),
  c=default_c_value(),
  dim::Int=default_dim(),
)
  ψ1 = exp_tensornetworkfunction(g, s; a, k, c=0.5 * c, dim)
  ψ2 = exp_tensornetworkfunction(g, s; a=(-a), k=(-k), c=0.5 * c, dim)

  return ψ1 + ψ2
end

"""Construct the bond dim 2 representation of the sinh(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function sinh_tensornetworkfunction(
  g::NamedGraph,
  s::AbstractIndexMap;
  k=default_k_value(),
  a=default_a_value(),
  c=default_c_value(),
  dim::Int=default_dim(),
)
  ψ1 = exp_tensornetworkfunction(g, s; a, k, c=0.5 * c, dim)
  ψ2 = exp_tensornetworkfunction(g, s; a=(-a), k=(-k), c=-0.5 * c, dim)

  return ψ1 + ψ2
end

"""Construct the bond dim n representation of the tanh(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function tanh_tensornetworkfunction(
  g::NamedGraph,
  s::AbstractIndexMap;
  k=default_k_value(),
  a=default_a_value(),
  c=default_c_value(),
  nterms::Int=default_nterms(),
  dim::Int=default_dim(),
)
  ψ = const_tensornetworkfunction(g, s)
  for n in 1:nterms
    ψt = exp_tensornetworkfunction(g, s; a=-2 * n * a, k=-2 * k * n, dim)
    setindex_preserve!(ψt, ψt[first(dimension_vertices(ψ, dim))] * 2 * (-1)^n, first(dimension_vertices(ψ, dim)))
    ψ = ψ + ψt
  end

  setindex_preserve!(ψ, ψ[first(dimension_vertices(ψ, dim))]*c, first(dimension_vertices(ψ, dim)))

  return ψ
end

"""Construct the bond dim 2 representation of the cos(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function cos_tensornetworkfunction(
  g::NamedGraph,
  s::AbstractIndexMap;
  k=default_k_value(),
  a=default_a_value(),
  c=default_c_value(),
  dim::Int=default_dim(),
)
  ψ1 = exp_tensornetworkfunction(g, s; a=a * im, k=k * im, c=0.5 * c, dim)
  ψ2 = exp_tensornetworkfunction(g, s; a=-a * im, k=-k * im, c=0.5 * c, dim)

  return ψ1 + ψ2
end

"""Construct the bond dim 2 representation of the sin(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function sin_tensornetworkfunction(
  g::NamedGraph,
  s::AbstractIndexMap;
  k=default_k_value(),
  a=default_a_value(),
  c=default_c_value(),
  dim::Int=default_dim(),
)
  ψ1 = exp_tensornetworkfunction(g, s; a=a * im, k=k * im, c=-0.5 * im * c, dim)
  ψ2 = exp_tensornetworkfunction(g, s; a=-a * im, k=-k * im, c=0.5 * im * c, dim)

  return ψ1 + ψ2
end

"""Build a representation of the function f(x) = sum_{i=0}^{n}coeffs[i+1]*(x)^{i}"""
function polynomial_tensornetworkfunction(
  g::NamedGraph,
  s::AbstractIndexMap,
  coeffs::Vector;
  dim::Int=default_dim(),
  k=default_k_value(),
  c=default_c_value(),
)
  n = length(coeffs)
  n == 1 && return const_itn(s; c=first(coeffs))

  coeffs = [c * (k^(i - 1)) for (i, c) in enumerate(coeffs)]
  #First treeify the index network (ignore edges that form loops)
  g_tree = undirected_graph(random_bfs_tree(g, first(vertices(g))))
  eltype = s isa RealIndexMap ? Float64 : ComplexF64

  ψ = const_tensornetworkfunction(g_tree, s; bond_dimension=n)
  dim_vertices = dimension_vertices(ψ, dim)
  source_vertex = first(dim_vertices)

  for v in dim_vertices
    sinds = siteinds(s, v)
    sinds_dim = filter(i -> dimension(s, i) == dim, sinds)
    sinds_not_dim = filter(i -> dimension(s, i) != dim, sinds)
    if v != source_vertex
      e = get_edge_toward_vertex(g_tree, v, source_vertex)
      betaindex = only(virtualinds(ψ, e))
      alphas = setdiff(inds(ψ[v]), [sinds_dim; sinds_not_dim; betaindex])
      ψv = Q_N_tensor(
        eltype,
        length(neighbors(g_tree, v)),
        sinds_dim,
        alphas,
        betaindex,
        index_values_to_scalars.((s,), sinds_dim),
      )
      ψv *= ITensor(1, sinds_not_dim)
      setindex_preserve!(ψ, ψv, v)
    elseif v == source_vertex
      betaindex = Index(n, "DummyInd")
      alphas = setdiff(inds(ψ[v]), sinds)
      ψv = Q_N_tensor(
        eltype,
        length(neighbors(g_tree, v)) + 1,
        sinds_dim,
        alphas,
        betaindex,
        index_values_to_scalars.((s,), sinds_dim),
      )
      ψv = ψv * ITensor(coeffs, betaindex) * ITensor(1, sinds_not_dim)
      setindex_preserve!(ψ, ψv, v)
    end
  end

  setindex_preserve!(ψ, ψ[first(dim_vertices)]*c, first(dim_vertices))

  #Put the transfer tensors in, these are special tensors that
  # go on the digits (sites) that don't correspond to the desired dimension
  for v in setdiff(vertices(ψ), dim_vertices)
    sinds = siteinds(s, v)
    e = get_edge_toward_vertex(g_tree, v, source_vertex)
    betaindex = only(virtualinds(ψ, e))
    alphas = setdiff(inds(ψ[v]), [sinds; betaindex])
    setindex_preserve!(ψ, transfer_tensor(sinds, betaindex, alphas), v)
  end

  return ψ
end

"Create a product state of a given bit configuration. Will make planes if all dims not specificed"
function delta_p(
  g::NamedGraph,
  s::AbstractIndexMap,
  xs::Vector{<:Number},
  dims::Vector{Int}=[i for i in 1:length(xs)];
  c = default_c_value(),
  kwargs...,
)
  ivmap = calculate_ind_values(s, xs, dims)
  vs = collect(vertices(g))
  ts = Dictionary(vs, ITensor[
    prod([
      sind ∈ keys(ivmap) ? onehot(sind => ivmap[sind] + 1) : ITensor(1, sind) for
      sind in siteinds(s, v)
    ]) for v in vs
  ])
  set!(ts, first(vs), ts[first(vs)]*c)

  tn = TensorNetwork(ts, g)
  insert_virtualinds!(tn)
  return TensorNetworkFunction(tn, s)
end

"Create a product state of a given bit configuration of a 1D function"
function delta_p(g::NamedGraph, s::AbstractIndexMap, x::Number, kwargs...)
  @assert dimension(s) == 1
  return delta_p(g, s, [x], [1]; kwargs...)
end

function delta_p(
  g::NamedGraph,
  s::AbstractIndexMap,
  points::Vector{<:Vector},
  points_dims::Vector{<:Vector}=[[i for i in 1:length(xs)] for xs in points];
  kwargs...,
)
  @assert length(points) != 0
  @assert length(points) == length(points_dims)
  ψ = reduce(
    +, [delta_p(g, s, xs, dims; kwargs...) for (xs, dims) in zip(points, points_dims)]
  )
  return ψ
end

" Function to manipulate delta functions. Defaults to map_to_zero behavior"
function delta_kernel(
  g::NamedGraph,
  s::AbstractIndexMap,
  points::Vector{<:Vector},
  points_dims::Vector{<:Vector}=[[i for i in 1:length(xs)] for xs in points];
  remove_overlap=true,
  coeff::Number=-1,
  include_identity=true,
  truncate_kwargs...,
)
  ψ = delta_p(g, s, points, points_dims; c = coeff, truncate_kwargs...)

  if include_identity
    ψ = const_tnf(g, s) + ψ
  end

  if remove_overlap && length(points) > 1
    overlap_points, overlap_dims = Vector{Vector}(), Vector{Vector}()
    # determine intersection of any points, 
    # and remove them with the opposite sign
    for i in 1:length(points)
      p1, d1 = points[i], points_dims[i]
      for j in (i + 1):length(points)
        p2, d2 = points[j], points_dims[j]

        # same dimensions, and no point overlap,
        # can safely ignore
        (all(d1 .≈ d2) && !all(p1 .≈ p2)) && continue

        # check if dims are the same. 
        # If they are, check the corresponding dim
        ps_ = [p1; p2]
        ds_ = [d1; d2]
        order = sortperm(ds_)
        ps, ds = [ps_[order[1]]], [ds_[order[1]]]
        for k in 2:length(ds_)
          if (ds_[order[k]] != ds_[order[k - 1]])
            push!(ps, ps_[order[k]])
            push!(ds, ds_[order[k]])
            continue
          end
          # found two matching elements
          if ps_[k] ≈ ps_[k - 1]
            continue # added previously
          else # there's no overap here, continue
            ps, ds = [], []
            break
          end
        end
        #(length(Set(ds)) != length(ds)) && continue
        (length(ds) == 0) && continue
        push!(overlap_points, Vector(ps))
        push!(overlap_dims, Vector(ds))
      end
    end
    if length(overlap_points) != 0
      ψ = ψ + delta_p(g, s, overlap_points, overlap_dims; c = -coeff, truncate_kwargs...)
    end
  end

  return ψ
end
const random_tnf = random_tensornetworkfunction
const const_tnf = const_tensornetworkfunction
const exp_tnf = exp_tensornetworkfunction
const cosh_tnf = cosh_tensornetworkfunction
const sinh_tnf = sinh_tensornetworkfunction
const tanh_tnf = tanh_tensornetworkfunction
const cos_tnf = cos_tensornetworkfunction
const sin_tnf = sin_tensornetworkfunction
const poly_tnf = polynomial_tensornetworkfunction
