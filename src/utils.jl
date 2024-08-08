using Graphs: AbstractGraph
using ITensors: ITensors, ITensor, Index, dim, inds, combiner, array, tr, tags, uniqueinds
using ITensors.ITensorMPS: ITensorMPS
using ITensorNetworks:
  AbstractITensorNetwork,
  BeliefPropagationCache,
  IndsNetwork,
  ITensorNetworks,
  QuadraticFormNetwork,
  random_tensornetwork,
  environment,
  update,
  factor,
  default_message_update,
  tensornetwork,
  partitioned_tensornetwork,
  operator_vertex,
  messages,
  default_message,
  optimal_contraction_sequence,
  norm,
  is_multi_edge,
  linkinds
using NamedGraphs: NamedGraph, NamedEdge, NamedGraphs, rename_vertices, src, dst
using NamedGraphs.GraphsExtensions: rem_vertex
using NamedGraphs.PartitionedGraphs:
  PartitionEdge, partitionvertices, partitioned_graph, PartitionVertex

"""Build the order L tensor corresponding to fx(x): x ∈ [0,1], default decomposition is binary"""
function build_full_rank_tensor(L::Int, fx::Function; base::Int=2)
  inds = [Index(base, "$i") for i in 1:L]
  dims = Tuple([base for i in 1:L])
  array = zeros(dims)
  for i in 0:(base^(L) - 1)
    xis = digits(i; base, pad=L)
    x = sum([xis[i] / (base^i) for i in 1:L])
    array[Tuple(xis + ones(Int, (L)))...] = fx(x)
  end

  return ITensor(array, inds)
end

"""Build the tensor C such that C_{phys_ind, virt_inds...} = delta_{virt_inds...}"""
function c_tensor(phys_inds::Vector, virt_inds::Vector)
  @assert allequal(dim.(virt_inds))
  T = delta(Int64, virt_inds)
  T = T * ITensor(1, phys_inds...)
  return T
end

function ITensors.inds(s::IndsNetwork, v)
  return s[v]
end

function ITensors.inds(s::IndsNetwork, verts::Vector)
  return reduce(vcat, [inds(s, v) for v in verts])
end

function ITensors.inds(s::IndsNetwork)
  return inds(s, collect(vertices(s)))
end

function base(s::IndsNetwork)
  indices = inds(s)
  dims = dim.(indices)
  @assert all(d -> d == first(dims), dims)
  return first(dims)
end

function ITensorNetworks.message(bp_cache::BeliefPropagationCache, edge::PartitionEdge)
  mts = messages(bp_cache)
  haskey(mts, edge) && return mts[edge]
  return default_message(bp_cache, edge)
end

function ITensorNetworks.default_message_update(
  contract_list::Vector{ITensor}; normalize=true, kwargs...
)
  sequence = optimal_contraction_sequence(contract_list)
  updated_messages = contract(contract_list; sequence, kwargs...)
  if normalize
    updated_messages /= norm(updated_messages)
  end
  return ITensor[updated_messages]
end

"""Compute the two-site rdm from a tree-tensor network, sclaes as O(Lchi^{z+1})"""
function two_site_rdm(
  ψ::AbstractITensorNetwork, v1, v2; (cache!)=nothing, cache_update_kwargs=(;)
)
  ψIψ_bpc = if isnothing(cache!)
    update(BeliefPropagationCache(QuadraticFormNetwork(ψ)); cache_update_kwargs...)
  else
    cache![]
  end
  ψIψ = tensornetwork(ψIψ_bpc)
  pg = partitioned_tensornetwork(ψIψ_bpc)

  path = PartitionEdge.(a_star(partitioned_graph(ψIψ_bpc), v1, v2))
  pg = rem_vertex(pg, operator_vertex(ψIψ, v1))
  pg = rem_vertex(pg, operator_vertex(ψIψ, v2))
  ψIψ_bpc_mod = BeliefPropagationCache(pg, messages(ψIψ_bpc), default_message)
  ψIψ_bpc_mod = update(
    ψIψ_bpc_mod, path; message_update=ms -> default_message_update(ms; normalize=false)
  )
  incoming_mts = environment(ψIψ_bpc_mod, [PartitionVertex(v2)])
  local_state = factor(ψIψ_bpc_mod, PartitionVertex(v2))
  rdm = contract(vcat(incoming_mts, local_state); sequence="automatic")

  rdm = array((rdm * combiner(inds(rdm; plev=0)...)) * combiner(inds(rdm; plev=1)...))
  rdm /= tr(rdm)
  return rdm
end

#Given an itensornetwork, contract away any tensors which don't have external indices.
function merge_internal_tensors(tn::AbstractITensorNetwork)
  tn = copy(tn)
  internal_vertices = filter(v -> isempty(uniqueinds(tn, v)), collect(vertices(tn)))
  external_vertices = filter(v -> !isempty(uniqueinds(tn, v)), collect(vertices(tn)))
  for v in internal_vertices
    vns = neighbors(tn, v)
    if !isempty(vns)
      tn = contract(tn, v => first(vns))
    else
      tn[first(external_vertices)] *= tn[v][]
      tn = rem_vertex(tn, v)
    end
  end
  return tn
end

#Overloading this here for now due to a bug when adding networks with the same edges but just 
#reversed

function edges_equal(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork)
  for e in edges(tn1)
    if e ∉ edges(tn2) && reverse(e) ∉ edges(tn2)
      return false
    end
  end
  return true
end

function ITensorMPS.add(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork)
  @assert issetequal(vertices(tn1), vertices(tn2))

  tn1 = combine_linkinds(tn1; edges=filter(is_multi_edge(tn1), edges(tn1)))
  tn2 = combine_linkinds(tn2; edges=filter(is_multi_edge(tn2), edges(tn2)))

  edges_tn1, edges_tn2 = edges(tn1), edges(tn2)

  if edges_equal(tn1, tn2)
    new_edges = union(edges_tn1, edges_tn2)
    tn1 = insert_linkinds(tn1, new_edges)
    tn2 = insert_linkinds(tn2, new_edges)
  end

  edges_tn1, edges_tn2 = edges(tn1), edges(tn2)

  tn12 = copy(tn1)
  new_edge_indices = Dict(
    zip(
      edges_tn1,
      [
        Index(
          dim(only(linkinds(tn1, e))) + dim(only(linkinds(tn2, e))),
          tags(only(linkinds(tn1, e))),
        ) for e in edges_tn1
      ],
    ),
  )

  #Create vertices of tn12 as direct sum of tn1[v] and tn2[v]. Work out the matching indices by matching edges. Make index tags those of tn1[v]
  for v in vertices(tn1)
    @assert issetequal(siteinds(tn1, v), siteinds(tn2, v))

    e1_v = filter(x -> src(x) == v || dst(x) == v, edges_tn1)
    e2_v = filter(x -> src(x) == v || dst(x) == v, edges_tn2)

    #@assert issetequal(e1_v, e2_v)
    tn1v_linkinds = Index[only(linkinds(tn1, e)) for e in e1_v]
    tn2v_linkinds = Index[only(linkinds(tn2, e)) for e in e1_v]
    tn12v_linkinds = Index[new_edge_indices[e] for e in e1_v]

    @assert length(tn1v_linkinds) == length(tn2v_linkinds)

    tn12[v] = ITensors.directsum(
      tn12v_linkinds,
      tn1[v] => Tuple(tn1v_linkinds),
      tn2[v] => Tuple(tn2v_linkinds);
      tags=tags.(Tuple(tn1v_linkinds)),
    )
  end

  return tn12
end
