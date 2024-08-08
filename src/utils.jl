using Graphs: AbstractGraph
using ITensors: ITensors, Index, dim, inds, combiner, array, tr
using ITensorNetworks:
  AbstractITensorNetwork,
  BeliefPropagationCache,
  IndsNetwork,
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
  norm
using NamedGraphs: NamedGraph, NamedEdge, NamedGraphs, rename_vertices
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
