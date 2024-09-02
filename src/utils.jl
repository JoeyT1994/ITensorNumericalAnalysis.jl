using Graphs: AbstractGraph
using ITensors:
  ITensors, ITensor, Index, dim, inds, combiner, array, tr, tags, uniqueinds, permute
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
  s = siteinds(ψ)
  rdm = permute(rdm, reduce(vcat, [s[v1], s[v2], s[v1]', s[v2]']))

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
