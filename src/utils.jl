using Graphs: AbstractGraph
using ITensors:
  ITensors, ITensor, Index, dim, inds, combiner, array, tr, tags, uniqueinds, permute
using NamedGraphs: NamedGraph, NamedEdge, NamedGraphs, rename_vertices, src, dst
using NamedGraphs.GraphsExtensions: rem_vertex
using NamedGraphs.PartitionedGraphs:
  PartitionEdge, partitionvertices, PartitionVertex
using Dictionaries: Dictionary, collect, values, keys
using TensorNetworkQuantumSimulator: AbstractTensorNetwork, add_tensor!

"""Build the order L tensor corresponding to fx(x): x ∈ [0,1], default decomposition is binary"""
function build_full_rank_tensor(L::Int, fx::Function; base::Int=2)
  inds = [Index(base, "$i") for i in 1:L]
  dims = Tuple([base for i in 1:L])
  array = zeros(dims)
  for i in 0:(base ^ (L) - 1)
    xis = digits(i; base, pad=L)
    x = sum([xis[i] / (base^i) for i in 1:L])
    array[Tuple(xis + ones(Int, (L)))...] = fx(x)
  end

  return ITensor(array, inds)
end

"""Build the tensor C such that C_{phys_ind, virt_inds...} = delta_{virt_inds...}"""
function c_tensor(phys_inds::Vector, virt_inds::Vector)
  @assert allequal(dim.(virt_inds))
  T = ITensors.delta(Int64, virt_inds)
  T = T * ITensor(1, phys_inds...)
  return T
end

function ITensors.inds(siteinds::Dictionary, verts::Vector)
  return reduce(vcat, [siteinds[v] for v in verts])
end

function ITensors.inds(s::Dictionary)
  return inds(s, collect(keys(s)))
end

function base(siteinds::Dictionary)
  indices = collect(values(siteinds))
  dims = dim.(indices)
  @assert all(d -> d == first(dims), dims)
  return first(dims)
end

# """Compute the two-site rdm from a tree-tensor network, sclaes as O(Lchi^{z+1})"""
# function two_site_rdm(
#   ψ::AbstractITensorNetwork, v1, v2; (cache!)=nothing, cache_update_kwargs=(;)
# )
#   ψIψ_bpc = if isnothing(cache!)
#     update(BeliefPropagationCache(QuadraticFormNetwork(ψ)); cache_update_kwargs...)
#   else
#     cache![]
#   end
#   ψIψ = tensornetwork(ψIψ_bpc)
#   pg = partitioned_tensornetwork(ψIψ_bpc)

#   path = PartitionEdge.(a_star(partitioned_graph(ψIψ_bpc), v1, v2))
#   pg = rem_vertex(pg, operator_vertex(ψIψ, v1))
#   pg = rem_vertex(pg, operator_vertex(ψIψ, v2))
#   ψIψ_bpc_mod = BeliefPropagationCache(pg, messages(ψIψ_bpc), default_message)
#   ψIψ_bpc_mod = update(
#     ψIψ_bpc_mod, path; message_update=ms -> default_message_update(ms; normalize=false)
#   )
#   incoming_mts = environment(ψIψ_bpc_mod, [PartitionVertex(v2)])
#   local_state = only(factors(ψIψ_bpc_mod, PartitionVertex(v2)))
#   rdm = contract(vcat(incoming_mts, local_state); sequence="automatic")
#   s = siteinds(ψ)
#   rdm = permute(rdm, reduce(vcat, [s[v1], s[v2], s[v1]', s[v2]']))

#   rdm = array((rdm * combiner(inds(rdm; plev=0)...)) * combiner(inds(rdm; plev=1)...))
#   rdm /= tr(rdm)
#   return rdm
# end

#Given an itensornetwork, contract away any tensors which don't have external indices.
function merge_internal_tensors(tn::AbstractTensorNetwork)
  tn = copy(tn)
  internal_vertices = filter(v -> isempty(uniqueinds(tn, v)), collect(vertices(tn)))
  external_vertices = filter(v -> !isempty(uniqueinds(tn, v)), collect(vertices(tn)))
  for v in internal_vertices
    vns = neighbors(tn, v)
    if !isempty(vns)
      tnvn = tn[v] * tn[first(vns)]
      rem_vertex!(tn, v)
      rem_vertex!(tn, first(vns))
      add_tensor!(tn, tnvn, first(vns))
    else
      setindex_preserve!(tn, tn[first(external_vertices)] * (tn[v][]), first(external_vertices))
      rem_vertex!(tn, v)
    end
  end
  return tn
end
