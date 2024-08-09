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

function ITensorNetworks.default_message_update(
  contract_list::Vector{ITensor}; normalize=true, kwargs...
)
  sequence = optimal_contraction_sequence(contract_list)
  updated_messages = contract(contract_list; sequence, kwargs...)
  message_norm = norm(updated_messages)
  if normalize && !iszero(message_norm)
    updated_messages /= message_norm
  end
  return ITensor[updated_messages]
end

function ITensorNetworks.message(bp_cache::BeliefPropagationCache, edge::PartitionEdge)
  mts = messages(bp_cache)
  return get(() -> default_message(bp_cache, edge), mts, edge)
end
