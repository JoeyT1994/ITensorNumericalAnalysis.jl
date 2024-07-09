using Graphs: AbstractEdge, src, dst
using ITensors: norm, tags, uniqueinds
using ITensorNetworks: AbstractITensorNetwork
using NamedGraphs.GraphsExtensions: bfs_tree, post_order_dfs_edges

#
# Possible improvements:
# Use `orthogonalize` with interpolative backend
#

function _interpolative_gauge_edge(
  tn::AbstractITensorNetwork, edge::AbstractEdge; kwargs...
)
  tn = copy(tn)
  col_inds = uniqueinds(tn, edge)
  col_tags = tags(tn, edge)
  C, Z, inf_error = interpolative(
    tn[src(edge)], col_inds; col_vertex=src(edge), tags=col_tags, kwargs...
  )
  tn[src(edge)] = Z
  tn[dst(edge)] *= C
  return tn
end

function interpolative_gauge(tn::AbstractITensorNetwork, edge::AbstractEdge; kwargs...)
  return _interpolative_gauge_edge(tn, edge; kwargs...)
end

function interpolative_gauge(tn::AbstractITensorNetwork, edge::Pair; kwargs...)
  return interpolative_gauge(tn, edgetype(tn)(edge); kwargs...)
end

"""
Bring an ITensorNetwork into interpolative gauge 
towards a source vertex, treating
the network as a tree spanned by a spanning tree.
"""
function interpolative_gauge(ψ::AbstractITensorNetwork, source_vertex)
  spanning_tree_edges = post_order_dfs_edges(bfs_tree(ψ, source_vertex), source_vertex)
  for e in spanning_tree_edges
    ψ = interpolative_gauge(ψ, e)
  end
  return ψ
end
