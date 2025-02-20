using Graphs: AbstractEdge, src, dst
using ITensors: norm, tags, uniqueinds
using ITensorNetworks:
  AbstractITensorNetwork, ortho_region, underlying_graph, ITensorNetwork
using NamedGraphs.GraphsExtensions: bfs_tree, post_order_dfs_edges
using Graphs: steiner_tree

#
# Possible improvements:
# Use `orthogonalize` with interpolative backend
#

function _interpolative_gauge_edge(
  tn::AbstractITensorNetwork, edge::AbstractEdge; kwargs...
)
  tn = copy(tn)
  col_inds = uniqueinds(tn, edge)
  site_inds = siteinds(tn, src(edge))
  col_tags = tags(tn, edge)
  C, Z, inf_error = interpolative(
    tn[src(edge)], col_inds, site_inds; col_vertex=src(edge), tags=col_tags, kwargs...
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
Bring a TreeTensorNetwork into interpolative gauge 
towards a region
"""
function interpolative_gauge(ψ::TreeTensorNetwork, region::Vector; kwargs...)
  issetequal(region, ortho_region(ψ)) && return ψ
  st = steiner_tree(ITensorNetwork(ψ), union(region, ortho_region(ψ)))
  path = post_order_dfs_edges(st, first(region))
  path = filter(e -> !((src(e) ∈ region) && (dst(e) ∈ region)), path)
  if !isempty(path)
    for e in path
      ψ = typeof(ψ)(interpolative_gauge(ITensorNetwork(ψ), e; kwargs...))
    end
  end
  return set_ortho_region(ψ, region)
end

"""
Bring a TreeTensorNetwork into interpolative gauge 
towards a region
"""
function interpolative_gauge(ψ::AbstractTTN, region)
  return interpolative_gauge(ψ, [region])
end
