using Base: Base
using Graphs: Graphs
using NamedGraphs: NamedGraphs
using ITensors: ITensors
using ITensorNetworks:
  ITensorNetworks, AbstractIndsNetwork, IndsNetwork, data_graph, underlying_graph

struct IndsNetworkMap{V,I,IN<:IndsNetwork{V,I},IM} <: AbstractIndsNetwork{V,I}
  indsnetwork::IN
  indexmap::IM
end

indsnetwork(inm::IndsNetworkMap) = inm.indsnetwork
indexmap(inm::IndsNetworkMap) = inm.indexmap

indtype(inm::IndsNetworkMap) = indtype(typeof(indsnetwork(inm)))
indtype(::Type{<:IndsNetworkMap{V,I,IN,IM}}) where {V,I,IN,IM} = I
ITensorNetworks.data_graph(inm::IndsNetworkMap) = data_graph(indsnetwork(inm))
function ITensorNetworks.underlying_graph(inm::IndsNetworkMap)
  return underlying_graph(data_graph(indsnetwork(inm)))
end
NamedGraphs.vertextype(::Type{<:IndsNetworkMap{V,I,IN,IM}}) where {V,I,IN,IM} = V
ITensorNetworks.underlying_graph_type(G::Type{<:IndsNetworkMap}) = NamedGraph{vertextype(G)}
Graphs.is_directed(::Type{<:IndsNetworkMap}) = false

function Base.copy(inm::IndsNetworkMap)
  return IndsNetworkMap(indsnetwork(inm), indexmap(inm))
end

#Constructors 
function IndsNetworkMap(
  s::IndsNetwork, dimension_vertices::Vector{Vector{V}}; kwargs...
) where {V}
  return IndsNetworkMap(s, IndexMap(s, dimension_vertices; kwargs...))
end

function IndsNetworkMap(s::IndsNetwork, dimension_indices::Vector{Vector{Index}})
  return IndsNetworkMap(s, IndexMap(dimension_indices; kwargs...))
end

function IndsNetworkMap(s::IndsNetwork; kwargs...)
  return IndsNetworkMap(s, IndexMap(s; kwargs...))
end

function IndsNetworkMap(g::AbstractGraph, args...; base::Int=2, kwargs...)
  s = digit_siteinds(g; base)
  return IndsNetworkMap(s, args...; kwargs...)
end

const continuous_siteinds = IndsNetworkMap

#Forward functionality from indexmap
for f in [
  :ind,
  :dimension,
  :dimension_inds,
  :dimensions,
  :digit,
  :digits,
  :calculate_ind_values,
  :calculate_p,
  :grid_points,
  :index_value_to_scalar,
  :index_values_to_scalars,
]
  @eval begin
    function $f(inm::IndsNetworkMap, args...; kwargs...)
      return $f(indexmap(inm), args...; kwargs...)
    end
  end
end

#Functions on indsnetwork that don't seem to autoforward
function ITensors.inds(inm::IndsNetworkMap, args...; kwargs...)
  return inds(indsnetwork(inm), args...; kwargs...)
end

base(inm::IndsNetworkMap) = base(indsnetwork(inm))

function vertices_dimensions(inm::IndsNetworkMap, verts::Vector)
  return [dimension(inm, i) for i in inds(inm, verts)]
end

function vertices_digits(inm::IndsNetworkMap, verts::Vector)
  return [digit(inm, i) for i in inds(inm, verts)]
end

function vertex_dimension(inm::IndsNetworkMap, v)
  return dimension(inm, only(inds(inm, v)))
end

function vertex_digit(inm::IndsNetworkMap, v)
  return digit(inm, only(inds(inm, v)))
end

function dimension_vertices(inm::IndsNetworkMap, dim::Int)
  return dimension_vertices(inm, [dim])
end

function dimension_vertices(inm::IndsNetworkMap, dims::Vector{Int})
  return filter(v -> vertex_dimension(inm, v) in dims, vertices(inm))
end

function vertex(inm::IndsNetworkMap, dim::Int, digit::Int)
  index = ind(inm, dim, digit)
  return only(filter(v -> index ∈ inm[v], vertices(inm)))
end
