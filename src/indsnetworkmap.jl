using Base: Base
using Graphs: Graphs
using NamedGraphs: NamedGraphs
using ITensors: ITensors
using ITensorNetworks:
  ITensorNetworks, AbstractIndsNetwork, IndsNetwork, data_graph, underlying_graph

struct IndsNetworkMap{V,I,IN<:IndsNetwork{V,I},IM<:AbstractIndexMap} <:
       AbstractIndsNetwork{V,I}
  indsnetwork::IN
  indexmap::IM
end

indsnetwork(inm::IndsNetworkMap) = inm.indsnetwork
indexmap(inm::IndsNetworkMap) = inm.indexmap

indtype(inm::IndsNetworkMap) = indtype(typeof(indsnetwork(inm)))
indtype(::Type{<:IndsNetworkMap{V,I,IN,IM}}) where {V,I,IN,IM} = I
indexmaptype(inm::IndsNetworkMap) = typeof(indexmap(inm))
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
function RealIndsNetworkMap(s::IndsNetwork, args...; kwargs...)
  return IndsNetworkMap(s, RealIndexMap(s, args...; kwargs...))
end

function RealIndsNetworkMap(g::AbstractGraph, args...; base::Int=2, kwargs...)
  s = digit_siteinds(g, args...; base)
  return RealIndsNetworkMap(s, args...; kwargs...)
end

function RealIndsNetworkMap(s::IndsNetwork; map_dimension::Int64=1)
  return RealIndsNetworkMap(s, default_dimension_vertices(s; map_dimension))
end

function ComplexIndsNetworkMap(s::IndsNetwork, args...; kwargs...)
  return IndsNetworkMap(s, ComplexIndexMap(s, args...; kwargs...))
end

function ComplexIndsNetworkMap(g::AbstractGraph, args...; base::Int=2, kwargs...)
  s = complex_digit_siteinds(g, args...; base)
  return ComplexIndsNetworkMap(s, args...; kwargs...)
end

function ComplexIndsNetworkMap(s::IndsNetwork; map_dimension::Int64=1)
  return ComplexIndsNetworkMap(
    s,
    default_dimension_vertices(s; map_dimension),
    default_dimension_vertices(s; map_dimension),
  )
end

const continuous_siteinds = RealIndsNetworkMap
const real_continuous_siteinds = RealIndsNetworkMap
const complex_continuous_siteinds = ComplexIndsNetworkMap

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

function vertex_dimensions(inm::IndsNetworkMap, v)
  return [dimension(inm, i) for i in inds(inm, v)]
end

function vertex_digits(inm::IndsNetworkMap, v)
  return [digit(inm, i) for i in inds(inm, v)]
end

function vertex_digit(inm::IndsNetworkMap, v)
  return digit(inm, only(inds(inm, v)))
end

#function dimension_vertices(inm::IndsNetworkMap, dimension::Int)
#  return filter(v -> all(d -> d == dimension, vertex_dimensions(inm, v)), vertices(inm))
#end

function dimension_vertices(inm::IndsNetworkMap, dimension::Int)
  return filter(v -> dimension ∈ vertex_dimensions(inm, v), vertices(inm))
end

function vertex(inm::IndsNetworkMap, dimension::Int, digit::Int)
  index = ind(inm, dimension, digit)
  return only(filter(v -> index ∈ inm[v], vertices(inm)))
end
