using Base: Base
using Graphs: Graphs
using NamedGraphs: NamedGraphs, NamedGraph
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
  s::IndsNetwork,
  real_dimension_vertices::Vector{Vector{V}},
  imaginary_dimension_vertices::Vector{Vector{V}}=Vector{vertextype(s)}[[]];
  is_complex=false,
  kwargs...,
) where {V}
  return IndsNetworkMap(
    s, IndexMap(s, real_dimension_vertices, imaginary_dimension_vertices; kwargs...)
  )
end

function IndsNetworkMap(
  s::IndsNetwork,
  real_dimension_indices::Vector{Vector{Index}},
  imaginary_dimension_indices::Vector{Vector{Index}}=Vector{Index}[[]];
  kwargs...,
)
  return IndsNetworkMap(
    s, IndexMap(real_dimension_indices, imaginary_dimension_indices; kwargs...)
  )
end

function IndsNetworkMap(s::IndsNetwork; kwargs...)
  return IndsNetworkMap(s, IndexMap(s; kwargs...))
end

function IndsNetworkMap(g::NamedGraph, args...; base::Int=2, is_complex=false, kwargs...)
  s = digit_siteinds(g; base, is_complex)
  return IndsNetworkMap(s, args...; is_complex, kwargs...)
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
  :calculate_x,
  :calculate_xyz,
  :grid_points,
  :index_value_to_scalar,
  :index_values_to_scalars,
  :imaginary_inds,
  :real_inds,
  :is_real,
  :is_imaginary,
  :is_complex,
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

function dimension_vertices(inm::IndsNetworkMap, dimension::Int)
  return filter(v -> all(d -> d == dimension, vertex_dimensions(inm, v)), vertices(inm))
end

function vertex(inm::IndsNetworkMap, dimension::Int, digit::Int)
  index = ind(inm, dimension, digit)
  return only(filter(v -> index ∈ inm[v], vertices(inm)))
end
