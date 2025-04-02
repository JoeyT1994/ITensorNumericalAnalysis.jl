using Base: Base
using ITensorNetworks:
  ITensorNetworks,
  ITensorNetwork,
  AbstractITensorNetwork,
  data_graph,
  data_graph_type,
  scalar,
  inner,
  TreeTensorNetwork,
  maxlinkdim,
  siteinds
using ITensors: ITensor, dim, contract, onehot
using Graphs: Graphs

default_contraction_alg() = "bp"

struct ITensorNetworkFunction{V,TN<:AbstractITensorNetwork{V},INM<:IndsNetworkMap} <:
       AbstractITensorNetwork{V}
  itensornetwork::TN
  indsnetworkmap::INM
end

itensornetwork(fitn::ITensorNetworkFunction) = fitn.itensornetwork
indsnetworkmap(fitn::ITensorNetworkFunction) = fitn.indsnetworkmap
indexmap(fitn::ITensorNetworkFunction) = indexmap(indsnetworkmap(fitn))

#Needed for interface from AbstractITensorNetwork
function ITensorNetworks.data_graph_type(TN::Type{<:ITensorNetworkFunction})
  return data_graph_type(fieldtype(TN, :itensornetwork))
end
ITensorNetworks.data_graph(fitn::ITensorNetworkFunction) = data_graph(itensornetwork(fitn))
function Base.copy(fitn::ITensorNetworkFunction)
  return ITensorNetworkFunction(copy(itensornetwork(fitn)), copy(indsnetworkmap(fitn)))
end

function ITensorNetworkFunction(
  itn::AbstractITensorNetwork, dimension_vertices::Vector{Vector{V}}
) where {V}
  s = siteinds(itn)
  return ITensorNetworkFunction(itn, RealIndsNetworkMap(s, dimension_vertices))
end

function ITensorNetworkFunction(
  itn::AbstractITensorNetwork,
  real_dimension_vertices::Vector{Vector{V}},
  imag_dimension_vertices::Vector{Vector{V}},
) where {V}
  s = siteinds(itn)
  return ITensorNetworkFunction(
    itn, ComplexIndsNetworkMap(s, real_dimension_vertices, imag_dimension_vertices)
  )
end

function ITensorNetworkFunction(itn::AbstractITensorNetwork)
  return ITensorNetworkFunction(itn, RealIndsNetworkMap(siteinds(itn)))
end

#Forward functionality from indsnetworkmap
for f in [
  :ind,
  :dimension,
  :dimensions,
  :digit,
  :digits,
  :calculate_ind_values,
  :calculate_p,
  :grid_points,
  :vertices_dimensions,
  :vertices_digits,
  :vertex_digit,
  :vertex_dimension,
  :dimension_vertices,
]
  @eval begin
    function $f(fitn::ITensorNetworkFunction, args...; kwargs...)
      return $f(indsnetworkmap(fitn), args...; kwargs...)
    end
  end
end

ITensors.siteinds(fitn::ITensorNetworkFunction) = indsnetwork(indsnetworkmap(fitn))

function project(fitn::ITensorNetworkFunction, ind_to_ind_value_map)
  fitn = copy(fitn)
  s = indsnetwork(indsnetworkmap(fitn))
  for v in vertices(fitn)
    indices = inds(s, v)
    for ind in indices
      fitn[v] = fitn[v] * onehot(eltype(fitn[v]), ind => ind_to_ind_value_map[ind] + 1)
    end
  end
  return fitn
end

function evaluate(
  fitn::ITensorNetworkFunction,
  xs::Vector,
  dims::Vector{<:Int}=dimensions(fitn);
  alg=default_contraction_alg(),
  kwargs...,
)
  ind_to_ind_value_map = calculate_ind_values(fitn, xs, dims)
  fitn_xyz = project(fitn, ind_to_ind_value_map)
  return scalar(itensornetwork(fitn_xyz); alg, kwargs...)
end

function evaluate(
  fitn::ITensorNetworkFunction, x::Number, dim::Int=first(dimensions(fitn)); kwargs...
)
  return evaluate(fitn, [x], [dim]; kwargs...)
end

function ITensorNetworks.truncate(fitn::ITensorNetworkFunction; kwargs...)
  @assert is_tree(fitn)
  ψ = truncate(ttn(itensornetwork(fitn)); kwargs...)
  return ITensorNetworkFunction(ITensorNetwork(ψ), indsnetworkmap(fitn))
end

function NamedGraphs.rename_vertices(f::Function, fitn::ITensorNetworkFunction)
  return ITensorNetworkFunction(
    rename_vertices(f, itensornetwork(fitn)), rename_vertices(f, indsnetworkmap(fitn))
  )
end
