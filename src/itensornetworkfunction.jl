using Base: Base
using ITensorNetworks:
  ITensorNetworks, AbstractITensorNetwork, data_graph, data_graph_type, scalar
using ITensors: ITensor, dim, contract, siteinds, onehot
using Graphs: Graphs

struct ITensorNetworkFunction{V,TN<:AbstractITensorNetwork{V},INM<:IndsNetworkMap} <:
       AbstractITensorNetwork{V}
  itensornetwork::TN
  indsnetworkmap::INM
end

itensornetwork(fitn::ITensorNetworkFunction) = fitn.itensornetwork
indsnetworkmap(fitn::ITensorNetworkFunction) = fitn.indsnetworkmap

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
  return ITensorNetworkFunction(itn, IndsNetworkMap(s, dimension_vertices))
end

function ITensorNetworkFunction(itn::AbstractITensorNetwork)
  return ITensorNetworkFunction(itn, IndsNetworkMap(siteinds(itn)))
end

#Forward functionality from indsnetworkmap
for f in [
  :ind,
  :dimension,
  :dimensions,
  :digit,
  :digits,
  :calculate_ind_values,
  :calculate_x,
  :calculate_xyz,
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

function calculate_fxyz(fitn::ITensorNetworkFunction, xs::Vector, dimensions::Vector{Int})
  ind_to_ind_value_map = calculate_ind_values(fitn, xs, dimensions)
  fitn_xyz = project(fitn, ind_to_ind_value_map)
  return scalar(itensornetwork(fitn_xyz); alg="bp")
end

function calculate_fxyz(fitn::ITensorNetworkFunction, xs::Vector)
  return calculate_fxyz(fitn, xs, [i for i in 1:length(xs)])
end

function calculate_fx(fitn::ITensorNetworkFunction, x::Number)
  @assert dimension(fitn) == 1
  return calculate_fxyz(fitn, [x], [1])
end

function ITensorNetworks.truncate(fitn::ITensorNetworkFunction; kwargs...)
  @assert is_tree(fitn)
  ψ = truncate(ttn(itensornetwork(fitn)); kwargs...)
  return ITensorNetworkFunction(ITensorNetwork(ψ), indsnetworkmap(fitn))
end
