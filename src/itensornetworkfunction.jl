using ITensorNetworks: ITensorNetworks, AbstractITensorNetwork, data_graph, data_graph_type
using ITensors: ITensor, dim, contract, siteinds, onehot
using Graphs: Graphs

struct ITensorNetworkFunction{V,TN<:AbstractITensorNetwork{V},IM<:IndexMap} <:
       AbstractITensorNetwork{V}
  itensornetwork::TN
  indexmap::IM
end

itensornetwork(fitn::ITensorNetworkFunction) = fitn.itensornetwork
indexmap(fitn::ITensorNetworkFunction) = fitn.indexmap

#Needed for interface from AbstractITensorNetwork
function ITensorNetworks.data_graph_type(TN::Type{<:ITensorNetworkFunction})
  return data_graph_type(fieldtype(TN, :itensornetwork))
end
ITensorNetworks.data_graph(fitn::ITensorNetworkFunction) = data_graph(itensornetwork(fitn))
function Base.copy(fitn::ITensorNetworkFunction)
  return ITensorNetworkFunction(copy(itensornetwork(fitn)), copy(indexmap(fitn)))
end

function ITensorNetworkFunction(
  itn::AbstractITensorNetwork, dimension_vertices::Vector{Vector{V}}
) where {V}
  s = siteinds(itn)
  return ITensorNetworkFunction(itn, IndexMap(s, dimension_vertices))
end

#Constructor, assume one-dimensional and ordered as vertices of the itn
function ITensorNetworkFunction(itn::AbstractITensorNetwork)
  return ITensorNetworkFunction(itn, IndexMap(siteinds(itn)))
end

#Forward functionality from indexmap
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
]
  @eval begin
    function $f(fitn::ITensorNetworkFunction, args...; kwargs...)
      return $f(indexmap(fitn), args...; kwargs...)
    end
  end
end

for f in [
  :vertices_dimensions,
  :vertices_digits,
  :vertex_digit,
  :vertex_dimension,
  :dimension_vertices,
]
  @eval begin
    function $f(fitn::ITensorNetworkFunction, args...; kwargs...)
      return $f(siteinds(fitn), indexmap(fitn), args...; kwargs...)
    end
  end
end

function project(fitn::ITensorNetworkFunction, ind_to_ind_value_map)
  fitn = copy(fitn)
  s = siteinds(fitn)
  for v in vertices(fitn)
    indices = inds(s, v)
    for ind in indices
      fitn[v] = fitn[v] * onehot(eltype(fitn[v]), ind => ind_to_ind_value_map[ind] + 1)
    end
  end
  return fitn
end

function calculate_fxyz(
  fitn::ITensorNetworkFunction, xs::Vector{Float64}, dimensions::Vector{Int64}
)
  ind_to_ind_value_map = calculate_ind_values(fitn, xs, dimensions)
  fitn_xyz = project(fitn, ind_to_ind_value_map)
  return contract(fitn_xyz)[]
end

function calculate_fxyz(fitn::ITensorNetworkFunction, xs::Vector{Float64})
  return calculate_fxyz(fitn, xs, [i for i in 1:length(xs)])
end

function calculate_fx(fitn::ITensorNetworkFunction, x::Float64)
  @assert dimension(fitn) == 1
  return calculate_fxyz(fitn, [x], [1])
end

function ITensorNetworks.truncate(fitn::ITensorNetworkFunction; kwargs...)
  @assert is_tree(fitn)
  ψ = truncate(ttn(itensornetwork(fitn)); kwargs...)
  return ITensorNetworkFunction(ITensorNetwork(ψ), indexmap(fitn))
end
