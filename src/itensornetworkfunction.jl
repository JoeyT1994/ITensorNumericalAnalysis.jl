using ITensorNetworks: data_graph_type, AbstractITensorNetwork
using ITensors: ITensor, dim, contract, siteinds

struct ITensorNetworkFunction{V,TN<:AbstractITensorNetwork{V},BM<:BitMap} <:
       AbstractITensorNetwork{V}
  itensornetwork::TN
  bit_map::BM
end

itensornetwork(fitn::ITensorNetworkFunction) = fitn.itensornetwork
bit_map(fitn::ITensorNetworkFunction) = fitn.bit_map

#Needed for interface from AbstractITensorNetwork
function ITensorNetworks.data_graph_type(TN::Type{<:ITensorNetworkFunction})
  return data_graph_type(fieldtype(TN, :itensornetwork))
end
ITensorNetworks.data_graph(fitn::ITensorNetworkFunction) = data_graph(itensornetwork(fitn))
function Base.copy(fitn::ITensorNetworkFunction)
  return ITensorNetworkFunction(copy(itensornetwork(fitn)), copy(bit_map(fitn)))
end

function ITensorNetworkFunction(
  itn::AbstractITensorNetwork, dimension_vertices::Vector{Vector{V}}
) where {V}
  return ITensorNetworkFunction(itn, BitMap(dimension_vertices))
end

#Constructor, assume one-dimensional and ordered as vertices of the itn
function ITensorNetworkFunction(itn::AbstractITensorNetwork)
  return ITensorNetworkFunction(itn, BitMap(itn))
end

#Forward functionality from bit_map
for f in [
  :vertex,
  :dimension,
  :bit,
  :(Graphs.vertices),
  :calculate_bit_values,
  :calculate_x,
  :calculate_xyz,
]
  @eval begin
    function $f(fitn::ITensorNetworkFunction, args...; kwargs...)
      return $f(bit_map(fitn), args...; kwargs...)
    end
  end
end

function project(fitn::ITensorNetworkFunction, vertex_to_bit_value_map)
  fitn = copy(fitn)
  s = siteinds(fitn)
  for v in keys(vertex_to_bit_value_map)
    proj = ITensor(
      [i != vertex_to_bit_value_map[v] ? 0 : 1 for i in 0:(dim(s[v]) - 1)], s[v]
    )
    fitn[v] = fitn[v] * proj
  end
  return fitn
end

function calculate_fxyz(
  fitn::ITensorNetworkFunction, xs::Vector{Float64}, dimensions::Vector{Int64}
)
  vertex_to_bit_value_map = calculate_bit_values(fitn, xs, dimensions)
  fitn_xyz = project(fitn, vertex_to_bit_value_map)
  return contract(fitn_xyz)[]
end

function calculate_fxyz(fitn::ITensorNetworkFunction, xs::Vector{Float64})
  return calculate_fxyz(fitn, xs, [i for i in 1:length(xs)])
end

function calculate_fx(fitn::ITensorNetworkFunction, x::Float64)
  @assert dimension(fitn) == 1
  return calculate_fxyz(fitn, [x], [1])
end
