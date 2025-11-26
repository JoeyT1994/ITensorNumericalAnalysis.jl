using Base: Base
using TensorNetworkQuantumSimulator: siteinds, tensornetwork, graph, add
using ITensors: ITensor, dim, contract, onehot
using Graphs: Graphs

struct TensorNetworkFunction{V, IM <:AbstractIndexMap{V}} <: AbstractTensorNetwork{V}
  tensornetwork::TensorNetwork{V}
  indexmap::IM
end

default_contraction_alg(tnf::TensorNetworkFunction) = is_tree(tnf) ? "bp" : "exact"

TensorNetworkQuantumSimulator.tensornetwork(tnf::TensorNetworkFunction) = tnf.tensornetwork
TensorNetworkQuantumSimulator.graph(tnf::TensorNetworkFunction) = graph(tensornetwork(tnf))
indexmap(tnf::TensorNetworkFunction) = tnf.indexmap
TensorNetworkQuantumSimulator.siteinds(tnf::TensorNetworkFunction) = siteinds(indexmap(tnf))
TensorNetworkQuantumSimulator.setindex_preserve!(tnf::TensorNetworkFunction, value::ITensor, v) = setindex_preserve!(tensornetwork(tnf), value, v)
Base.getindex(tnf::TensorNetworkFunction, v) = getindex(tensornetwork(tnf), v)

#Needed for interface from AbstractTensorNetwork
function Base.copy(tnf::TensorNetworkFunction)
  return TensorNetworkFunction(copy(tensornetwork(tnf)), copy(indexmap(tnf)))
end

function TensorNetworkFunction(
  tn::AbstractTensorNetwork, dimension_vertices::Vector{Vector{V}}
) where {V}
  if tn isa TensorNetworkState
    tn = tensornetwork(tn)
  end
  s = siteinds(tn)
  return TensorNetworkFunction(tn, RealIndexMap(s, dimension_vertices))
end

function TensorNetworkFunction(
  tn::AbstractTensorNetwork,
  real_dimension_vertices::Vector{Vector{V}},
  imag_dimension_vertices::Vector{Vector{V}},
) where {V}
  if tn isa TensorNetworkState
    tn = tensornetwork(tn)
  end
  s = siteinds(tn)
  return TensorNetworkFunction(
    tn, ComplexIndexMap(s, real_dimension_vertices, imag_dimension_vertices)
  )
end

function TensorNetworkFunction(tn::TensorNetworkState)
  return TensorNetworkFunction(tensornetwork(tn), RealIndexMap(TensorNetworkQuantumSimulator.siteinds(tn)))
end

#Forward functionality from indexmap
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
    function $f(tnf::TensorNetworkFunction, args...; kwargs...)
      return $f(indexmap(tnf), args...; kwargs...)
    end
  end
end

function project(tnf::TensorNetworkFunction, ind_to_ind_value_map)
  tnf = copy(tnf)
  s = indexmap(tnf)
  for v in vertices(tnf)
    indices = siteinds(s, v)
    for ind in indices
      setindex_preserve!(tnf, tnf[v] * onehot(eltype(tnf[v]), ind => ind_to_ind_value_map[ind] + 1), v)
    end
  end
  return tnf
end

function evaluate(
  tnf::TensorNetworkFunction,
  xs::Vector,
  dims::Vector{<:Int}=dimensions(tnf);
  alg=default_contraction_alg(tnf),
  kwargs...,
)
  ind_to_ind_value_map = calculate_ind_values(tnf, xs, dims)
  tnf_xyz = project(tnf, ind_to_ind_value_map)
  return contract(tensornetwork(tnf_xyz); alg, kwargs...)
end

function evaluate(
  tnf::TensorNetworkFunction, x::Number, dim::Int=first(dimensions(tnf)); kwargs...
)
  return evaluate(tnf, [x], [dim]; kwargs...)
end

function TensorNetworkQuantumSimulator.truncate(tnf::TensorNetworkFunction; alg = is_tree(tnf) ? "bp" : nothing, kwargs...)
  tnf = copy(tnf)
  tn = TensorNetworkState(tensornetwork(tnf), siteinds(tnf))
  if alg == "boundarymps"
    tn = truncate(tn; alg, normalize_tensors = false, gauge_state = false, kwargs...)
  else
    tn = truncate(tn; alg, normalize_tensors = false, kwargs...)
  end

  return TensorNetworkFunction(tensornetwork(tn), indexmap(tnf))
end

function NamedGraphs.rename_vertices(f::Function, tnf::TensorNetworkFunction)
  return TensorNetworkFunction(
    rename_vertices(f, TensorNetwork(tnf)), rename_vertices(f, indsnetworkmap(tnf))
  )
end

function TensorNetworkQuantumSimulator.add(tnf1::TensorNetworkFunction, tnf2::TensorNetworkFunction)
  @assert siteinds(tnf1) == siteinds(tnf2)
  return TensorNetworkFunction(
    add(tensornetwork(tnf1), tensornetwork(tnf2)), indexmap(tnf1)
  )
end