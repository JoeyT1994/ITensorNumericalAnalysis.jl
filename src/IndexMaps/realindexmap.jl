using Base: Base
using Dictionaries: Dictionary, set!
using ITensors: ITensors, Index, dim
using ITensorNetworks: IndsNetwork, vertex_data

struct RealIndexMap{VB,VD} <: AbstractIndexMap{VB,VD}
  index_digit::VB
  index_dimension::VD
end

index_digit(imap::RealIndexMap) = imap.index_digit
index_dimension(imap::RealIndexMap) = imap.index_dimension
function index_value_to_scalar(imap::RealIndexMap, ind::Index, value::Int)
  return (value) / (dim(ind)^digit(imap, ind))
end
function Base.copy(imap::RealIndexMap)
  return IndexMap(copy(index_digit(imap)), copy(index_dimension(imap)))
end
function ITensors.inds(imap::RealIndexMap)
  @assert keys(index_dimension(imap)) == keys(index_digit(imap))
  return collect(keys(index_dimension(imap)))
end
function ind(imap::RealIndexMap, dim::Int, digit::Int)
  return only(
    filter(
      i -> index_dimension(imap)[i] == dim && index_digit(imap)[i] == digit,
      keys(index_dimension(imap)),
    ),
  )
end

function RealIndexMap(
  s::IndsNetwork, dimension_vertices::Vector{Vector{V}}=default_dimension_vertices(s)
) where {V}
  dimension_indices = Vector{Index}[inds(s, vertices) for vertices in dimension_vertices]
  return RealIndexMap(dimension_indices)
end

function RealIndexMap(dimension_indices::Vector{Vector{V}}) where {V<:Index}
  index_digit = Dictionary()
  index_dimension = Dictionary()
  for (d, indices) in enumerate(dimension_indices)
    for (bit, ind) in enumerate(indices)
      set!(index_digit, ind, bit)
      set!(index_dimension, ind, d)
    end
  end
  return RealIndexMap(index_digit, index_dimension)
end

function calculate_ind_values(imap::RealIndexMap, xs::Vector, dims::Vector{Int})
  @assert length(xs) == length(dims)
  ind_to_ind_value_map = Dictionary()
  for (i, x) in enumerate(xs)
    d = dims[i]
    indices = dimension_inds(imap, d)
    sorted_inds = sort(indices; by=indices -> digit(imap, indices))
    set_ind_values!(ind_to_ind_value_map, imap, sorted_inds, x)
  end
  return ind_to_ind_value_map
end

function grid_points(imap::RealIndexMap, N::Int, d::Int)
  dims = dim.(dimension_inds(imap, d))
  @assert all(y -> y == first(dims), dims)
  base = first(dims)
  L = length(dimension_inds(imap, d))
  a = round(base^L / N)
  grid_points = [i * (a / base^L) for i in 0:(N + 1)]
  return filter(x -> x < 1, grid_points)
end
