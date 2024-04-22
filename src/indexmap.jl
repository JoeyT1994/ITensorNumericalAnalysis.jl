using Base: Base
using Dictionaries: Dictionary, set!
using ITensors: ITensors, Index, dim
using ITensorNetworks: IndsNetwork, vertex_data

struct IndexMap{VB,VD}
  index_digit::VB
  index_dimension::VD
end

index_digit(im::IndexMap) = im.index_digit
index_dimension(im::IndexMap) = im.index_dimension

function default_digit_map(indices::Vector{Index}; map_dimension::Int=1)
  return Dictionary(
    indices, [ceil(Int, i / map_dimension) for (i, ind) in enumerate(indices)]
  )
end
function default_dimension_map(indices::Vector{Index}; map_dimension::Int)
  return Dictionary(indices, [(i % map_dimension) + 1 for (i, ind) in enumerate(indices)])
end

function IndexMap(s::IndsNetwork; map_dimension::Int=1)
  indices = inds(s)
  return IndexMap(
    default_digit_map(indices; map_dimension), default_dimension_map(indices; map_dimension)
  )
end

function IndexMap(s::IndsNetwork, dimension_vertices::Vector{Vector{V}}) where {V}
  dimension_indices = Vector{Index}[]
  for vertices in dimension_vertices
    indices = inds(s, vertices)
    push!(dimension_indices, indices)
  end
  return IndexMap(dimension_indices)
end

function IndexMap(dimension_indices::Vector{Vector{V}}) where {V<:Index}
  index_digit = Dictionary()
  index_dimension = Dictionary()
  for (dimension, indices) in enumerate(dimension_indices)
    for (bit, ind) in enumerate(indices)
      set!(index_digit, ind, bit)
      set!(index_dimension, ind, dimension)
    end
  end
  return IndexMap(index_digit, index_dimension)
end

function Base.copy(im::IndexMap)
  return IndexMap(copy(index_digit(im)), copy(index_dimension(im)))
end

dimension(im::IndexMap) = maximum(collect(values(index_dimension(im))))
dimension(im::IndexMap, ind::Index) = index_dimension(im)[ind]
dimensions(im::IndexMap, inds::Vector{Index}) = dimension.(inds)
digit(im::IndexMap, ind::Index) = index_digit(im)[ind]
digits(im::IndexMap, inds::Vector{Index}) = digit.(inds)
function index_value_to_scalar(im::IndexMap, ind::Index, value::Int)
  return (value) / (dim(ind)^digit(im, ind))
end
function index_values_to_scalars(im::IndexMap, ind::Index)
  return [index_value_to_scalar(im, ind, i) for i in 0:(dim(ind) - 1)]
end

function ITensors.inds(im::IndexMap)
  @assert keys(index_dimension(im)) == keys(index_digit(im))
  return collect(keys(index_dimension(im)))
end
function dimension_inds(im::IndexMap, dimension::Int)
  return collect(
    filter(i -> index_dimension(im)[i] == dimension, keys(index_dimension(im)))
  )
end
function ind(im::IndexMap, dimension::Int, digit::Int)
  return only(
    filter(
      i -> index_dimension(im)[i] == dimension && index_digit(im)[i] == digit,
      keys(index_dimension(im)),
    ),
  )
end

function dimension_indices(im::IndexMap, dim::Int)
  return filter(ind -> dimension(im, ind) == dim, inds(im))
end

function calculate_xyz(im::IndexMap, ind_to_ind_value_map, dimensions::Vector{Int})
  out = Number[]
  for dimension in dimensions
    indices = dimension_inds(im, dimension)
    push!(
      out,
      sum([index_value_to_scalar(im, ind, ind_to_ind_value_map[ind]) for ind in indices]),
    )
  end
  return out
end

function calculate_xyz(im::IndexMap, ind_to_ind_value_map)
  return calculate_xyz(im, ind_to_ind_value_map, [i for i in 1:dimension(im)])
end
function calculate_x(im::IndexMap, ind_to_ind_value_map, dimension::Int)
  return only(calculate_xyz(im, ind_to_ind_value_map, [dimension]))
end
function calculate_x(im::IndexMap, ind_to_ind_value_map)
  return calculate_x(im, ind_to_ind_value_map, 1)
end

function calculate_ind_values(
  im::IndexMap, xs::Vector, dimensions::Vector{Int}; print_x=false
)
  @assert length(xs) == length(dimensions)
  ind_to_ind_value_map = Dictionary()
  for (i, x) in enumerate(xs)
    dimension = dimensions[i]
    x_rn = x
    indices = dimension_inds(im, dimension)
    sorted_inds = sort(indices; by=indices -> digit(im, indices))
    for ind in sorted_inds
      ind_val = dim(ind) - 1
      ind_set = false
      while (!ind_set)
        if x_rn >= index_value_to_scalar(im, ind, ind_val)
          set!(ind_to_ind_value_map, ind, ind_val)
          x_rn -= index_value_to_scalar(im, ind, ind_val)
          ind_set = true
        else
          ind_val = ind_val - 1
        end
      end
    end

    if print_x
      x_bitstring = calculate_x(im, ind_to_ind_value_map, dimension)
      println(
        "Dimension $dimension. Actual value of x is $x but bitstring rep. is $x_bitstring"
      )
    end
  end
  return ind_to_ind_value_map
end

function calculate_ind_values(im::IndexMap, x::Number, dimension::Int; kwargs...)
  return calculate_ind_values(im, [x], [dimension]; kwargs...)
end
function calculate_ind_values(im::IndexMap, xs::Vector; kwargs...)
  return calculate_ind_values(im, xs, [i for i in 1:length(xs)]; kwargs...)
end
function calculate_ind_values(im::IndexMap, x::Number; kwargs...)
  return calculate_ind_values(im, [x], [1]; kwargs...)
end

function grid_points(im::IndexMap, N::Int, dimension::Int)
  dims = dim.(dimension_inds(im, dimension))
  @assert all(y -> y == first(dims), dims)
  base = first(dims)
  vals = Vector
  L = length(dimension_inds(im, dimension))
  a = round(base^L / N)
  grid_points = [i * (a / base^L) for i in 0:(N + 1)]
  return filter(x -> x <= 1, grid_points)
end
