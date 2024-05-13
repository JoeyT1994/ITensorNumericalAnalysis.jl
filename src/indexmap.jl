using Base: Base
using Dictionaries: Dictionary, set!
using ITensors: ITensors, Index, dim, hastags
using ITensorNetworks: IndsNetwork, vertex_data
using NamedGraphs: vertextype

struct IndexMap{VB,VD}
  index_digit::VB
  index_dimension::VD
end

function is_real(ind::Index)
  return !hastags(ind, "DigitIm")
end

index_digit(imap::IndexMap) = imap.index_digit
index_dimension(imap::IndexMap) = imap.index_dimension

function default_digit_map(indices::Vector{Index}; map_dimension::Int=1)
  return Dictionary(
    indices, [ceil(Int, i / map_dimension) for (i, ind) in enumerate(indices)]
  )
end
function default_dimension_map(indices::Vector{Index}; map_dimension::Int)
  return Dictionary(indices, [(i % map_dimension) + 1 for (i, ind) in enumerate(indices)])
end
function default_dimension_vertices(g::AbstractGraph; map_dimension)
  verts = collect(vertices(g))
  L = length(verts)
  return Vector{vertextype(g)}[verts[i:map_dimension:L] for i in 1:map_dimension]
end

function IndexMap(s::IndsNetwork, dimension_vertices::Vector{Vector{V}}; kwargs...) where {V}
  dimension_indices = Vector{Index}[]
  for vertices in dimension_vertices
    indices = inds(s, vertices)
    push!(dimension_indices, indices)
  end
  return IndexMap(dimension_indices; kwargs...)
end

IndexMap(s::IndsNetwork; map_dimension::Int64 = 1, kwargs...) = IndexMap(s, default_dimension_vertices(s; map_dimension); kwargs...)

function IndexMap(dimension_indices::Vector{Vector{V}}; is_complex = false) where {V<:Index}
  index_digit = Dictionary()
  index_dimension = Dictionary()
  for (dimension, indices) in enumerate(dimension_indices)
    digit_counter = 1
    for (i, ind) in enumerate(indices)
      set!(index_digit, ind, digit_counter)
      if is_complex
        digit_counter += isodd(i) ? 0 : 1
      else
        digit_counter += 1
      end
      set!(index_dimension, ind, dimension)
    end
  end
  return IndexMap(index_digit, index_dimension)
end

function Base.copy(imap::IndexMap)
  return IndexMap(copy(index_digit(imap)), copy(index_dimension(imap)))
end

dimension(imap::IndexMap) = maximum(collect(values(index_dimension(imap))))
dimension(imap::IndexMap, ind::Index) = index_dimension(imap)[ind]
dimensions(imap::IndexMap, inds::Vector{Index}) = dimension.(inds)
digit(imap::IndexMap, ind::Index) = index_digit(imap)[ind]
digits(imap::IndexMap, inds::Vector{Index}) = digit.(inds)
function index_value_to_scalar(imap::IndexMap, ind::Index, value::Int)
  out = (value) / (dim(ind)^digit(imap, ind))
  out = is_real(ind) ? out : 1.0*im*out
  return out
end
function index_values_to_scalars(imap::IndexMap, ind::Index)
  return [index_value_to_scalar(imap, ind, i) for i in 0:(dim(ind) - 1)]
end

function ITensors.inds(imap::IndexMap)
  @assert keys(index_dimension(imap)) == keys(index_digit(imap))
  return collect(keys(index_dimension(imap)))
end
function dimension_inds(imap::IndexMap, dimension::Int)
  return collect(
    filter(i -> index_dimension(imap)[i] == dimension, keys(index_dimension(imap)))
  )
end
function ind(imap::IndexMap, dimension::Int, digit::Int)
  return only(
    filter(
      i -> index_dimension(imap)[i] == dimension && index_digit(imap)[i] == digit,
      keys(index_dimension(imap)),
    ),
  )
end

function calculate_xyz(imap::IndexMap, ind_to_ind_value_map, dimensions::Vector{Int}; is_complex = false)
  out = Number[]
  for dimension in dimensions
    if !is_complex
      indices = dimension_inds(imap, dimension)
      push!(
        out,
        sum([index_value_to_scalar(imap, ind, ind_to_ind_value_map[ind]) for ind in indices]),
      )
    else
      indices = dimension_inds(imap, dimension)
      real_indices = filter(ind -> is_real(ind), indices)
      imag_indices = filter(ind -> !is_real(ind), indices)
      real = sum([index_value_to_scalar(imap, ind, ind_to_ind_value_map[ind]) for ind in real_indices])
      imag = sum([index_value_to_scalar(imap, ind, ind_to_ind_value_map[ind]) for ind in imag_indices])
      push!(out, real + 1.0*im*imag)
    end
  end
  return out
end

function calculate_xyz(imap::IndexMap, ind_to_ind_value_map; kwargs...)
  return calculate_xyz(imap, ind_to_ind_value_map, [i for i in 1:dimension(imap)]; kwargs...)
end
function calculate_x(imap::IndexMap, ind_to_ind_value_map, dimension::Int; kwargs...)
  return only(calculate_xyz(imap, ind_to_ind_value_map, [dimension]; kwargs...))
end
function calculate_x(imap::IndexMap, ind_to_ind_value_map; kwargs...)
  return calculate_x(imap, ind_to_ind_value_map, 1; kwargs...)
end

function set_ind_values(imap::IndexMap, sorted_inds::Vector, ind_to_ind_value_map::Dictionary, x::Number)
  ind_to_ind_value_map = copy(ind_to_ind_value_map)
  x_rn = copy(x)
  for ind in sorted_inds
    ind_val = dim(ind) - 1
    ind_set = false
    while (!ind_set)
      if x_rn >= abs(index_value_to_scalar(imap, ind, ind_val))
        set!(ind_to_ind_value_map, ind, ind_val)
        x_rn -= abs(index_value_to_scalar(imap, ind, ind_val))
        ind_set = true
      else
        ind_val = ind_val - 1
      end
    end
  end
  return ind_to_ind_value_map
end

function calculate_ind_values(
  imap::IndexMap, xs::Vector, dimensions::Vector{Int}; print_x=false, is_complex = false
)
  @assert length(xs) == length(dimensions)
  ind_to_ind_value_map = Dictionary()
  for (i, x) in enumerate(xs)
    dimension = dimensions[i]
    indices = dimension_inds(imap, dimension)
    if !is_complex
      sorted_inds = sort(indices; by=indices -> digit(imap, indices))
      ind_to_ind_value_map = set_ind_values(imap, sorted_inds, ind_to_ind_value_map, x)
    else
      real_indices = filter(ind -> is_real(ind), indices)
      sorted_real_inds = sort(real_indices; by=real_indices -> digit(imap, real_indices))
      ind_to_ind_value_map = set_ind_values(imap, sorted_real_inds, ind_to_ind_value_map, real(x))
      imag_indices = filter(ind -> !is_real(ind), indices)
      sorted_imag_inds = sort(imag_indices; by=imag_indices -> digit(imap, imag_indices))
      ind_to_ind_value_map = set_ind_values(imap, sorted_imag_inds, ind_to_ind_value_map,  imag(x))
    end

    if print_x && !is_complex
      x_bitstring = calculate_x(imap, ind_to_ind_value_map, dimension)
      println(
        "Dimension $dimension. Actual value of x is $x but bitstring rep. is $x_bitstring"
      )
    end
  end
  return ind_to_ind_value_map
end

function calculate_ind_values(imap::IndexMap, x::Number, dimension::Int; kwargs...)
  return calculate_ind_values(imap, [x], [dimension]; kwargs...)
end
function calculate_ind_values(imap::IndexMap, xs::Vector; kwargs...)
  return calculate_ind_values(imap, xs, [i for i in 1:length(xs)]; kwargs...)
end
function calculate_ind_values(imap::IndexMap, x::Number; kwargs...)
  return calculate_ind_values(imap, [x], [1]; kwargs...)
end

function grid_points(imap::IndexMap, N::Int, dimension::Int)
  dims = dim.(dimension_inds(imap, dimension))
  @assert all(y -> y == first(dims), dims)
  base = first(dims)
  L = length(dimension_inds(imap, dimension))
  a = round(base^L / N)
  grid_points = [i * (a / base^L) for i in 0:(N + 1)]
  return filter(x -> x <= 1, grid_points)
end

" Obtain all grid points of a given dimension"
function grid_points(imap::IndexMap, dimension::Int)
  dims = dim.(dimension_inds(imap, dimension))
  @assert all(y -> y == first(dims), dims)
  base = dims[dimension]
  L = length(dimension_inds(imap, dimension))
  return grid_points(imap, base^L, dimension)
end
