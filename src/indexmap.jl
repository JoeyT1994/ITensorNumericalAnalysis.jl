using Base: Base
using Dictionaries: Dictionary, set!
using ITensors: ITensors, Index, dim
using ITensorNetworks: IndsNetwork, vertex_data
using Random: Random, rand, AbstractRNG

struct IndexMap{VB,VD}
  index_digit::VB
  index_dimension::VD
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
  for (d, indices) in enumerate(dimension_indices)
    for (bit, ind) in enumerate(indices)
      set!(index_digit, ind, bit)
      set!(index_dimension, ind, d)
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
  return (value) / (dim(ind)^digit(imap, ind))
end
function index_values_to_scalars(imap::IndexMap, ind::Index)
  return [index_value_to_scalar(imap, ind, i) for i in 0:(dim(ind) - 1)]
end

function ITensors.inds(imap::IndexMap)
  @assert keys(index_dimension(imap)) == keys(index_digit(imap))
  return collect(keys(index_dimension(imap)))
end
function dimension_inds(imap::IndexMap, dim::Int)
  return collect(filter(i -> index_dimension(imap)[i] == dim, keys(index_dimension(imap))))
end
function dimension_indices(imap::IndexMap, dim::Int)
  return filter(ind -> dimension(imap, ind) == dim, inds(imap))
end
function ind(imap::IndexMap, dim::Int, digit::Int)
  return only(
    filter(
      i -> index_dimension(imap)[i] == dim && index_digit(imap)[i] == digit,
      keys(index_dimension(imap)),
    ),
  )
end

function calculate_p(
  imap::IndexMap, ind_to_ind_value_map, dims::Vector{Int}=[i for i in 1:dimension(imap)]
)
  out = Number[]
  for d in dims
    indices = dimension_inds(imap, d)
    push!(
      out,
      sum([index_value_to_scalar(imap, ind, ind_to_ind_value_map[ind]) for ind in indices]),
    )
  end
  return out
end

function calculate_p(imap::IndexMap, ind_to_ind_value_map, dim::Int)
  return calculate_p(imap, ind_to_ind_value_map, [dim])
end

function calculate_ind_values(imap::IndexMap, xs::Vector, dims::Vector{Int}; print_x=false)
  @assert length(xs) == length(dims)
  ind_to_ind_value_map = Dictionary()
  for (i, x) in enumerate(xs)
    d = dims[i]
    x_rn = x
    indices = dimension_inds(imap, d)
    sorted_inds = sort(indices; by=indices -> digit(imap, indices))
    for ind in sorted_inds
      ind_val = dim(ind) - 1
      ind_set = false
      while (!ind_set)
        if x_rn >= index_value_to_scalar(imap, ind, ind_val)
          set!(ind_to_ind_value_map, ind, ind_val)
          x_rn -= index_value_to_scalar(imap, ind, ind_val)
          ind_set = true
        else
          ind_val = ind_val - 1
        end
      end
    end

    if print_x
      x_bitstring = only(calculate_p(imap, ind_to_ind_value_map, d))
      println(
        "Dimension $dimension. Actual value of x is $x but bitstring rep. is $x_bitstring"
      )
    end
  end
  return ind_to_ind_value_map
end

function calculate_ind_values(imap::IndexMap, x::Number, dim::Int; kwargs...)
  return calculate_ind_values(imap, [x], [dim]; kwargs...)
end
function calculate_ind_values(imap::IndexMap, xs::Vector; kwargs...)
  return calculate_ind_values(imap, xs, [i for i in 1:length(xs)]; kwargs...)
end
function calculate_ind_values(imap::IndexMap, x::Number; kwargs...)
  return calculate_ind_values(imap, [x], [1]; kwargs...)
end

""" 
    grid_points(imap, N, d, range; exact_grid=true)

  Gives `N` grid points from a given dimension of `imap` within a specified range.

  # Arguments
  - `imap::IndexMap`: An IndexMap specifying the structure of the TN being used
  - `N::Int`: The number of grid points requested.
  - `d::Int` The index of the dimension of `imap` requested
  - `span::AbstractVector{<:Number}`: A two element number vector [a,b] with 0≤a<b≤1. The right endpoint of this span is not included as a gridpoint in the output.
  - `exact_grid::Bool`: Flag specifying whether the function should give exact grid points 
  (note: using `exact_grid=true` may cause less than `N` grid points to be returned)
  - `enforced::AbstractVector{<:Number}`: A list of points that we want to enforce to show up in the grid.
"""
function grid_points(
  imap::IndexMap, N::Int, d::Int, span=[0, 1]; exact_grid::Bool=true, enforced=[]
)
  if length(span) != 2 || span[1] >= span[2] || span[1] < 0 || span[2] > 1
    throw(
      "expected a two-element vector [a,b] with 0≤a<b≤1 as input span, instead found $span"
    )
  end

  dims = dim.(dimension_inds(imap, d))
  @assert all(y -> y == first(dims), dims)
  base = first(dims)
  L = length(dimension_inds(imap, d))

  if !exact_grid
    grid_points = range(span[1], span[2] - (span[2] - span[1]) / N; length=N)
  else #exact_grid = true
    if span[1] == 0 && span[2] == 1
      inv_step = min(Int(floor(log(base, N))), L)
      if !(log(base, N) ≈ inv_step)
        @warn "rounding $N down to $(Int(base^inv_step)) exact gridpoints!"
      end
      # now 1/inv_step should provide the step
      grid_points = collect(span[1]:(1 / base^inv_step):span[2])[1:(end - 1)]

    else

      #get a count of the number of gridpoints within the span
      points_in_span = floor(span[end] * base^L) - ceil(span[1] * base^L) + 1
      #TODO: figure out a way to calculate this without exponentiating to the power of L

      #exclude the endpoint in the count
      if floor(span[end] * base^L) == span[end] * base^L
        points_in_span = points_in_span - 1
      end

      if points_in_span <= 0
        @warn "No exact gridpoints found in this span!"
        grid_points = []
      else
        oldN = N
        stepsize = ceil(points_in_span / N)
        N = Int(floor(points_in_span / stepsize)) + 1
        startval = ceil(span[1] * base^L) / base^L #startval is the smallest grid point ≥ span[1]

        if startval + (N - 1) * stepsize / base^L >= span[2]
          N = N - 1
        end

        if N < oldN
          @warn "rounding $oldN down to $N exact gridpoints to maintain equal spacing."
        end

        grid_points = [startval + i * (stepsize / base^L) for i in 0:(N - 1)]
      end
    end
  end

  #add enforced points in naively
  for point in enforced
    if point >= span[2] || point < span[1]
      @warn "enforced point $point outside of span $span, not including it"
      continue
    end
    nearest_idx = searchsortedfirst(grid_points, point)
    if nearest_idx > length(grid_points) || grid_points[nearest_idx] != point
      insert!(grid_points, nearest_idx, point)
    end
  end

  return grid_points
end

function grid_points(
  imap::IndexMap, d::Int, span::AbstractVector{<:Number}=[0, 1]; kwargs...
)
  dims = dim.(dimension_inds(imap, d))
  @assert all(y -> y == first(dims), dims)
  base = dims[d]
  L = length(dimension_inds(imap, d))
  return grid_points(imap, base^L, d, span; kwargs...)
end

""" 
  Picks a random grid point from `imap` given a dimension
"""
function rand_p(rng::AbstractRNG, imap::IndexMap, d::Integer)
  dims = dim.(dimension_inds(imap, d))
  @assert all(y -> y == first(dims), dims)
  base = dims[d]
  L = length(dimension_inds(imap, d))
  return rand(rng, 0:(base^L - 1)) / base^L
end

function rand_p(rng::AbstractRNG, imap::IndexMap)
  return [rand_p(rng, imap, i) for i in 1:dimension(imap)]
end

rand_p(imap::IndexMap, args...) = rand_p(Random.default_rng(), imap, args...)
