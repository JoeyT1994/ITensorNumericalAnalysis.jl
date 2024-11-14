using Base: Base
using Dictionaries: Dictionaries, Dictionary, set!
using ITensors: ITensors, Index, dim
using ITensorNetworks: IndsNetwork, vertex_data
using Random: AbstractRNG

struct RealIndexMap{VB,VD} <: AbstractIndexMap{VB,VD}
  index_digit::VB
  index_dimension::VD
end

index_digit(imap::RealIndexMap) = imap.index_digit
index_dimension(imap::RealIndexMap) = imap.index_dimension
function index_value_to_scalar(imap::RealIndexMap, ind::Index, value::Int)
  return (value) * (float(dim(ind))^-digit(imap, ind))
end
function Base.copy(imap::RealIndexMap)
  return RealIndexMap(copy(index_digit(imap)), copy(index_dimension(imap)))
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

function rem_index(imap::RealIndexMap, ind::Index)
  imap_r = copy(imap)
  delete!(index_digit(imap_r), ind)
  delete!(index_dimension(imap_r), ind)
  return imap_r
end

function RealIndexMap(
  s::IndsNetwork, dimension_vertices::Vector{Vector{V}}=default_dimension_vertices(s)
) where {V}
  dimension_indices = Vector{Index}[
    !isempty(vertices) ? inds(s, vertices) : Index[] for vertices in dimension_vertices
  ]
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

function Dictionaries.merge(imap1::RealIndexMap, imap2::RealIndexMap)
  return RealIndexMap(
    merge(index_digit(imap1), index_digit(imap2)),
    merge(index_dimension(imap1), index_dimension(imap2)),
  )
end

function calculate_ind_values(imap::RealIndexMap, xs::Vector, dims::Vector{Int})
  @assert length(xs) == length(dims)
  ind_to_ind_value_map = Dictionary()
  for (d, x) in zip(dims, xs)
    indices = dimension_inds(imap, d)
    sorted_inds = sort(indices; by=indices -> digit(imap, indices))
    set_ind_values!(ind_to_ind_value_map, imap, sorted_inds, x)
  end
  return ind_to_ind_value_map
end

""" 
    grid_points(imap, N, d, span; exact_grid=true, enforced=[])

  Gives `N` grid points from a given dimension of `imap` within a specified range.

  # Arguments
  - `imap::RealIndexMap`: An IndexMap specifying the structure of the TN being used
  - `N::Int`: The number of grid points requested.
  - `d::Int` The index of the dimension of `imap` requested
  - `span::AbstractVector{<:Number}`: A two element number vector [a,b] with 0≤a<b≤1. The right endpoint of this span is not included as a gridpoint in the output.
  - `exact_grid::Bool`: Flag specifying whether the function should give exact grid points 
  (note: using `exact_grid=true` may cause less than `N` grid points to be returned)
  - `enforced::AbstractVector{<:Number}`: A list of points that we want to enforce to show up in the grid.
"""
function grid_points(
  imap::RealIndexMap,
  N::Int,
  d::Int;
  span::AbstractVector{<:Number}=[0, 1],
  exact_grid::Bool=true,
  enforced=[],
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

  # generate grid_points within the span (exclusive of right endpoint)
  grid_points = collect(range(span[1], span[2] - (span[2] - span[1]) / N; length=N))
  if exact_grid
    for (i, point) in enumerate(grid_points)
      grid_points[i] = round_to_nearest_exact_point(point, L)
    end
    # define lambda function
    # round_near = point -> round_to_nearest_exact_point(point, L)
    # grid_points = map(round_near, grid_points)
    grid_points = unique(grid_points)
  end

  # add enforced points
  if !isempty(enforced)
    grid_points = sort(unique(vcat(grid_points, enforced_points)))
  end

  return grid_points
end

function round_to_nearest_exact_point(point::Number, L::Int)
  if point < 0 || point >= 1
    throw("Input point must be between 0 and 1")
  end
  return round(point * 2.0^L) / 2.0^L
end

function grid_points(imap::RealIndexMap, d::Int; kwargs...)
  dims = dim.(dimension_inds(imap, d))
  @assert all(y -> y == first(dims), dims)
  base = dims[d]
  L = length(dimension_inds(imap, d))
  return grid_points(imap, base^L, d; kwargs...)
end

grid_points(imap::RealIndexMap; kwargs...) = grid_points(imap, 1; kwargs...)

#multi-dimensional grid_points
function grid_points(imap::RealIndexMap, Ns::Vector{Int}, dims::Vector{Int}; kwargs...)
  if length(Ns) != length(dims)
    throw("length of Ns and dims do not match!")
  end
  coords = [grid_points(imap, pair[1], pair[2]; kwargs...) for pair in zip(Ns, dims)]
  gp = Base.Iterators.product(coords...)
  return [collect(point) for point in gp]
end

function grid_points(imap::RealIndexMap, dims::Vector{Int}; kwargs...)
  coords = [grid_points(imap, d; kwargs...) for d in dims]
  gp = Base.Iterators.product(coords...)
  return [collect(point) for point in gp]
end


""" 
  Picks a random grid point from `imap` given a dimension
"""
function rand_p(rng::AbstractRNG, imap::RealIndexMap, d::Integer)
  dims = dim.(dimension_inds(imap, d))
  @assert all(y -> y == first(dims), dims)
  base = dims[d]
  L = length(dimension_inds(imap, d))
  return rand(rng, 0:(big(base)^L - 1)) / big(base)^L
end
