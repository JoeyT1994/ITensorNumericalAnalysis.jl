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
  imap::RealIndexMap, N::Int, d::Int; span::AbstractVector{<:Number}=[0,1], exact_grid::Bool=true, enforced=[]
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
  imap::RealIndexMap, d::Int; kwargs...
)
  dims = dim.(dimension_inds(imap, d))
  @assert all(y -> y == first(dims), dims)
  base = dims[d]
  L = length(dimension_inds(imap, d))
  return grid_points(imap, base^L, d; kwargs...)
end



""" 
  Picks a random grid point from `imap` given a dimension
"""
function rand_p(rng::AbstractRNG, imap::RealIndexMap, d::Integer)
  dims = dim.(dimension_inds(imap, d))
  @assert all(y -> y == first(dims), dims)
  base = dims[d]
  L = length(dimension_inds(imap, d))
  return rand(rng, 0:(base^L - 1)) / base^L
end