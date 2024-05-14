using Base: Base
using Dictionaries: Dictionary, set!
using ITensors: ITensors, Index, dim, hastags
using ITensorNetworks: IndsNetwork, vertex_data
using NamedGraphs: vertextype

struct IndexMap{VB,VD,VC}
  index_digit::VB
  index_dimension::VD
  index_imaginary::VC
end

index_digit(imap::IndexMap) = imap.index_digit
index_dimension(imap::IndexMap) = imap.index_dimension
index_imaginary(imap::IndexMap) = imap.index_imaginary

function default_digit_map(indices::Vector{Index}; map_dimension::Int=1)
  return Dictionary(
    indices, [ceil(Int, i / map_dimension) for (i, ind) in enumerate(indices)]
  )
end
function default_dimension_map(indices::Vector{Index}; map_dimension::Int)
  return Dictionary(indices, [(i % map_dimension) + 1 for (i, ind) in enumerate(indices)])
end
function default_imaginary_map(indices::Vector{Index})
  return Dictionary(indices, [false for _ in enumerate(indices)])
end
function default_dimension_vertices(
  g::AbstractGraph; map_dimension::Int64=1, is_complex::Bool=false
)
  verts = collect(vertices(g))
  L = length(verts)
  real_dimension_vertices = Vector{vertextype(g)}[
    verts[i:map_dimension:L] for i in 1:map_dimension
  ]
  imag_dimension_vertices = if is_complex
    Vector{vertextype(g)}[verts[i:map_dimension:L] for i in 1:map_dimension]
  else
    Vector{vertextype(g)}[[]]
  end
  return real_dimension_vertices, imag_dimension_vertices
end

function IndexMap(
  s::IndsNetwork,
  real_dimension_vertices::Vector{Vector{V}},
  imag_dimension_vertices::Vector{Vector{V}}=[[]];
  kwargs...,
) where {V}
  real_dimension_indices = Vector{Index}[
    [first(inds(s, v)) for v in vertices] for vertices in real_dimension_vertices
  ]
  imaginary_dimension_indices = Vector{Index}[
    [last(inds(s, v)) for v in vertices] for vertices in imag_dimension_vertices
  ]
  return IndexMap(real_dimension_indices, imaginary_dimension_indices; kwargs...)
end

function IndexMap(s::IndsNetwork; map_dimension::Int64=1, is_complex=false, kwargs...)
  return IndexMap(s, default_dimension_vertices(s; map_dimension, is_complex)...; kwargs...)
end

function IndexMap(
  real_dimension_indices::Vector{Vector{I}},
  imaginary_dimension_indices::Vector{Vector{I}}=Vector{Index}[[]],
) where {I<:Index}
  index_digit = Dictionary()
  index_dimension = Dictionary()
  index_imaginary = Dictionary()
  for (dimension, indices) in enumerate(real_dimension_indices)
    for (digit, ind) in enumerate(indices)
      set!(index_digit, ind, digit)
      set!(index_dimension, ind, dimension)
      set!(index_imaginary, ind, false)
    end
  end

  for (dimension, indices) in enumerate(imaginary_dimension_indices)
    for (digit, ind) in enumerate(indices)
      set!(index_digit, ind, digit)
      set!(index_dimension, ind, dimension)
      set!(index_imaginary, ind, true)
    end
  end

  return IndexMap(index_digit, index_dimension, index_imaginary)
end

function Base.copy(imap::IndexMap)
  return IndexMap(
    copy(index_digit(imap)), copy(index_dimension(imap)), copy(index_imaginary(imap))
  )
end

dimension(imap::IndexMap) = maximum(collect(values(index_dimension(imap))))
dimension(imap::IndexMap, ind::Index) = index_dimension(imap)[ind]
dimensions(imap::IndexMap, inds::Vector{Index}) = dimension.(inds)
digit(imap::IndexMap, ind::Index) = index_digit(imap)[ind]
digits(imap::IndexMap, inds::Vector{Index}) = digit.(inds)
is_imaginary(imap::IndexMap, ind::Index) = index_imaginary(imap)[ind]
is_real(imap::IndexMap, ind::Index) = !is_imaginary(imap, ind)
is_real(imap::IndexMap) = all(i -> is_real(imap, i), keys(index_imaginary(imap)))
is_complex(imap::IndexMap) = !is_real(imap)
function index_value_to_scalar(imap::IndexMap, ind::Index, value::Int)
  out = (value) / (dim(ind)^digit(imap, ind))
  out = is_real(imap, ind) ? out : 1.0 * im * out
  return out
end
function index_values_to_scalars(imap::IndexMap, ind::Index)
  return [index_value_to_scalar(imap, ind, i) for i in 0:(dim(ind) - 1)]
end

function ITensors.inds(imap::IndexMap)
  @assert keys(index_dimension(imap)) == keys(index_digit(imap))
  return collect(keys(index_dimension(imap)))
end
function dimension_inds(imap::IndexMap, dim::Int)
  return collect(filter(i -> dimension(imap, i) == dim, keys(index_dimension(imap))))
end
function real_inds(imap::IndexMap, dimensions::Vector{Int64}=[i for i in 1:dimension(imap)])
  return collect(
    filter(
      i -> is_real(imap, i) && dimension(imap, i) ∈ dimensions, keys(index_dimension(imap))
    ),
  )
end
function imaginary_inds(
  imap::IndexMap, dimensions::Vector{Int64}=[i for i in 1:dimension(imap)]
)
  return collect(
    filter(
      i -> !is_real(imap, i) && dimension(imap, i) ∈ dimensions, keys(index_dimension(imap))
    ),
  )
end
function ind(imap::IndexMap, dim::Int, dig::Int, is_re::Bool=true)
  return only(
    filter(
      i -> dimension(imap, i) == dim && digit(imap, i) == dig && is_real(imap, i) == is_re,
      keys(index_dimension(imap)),
    ),
  )
end

function calculate_xyz(imap::IndexMap, ind_to_ind_value_map, dimensions::Vector{Int})
  out = Number[]
  for dimension in dimensions
    real_indices, imag_indices = real_inds(imap, [dimension]),
    imaginary_inds(imap, [dimension])
    real = sum([
      index_value_to_scalar(imap, ind, ind_to_ind_value_map[ind]) for ind in real_indices
    ])
    imag = if isempty(imag_indices)
      0.0
    else
      sum([
      index_value_to_scalar(imap, ind, ind_to_ind_value_map[ind]) for ind in imag_indices
    ])
    end
    push!(out, real + imag)
  end
  return out
end

function calculate_xyz(imap::IndexMap, ind_to_ind_value_map; kwargs...)
  return calculate_xyz(
    imap, ind_to_ind_value_map, [i for i in 1:dimension(imap)]; kwargs...
  )
end
function calculate_x(imap::IndexMap, ind_to_ind_value_map, dimension::Int; kwargs...)
  return only(calculate_xyz(imap, ind_to_ind_value_map, [dimension]; kwargs...))
end
function calculate_x(imap::IndexMap, ind_to_ind_value_map; kwargs...)
  return calculate_x(imap, ind_to_ind_value_map, 1; kwargs...)
end

function set_ind_values(
  imap::IndexMap, sorted_inds::Vector, ind_to_ind_value_map::Dictionary, x::Number
)
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
  imap::IndexMap, xs::Vector, dimensions::Vector{Int}; print_x=false
)
  @assert length(xs) == length(dimensions)
  ind_to_ind_value_map = Dictionary()
  for (i, x) in enumerate(xs)
    dimension = dimensions[i]
    real_indices = real_inds(imap, [dimension])
    imag_indices = imaginary_inds(imap, [dimension])

    sorted_real_inds = sort(real_indices; by=real_indices -> digit(imap, real_indices))
    ind_to_ind_value_map = set_ind_values(
      imap, sorted_real_inds, ind_to_ind_value_map, real(x)
    )
    if !isempty(imag_indices)
      sorted_imag_inds = sort(imag_indices; by=imag_indices -> digit(imap, imag_indices))
      ind_to_ind_value_map = set_ind_values(
        imap, sorted_imag_inds, ind_to_ind_value_map, imag(x)
      )
    end
    if print_x
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
