using Base: Base
using Dictionaries: Dictionary, set!
using ITensors: ITensors, Index, dim, hastags
using ITensorNetworks: IndsNetwork, vertex_data

struct ComplexIndexMap{VB,VD,VR} <: AbstractIndexMap{VB,VD}
  index_digit::VB
  index_dimension::VD
  index_real::VR
end

index_digit(imap::ComplexIndexMap) = imap.index_digit
index_dimension(imap::ComplexIndexMap) = imap.index_dimension
index_real(imap::ComplexIndexMap) = imap.index_real
is_real(imap::ComplexIndexMap, ind::Index) = index_real(imap)[ind]
real_indices(imap::ComplexIndexMap) = filter(i -> is_real(imap, i), inds(imap))
imaginary_indices(imap::ComplexIndexMap) = filter(i -> !is_real(imap, i), inds(imap))
function real_indices(imap::ComplexIndexMap, dim::Int64)
  return filter(i -> is_real(imap, i) && dimension(imap, i) == dim, inds(imap))
end
function imaginary_indices(imap::ComplexIndexMap, dim::Int64)
  return filter(i -> !is_real(imap, i) && dimension(imap, i) == dim, inds(imap))
end
scalartype(imap::ComplexIndexMap) = ComplexF64

function index_value_to_scalar(imap::ComplexIndexMap, ind::Index, value::Int)
  return if is_real(imap, ind)
    (value) / (dim(ind)^digit(imap, ind))
  else
    im * (value) / (dim(ind)^digit(imap, ind))
  end
end
function Base.copy(imap::ComplexIndexMap)
  return IndexMap(
    copy(index_digit(imap)), copy(index_dimension(imap)), copy(index_real(imap))
  )
end
function ITensors.inds(imap::ComplexIndexMap)
  @assert keys(index_dimension(imap)) == keys(index_digit(imap)) == keys(index_real(imap))
  return collect(keys(index_dimension(imap)))
end
function ind(imap::ComplexIndexMap, dim::Int, digit::Int, real_ind::Bool=true)
  return only(
    filter(
      i ->
        index_dimension(imap)[i] == dim &&
          index_digit(imap)[i] == digit &&
          (real_ind == is_real(imap, i)),
      inds(imap),
    ),
  )
end

"""
Indices that reflect real valued digits should have the "Real" tag in IndsNetwork,
whilst imaginary valued digits should have the "Imag" tag. The complex_continuous_siteinds(...)
constructor will do this by default
"""
function ComplexIndexMap(
  s::IndsNetwork,
  real_dimension_vertices::Vector{Vector{V}}=default_dimension_vertices(s),
  imag_dimension_vertices::Vector{Vector{V}}=default_dimension_vertices(s),
) where {V}
  real_dimension_indices = Vector{Index}[
    filter(i -> hastags(i, "Real"), inds(s, vertices)) for
    vertices in real_dimension_vertices
  ]
  imag_dimension_indices = Vector{Index}[
    filter(i -> hastags(i, "Imag"), inds(s, vertices)) for
    vertices in imag_dimension_vertices
  ]
  return ComplexIndexMap(real_dimension_indices, imag_dimension_indices)
end

function ComplexIndexMap(
  real_dimension_indices::Vector{Vector{I}}, imag_dimension_indices::Vector{Vector{I}}
) where {I<:Index}
  index_digit = Dictionary()
  index_dimension = Dictionary()
  index_real = Dictionary()

  for (d, real_indices) in enumerate(real_dimension_indices)
    for (bit, real_ind) in enumerate(real_indices)
      set!(index_digit, real_ind, bit)
      set!(index_dimension, real_ind, d)
      set!(index_real, real_ind, true)
    end
  end

  for (d, imag_indices) in enumerate(imag_dimension_indices)
    for (bit, imag_ind) in enumerate(imag_indices)
      set!(index_digit, imag_ind, bit)
      set!(index_dimension, imag_ind, d)
      set!(index_real, imag_ind, false)
    end
  end
  return ComplexIndexMap(index_digit, index_dimension, index_real)
end

function calculate_ind_values(imap::ComplexIndexMap, xs::Vector, dims::Vector{Int})
  @assert length(xs) == length(dims)
  ind_to_ind_value_map = Dictionary()
  for (i, x) in enumerate(real.(xs))
    real_inds = real_indices(imap, dims[i])
    sorted_real_inds = sort(real_inds; by=real_index -> digit(imap, real_index))
    set_ind_values!(ind_to_ind_value_map, imap, sorted_real_inds, x)
  end

  for (i, x) in enumerate(imag.(xs))
    imag_inds = imaginary_indices(imap, dims[i])
    sorted_imag_inds = sort(imag_inds; by=imag_indices -> digit(imap, imag_indices))
    set_ind_values!(ind_to_ind_value_map, imap, sorted_imag_inds, x)
  end

  return ind_to_ind_value_map
end

function grid_points(imap::ComplexIndexMap, N::Int, d::Int)
  dims = dim.(dimension_inds(imap, d))
  @assert all(y -> y == first(dims), dims)
  base = first(dims)
  Lre, Lim = length(real_indices(imap, d)), length(imag_indices(imap, d))
  are, aim = round(base^Lre / N), round(base^Lim / N)
  grid_points = [
    i * (are / base^Lre) + im * j * (aim / base^Lim) for i in 0:(N + 1) for j in 0:(N + 1)
  ]
  return filter(x -> real(x) < 1 && imag(x) < 1, grid_points)
end
