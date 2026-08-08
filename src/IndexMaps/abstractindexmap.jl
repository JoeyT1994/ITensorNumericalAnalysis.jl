using Base: Base
using Dictionaries: Dictionary, set!
using ITensors: ITensors, Index, dim
using Graphs

abstract type AbstractIndexMap{V} end

#These functions need to be defined on the concrete type for implementation

"""
  index_digit(imap::AbstractIndexMap)

Return the mapping from indices to digit number
"""
function index_digit end

"""
  index_dimension(imap::AbstractIndexMap)

Return the mapping from index to corresponding dimension
"""
function index_dimension end

#function Base.copy(imap::AbstractIndexMap) end

"""
  index_value_to_scalar(imap::AbstractIndexMap, ind::Index, value::Int)

Calculate the corresponding grid point given an Index
with a setting of value to its corresponding grid value

calculate_ind_values and index_value_to_scalar should match input/outputs
"""
function index_value_to_scalar end

#function ITensors.inds(imap::AbstractIndexMap) end

"""
  ind(imap::AbstractIndexMap, args...)

Return the matching index given args (e.g. dim, digit)

"""
function ind end

"""
  calculate_ind_values(
    imap::AbstractIndexMap, xs::Vector, dims::Vector{Int}; kwargs...
    )

Given a set of points xs with corresponding dimensions dims,
return a dictionary mapping each index value to a setting

calculate_ind_values and index_value_to_scalar should match input/outputs
"""
function calculate_ind_values end

"""
  grid_points(imap::AbstractIndexMap, N::Int, d::Int)

Convenience function, return N uniform grid points of the dimension d
"""
function grid_points end

"""
  rem_index(imap::AbstractIndexMap, ind::Index)

Remove an index from the index map
"""
function rem_index end

dimensions(imap::AbstractIndexMap) = Int64.(unique(collect(values(index_dimension(imap)))))
dimension(imap::AbstractIndexMap) = maximum(dimensions(imap))
dimension(imap::AbstractIndexMap, ind::Index) = index_dimension(imap)[ind]
dimensions(imap::AbstractIndexMap, inds::Vector{Index}) = dimension.(inds)
digit(imap::AbstractIndexMap, ind::Index) = index_digit(imap)[ind]
digits(imap::AbstractIndexMap, inds::Vector{Index}) = digit.(inds)
TensorNetworkQuantumSimulator.siteinds(imap::AbstractIndexMap, vertex) = siteinds(imap)[vertex]
Graphs.vertices(imap::AbstractIndexMap) = collect(keys(siteinds(imap)))

function index_values_to_scalars(imap::AbstractIndexMap, ind::Index)
    return [index_value_to_scalar(imap, ind, i) for i in 0:(dim(ind) - 1)]
end

function dimension_inds(imap::AbstractIndexMap, dims::Vector{<:Int})
    return collect(filter(i -> index_dimension(imap)[i] ∈ dims, keys(index_dimension(imap))))
end

function dimension_inds(imap::AbstractIndexMap, dim::Int)
    return dimension_inds(imap, [dim])
end

function reduced_indexmap(imap::AbstractIndexMap, dims::Vector{<:Int})
    imap_dim = copy(imap)
    for ind in setdiff(inds(imap), dimension_inds(imap, dims))
        imap_dim = rem_index(imap_dim, ind)
    end
    return imap_dim
end

function reduced_indexmap(imap::AbstractIndexMap, dim::Int)
    return reduced_indexmap(imap, [dim])
end

function calculate_p(imap::AbstractIndexMap, input::Vector{<:Pair})
    ndim = dimension(imap)
    out = zeros(NDTensors.scalartype(imap), ndim)
    @inbounds for (ind, value) in input
        d = dimension(imap, ind)
        out[d] += index_value_to_scalar(imap, ind, value - 1)
    end
    length(out) == 1 && return first(out)
    return out
end
function calculate_p(
    imap::AbstractIndexMap, ind_to_ind_value_map, dims::Vector{Int} = dimensions(imap)
)
    out = Vector{Number}(undef, length(dims))
    @inbounds for k in eachindex(dims)
        d = dims[k]
        indices = filter(i -> dimension(imap, i) == d, keys(ind_to_ind_value_map))
        out[k] = sum(
            [index_value_to_scalar(imap, ind, ind_to_ind_value_map[ind]) for ind in indices]
        )
    end
    length(out) == 1 && return first(out)
    return out
end

function calculate_p(imap::AbstractIndexMap, ind_to_ind_value_map, dim::Int)
    return calculate_p(imap, ind_to_ind_value_map, [dim])
end

function set_ind_values!(
        ind_to_ind_value_map::Dictionary, imap::AbstractIndexMap, sorted_inds::Vector, x::Number
    )
    x_rn = copy(x)
    for ind in sorted_inds
        ind_val = dim(ind) - 1
        ind_set = false
        while !ind_set
            if x_rn >= abs(index_value_to_scalar(imap, ind, ind_val))
                set!(ind_to_ind_value_map, ind, ind_val)
                x_rn -= abs(index_value_to_scalar(imap, ind, ind_val))
                ind_set = true
            else
                ind_val -= 1
            end
        end
    end
    return
end

function calculate_ind_values(
        imap::AbstractIndexMap, x::Number, dim::Int = first(dimensions(imap)); kwargs...
    )
    return calculate_ind_values(imap, [x], [dim]; kwargs...)
end
function calculate_ind_values(imap::AbstractIndexMap, xs::Vector; kwargs...)
    return calculate_ind_values(imap, xs, [i for i in 1:length(xs)]; kwargs...)
end

function grid_points(imap::AbstractIndexMap, d::Int)
    dims = dim.(dimension_inds(imap, d))
    @assert all(y -> y == first(dims), dims)
    base = first(dims)
    L = length(dimension_inds(imap, d))
    return grid_points(imap, base^L, d)
end

base(imap::AbstractIndexMap) = base(siteinds(imap))

function vertices_dimensions(imap::AbstractIndexMap, verts::Vector)
    return [dimension(imap, i) for i in siteinds(imap, verts)]
end

function vertices_digits(imap::AbstractIndexMap, verts::Vector)
    return [digit(imap, i) for i in siteinds(imap, verts)]
end

function vertex_dimension(imap::AbstractIndexMap, v)
    return dimension(imap, only(siteinds(imap, v)))
end

function vertex_dimensions(imap::AbstractIndexMap, v)
    return [dimension(imap, i) for i in siteinds(imap, v)]
end

function vertex_digits(imap::AbstractIndexMap, v)
    return [digit(imap, i) for i in siteinds(imap, v)]
end

function vertex_digit(imap::AbstractIndexMap, v)
    return digit(imap, only(siteinds(imap, v)))
end

function dimension_vertices(imap::AbstractIndexMap, dimension::Int)
    return filter(v -> dimension ∈ vertex_dimensions(imap, v), vertices(imap))
end

function dimension_vertices(imap::AbstractIndexMap, dims::Vector{Int})
    return filter(v -> vertex_dimension(imap, v) in dims, vertices(imap))
end

function vertex(imap::AbstractIndexMap, dimension::Int, digit::Int)
    index = ind(imap, dimension, digit)
    sinds = siteinds(imap)
    return only(filter(v -> index ∈ sinds[v], vertices(imap)))
end
