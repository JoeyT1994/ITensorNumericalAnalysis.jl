using Random: AbstractRNG, default_rng

abstract type AbstractPivotSearch end


isexhaustive(::AbstractPivotSearch) = false

"""
    FullPivot

Pivot search strategy that exhaustively scans the entire residual matrix
to find the element with the largest absolute value.
"""
struct FullPivot <: AbstractPivotSearch end

isexhaustive(::FullPivot) = true

"""
    RookPivot

Pivot search strategy that starts from a random row and alternates between taking the
largest element of the current row and of the resulting column, up to `maxsteps` times
or until neither moves. Costs O(maxsteps * (m + n)) rather than the O(m * n) of
[`FullPivot`](@ref), at the price of returning a pivot that is maximal only within its
own row and column.

# Fields
- `maxsteps::Integer`: Maximum number of alternating row/column searches before termination.
- `rng::AbstractRNG`: Random number generator used to pick the starting row.
"""
@kwdef struct RookPivot{I<:Integer,R<:AbstractRNG} <: AbstractPivotSearch
    maxsteps::I = 2
    rng::R = default_rng()
end

function searchpivot(M::AbstractMatrix, ::FullPivot)
    return findmax(abs, M)
end
function searchpivot(M::AbstractMatrix, pivotsearch::RookPivot)
    i = rand(pivotsearch.rng, axes(M, 1))
    j = argmax(j -> abs(M[i, j]), axes(M, 2))
    @inbounds for _ in Base.OneTo(pivotsearch.maxsteps)
        inew = argmax(i -> abs(M[i, j]), axes(M, 1))
        inew == i && break
        i = inew
        jnew = argmax(j -> abs(M[i, j]), axes(M, 2))
        jnew == j && break
        j = jnew
    end
    idx = CartesianIndex(i, j)
    return abs(M[idx]), idx
end

"""
Build a vector of pivot search strategies by sweeping one or more fields, for
benchmarking. Every swept field must have the same length.
"""
function pivotsearch_grid(::Type{T}; kwargs...) where {T<:AbstractPivotSearch}
    ks = keys(kwargs)
    vs = values(values(kwargs))
    allequal(length.(vs)) ||
        throw(ArgumentError("all swept fields must have the same length"))
    return [T(; (k => v[n] for (k, v) in zip(ks, vs))...) for n in eachindex(first(vs))]
end