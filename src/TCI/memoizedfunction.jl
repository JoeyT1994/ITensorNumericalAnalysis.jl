"""
    MemoizedFunction{F, PivotType, ElType} <: Function

Wraps a function `f` with a dictionary-based cache. 

# Fields
- `f::F`: The underlying function to be evaluated.
- `cache::Dict{PivotType, ElType}`: A dictionary mapping function arguments (pivots) to their computed values.
- `use_caching::Bool`: A flag to enable or disable the caching mechanism. If `false`, `f` is always evaluated directly.
"""
struct MemoizedFunction{F,PivotType,ElType} <: Function
    f::F
    cache::Dict{PivotType,ElType}
    use_caching::Bool
end
function MemoizedFunction(f, initial_pivot; use_caching = true)
    key = canonicalize(initial_pivot)
    initial_val = f(key)
    ElT = typeof(initial_val)
    PivT = typeof(key)
    cache = Dict{PivT,ElT}(key => initial_val)
    return MemoizedFunction{typeof(f),PivT,ElT}(f, cache, use_caching)
end

cache(nf::MemoizedFunction) = nf.cache
pivots(nf::MemoizedFunction) = collect(keys(cache(nf)))
f(nf::MemoizedFunction) = nf.f
use_caching(nf::MemoizedFunction) = nf.use_caching
Base.eltype(::MemoizedFunction{F,P,E}) where {F,P,E} = E
canonicalize(arg::AbstractVector{<:Pair}) = sort(arg; by = p -> id(first(p)))

function (nf::MemoizedFunction)(arg)
    key = canonicalize(arg)
    use_caching(nf) || return f(nf)(key)
    return get!(() -> f(nf)(key), cache(nf), key)
end