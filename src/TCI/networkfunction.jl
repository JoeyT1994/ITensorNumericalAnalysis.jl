using ITensorNetworks: ITensorNetworks, position

struct NetworkFunction{PivotType,ElType}
  f::Function
  fcache::Dict{PivotType,ElType}
  use_caching::Bool
end

function NetworkFunction(f, initial_pivot; use_caching=true)
  initial_val = f(initial_pivot)
  fcache = Dict(initial_pivot => initial_val)
  return NetworkFunction{typeof(initial_pivot),typeof(initial_val)}(f, fcache, use_caching)
end

cache(nf::NetworkFunction) = nf.fcache
f(nf::NetworkFunction) = nf.f
use_caching(nf::NetworkFunction) = nf.use_caching
Base.eltype(::NetworkFunction{P,E}) where {P,E} = E

#ITensorNetworks.position(nf::NetworkFunction, state, region) = nf

function (nf::NetworkFunction)(arg)
  if use_caching(nf)
    return get!(cache(nf), arg) do
      f(nf)(arg)
    end
  end
  return f(nf)(arg)
end
