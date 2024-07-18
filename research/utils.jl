using ITensors: inds, dim
using ITensorNetworks: AbstractITensorNetwork
using NamedGraphs: vertices
using LinearAlgebra: dot

function calc_error(exact_vals::Vector, approx_vals::Vector)
  @assert length(exact_vals) == length(approx_vals)

  m = minimum(exact_vals) + 1
  exact_vals = copy(exact_vals .+ m)
  approx_vals = copy(approx_vals .+ m)
  eps = 0
  for (i, e) in enumerate(exact_vals)
    eps += abs((e - approx_vals[i]) / e)
  end
  return eps / length(exact_vals)
end

function no_elements(tn::AbstractITensorNetwork)
    no_elements = 0
    for v in vertices(tn)
        dims = dim.(inds(tn[v]))
        no_elements += prod(dims)
    end
    return no_elements
end


function calc_error_V2(exact_vals::Vector, approx_vals::Vector)
  @assert length(exact_vals) == length(approx_vals)

 return dot(exact_vals, approx_vals) / sqrt(dot(exact_vals, exact_vals) * dot(approx_vals, approx_vals))
end