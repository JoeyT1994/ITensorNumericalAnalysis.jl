using ITensors: inds, dim
using ITensorNetworks: AbstractITensorNetwork
using NamedGraphs: vertices

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