using Test
using ITensorNumericalAnalysis
using ITensors: inds

using ITensorNetworks: maxlinkdim
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_grid

using ITensorNumericalAnalysis: vertex_digits, vertex_dimensions, indexmap

function build_laughlin_term(s::IndsNetworkMap, v::Int64, dim1::Int, dim2::Int; k::Float64, cutoff::Float64 = 1e-16, maxdim)
  psi_x_powers = vcat([poly_itn(s, [i == j ? 1.0 : 0.0 for i in 1:j]; k, dimension = dim1) for j in 1:(v+1)])
  psi_y_powers = vcat([poly_itn(s, [i == j ? 1.0 : 0.0 for i in 1:j]; k = -1.0*k, dimension = dim2) for j in 1:(v+1)])

  terms = [const_itn(s; c = binomial(v, k))*psi_x_powers[v-k + 1]*psi_y_powers[k + 1] for k in 0:v]
  psi = first(terms)
  for term in terms[2:(v+1)]
    psi = truncate(psi; cutoff, maxdim)
    psi = psi + term
  end
  return psi
end

function build_laughlin(s::IndsNetworkMap, v::Int64, N::Int64; k::Float64, cutoff::Float64 = 1e-16, maxdim)
  fx = const_itn(s)
  for i in 1:N
    for j in i+1:N
      term = build_laughlin_term(s, 1, i, j; k, cutoff, maxdim)
      term = truncate(term; cutoff, maxdim)
      fx = fx * term
      fx = truncate(fx; cutoff, maxdim)
    end
  end
  for i in 2:v
    fx = fx * fx
    fx = truncate(fx; cutoff, maxdim)
  end
  return fx
end

function calculate_laughlin(zs, v::Int; k::Float64)
  N = length(zs)
  out = 1
  for i in 1:N
    for j in i+1:N
      out *= (k*zs[i] - k*zs[j])^v
    end
  end
  return out
end