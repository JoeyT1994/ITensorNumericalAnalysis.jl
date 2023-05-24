using ITensors
using ITensorNetworks
using NamedGraphs
using EllipsisNotation
using Graphs
using LinearAlgebra

using ITensorNetworks: delta_network
using NamedGraphs: add_edges

include("../src/QTT_utils.jl")

function coeff(n, r, c)
  @assert r ≥ c
  # make this more efficient later
  multi = factorial(n) / (factorial(r - c) * factorial(c - 1) * factorial(n + 1 - r))

  return multi * 1 / sqrt(binomial(n, n + 1 - r)) * 1 / sqrt(binomial(n, c - 1))
end

function main()
  L = 10
  g = named_grid((L, 1))
  s = siteinds("S=1/2", g)

  # construct f(x) = ax^n
  n = 2 # degree
  a = 1.0

  #Define a map which determines a canonical ordering of the vertices of the network
  vertex_map = Dict(vertices(g) .=> [i for i in 1:L])

  ψ = delta_network(s; link_space=n + 1)
  for v in vertices(ψ)
    ψ[v] = ITensor(Float64, inds(ψ[v]))
    xi = 1.0 / 2^vertex_map[v]
    if (vertex_map[v] in [1]) # left edge
      vec = zeros((1, n + 1))
      vec[end] = 1
      ψ[v][1, :] = vec

      for i in 1:(n + 1)
        vec[i] = sqrt(binomial(n, n - (i - 1))) * xi^(n - (i - 1))
      end
      ψ[v][2, :] = vec
    elseif vertex_map[v] in [L] # right edge
      vec = zeros((1, n + 1))
      vec[1] = 1
      ψ[v][1, :] = vec

      for i in 1:(n + 1)
        vec[end - i + 1] = sqrt(binomial(n, n - (i - 1))) * xi^(n - (i - 1))
      end
      ψ[v][2, :] = vec

    else # bulk
      mat = Matrix{Float64}(I, n + 1, n + 1)
      ψ[v][1, :, :] = mat
      for i in 2:(n + 1)
        for j in 1:(i - 1)
          c = coeff(n, i, j)
          mat[i, j] = c * xi^(i - j)
        end
      end
      ψ[v][2, :, :] = mat
    end
  end
  ψ[first(vertices(ψ))] *= a

  valid_points = [i / 2^L for i in 0:(2^L - 1)]
  values = []
  for point in valid_points
    xis = calculate_xis(point, vertex_map; print_x=false)
    eval_point = calculate_x(xis, vertex_map)
    @assert eval_point ≈ point
    ψ12proj = get_bitstring_network(ψ, s, xis)
    push!(values, ITensors.contract(ψ12proj)[])
  end

  @show (a * valid_points .^ n) ≈ values
end

main()
