using ITensors
using ITensorNetworks
using NamedGraphs
using EllipsisNotation
using Graphs
using LinearAlgebra

using ITensorNetworks: delta_network
using NamedGraphs: add_edges

include("../src/QTT_utils.jl")

function main()
  L = 10
  g = named_grid((L, 1))
  s = siteinds("S=1/2", g)

  # polynomial degree, i.e. f(x) = ∑ᵢⁿ cᵢ xⁱ 
  coeffs = [-0.021, 0.31, -1.1, 1.0]
  # coeffs will build f(x) = (x-0.1)(x-0.3)(x-0.7)
  #                        = -0.021 + 0.31 x - 1.1 x^2 + x^3
  n = length(coeffs) - 1

  #Define a map which determines a canonical ordering of the vertices of the network
  vertex_map = Dict(vertices(g) .=> [i for i in 1:L])
  #@show [edges(v) for v in vertices(underlying_graph(s))]

  ψ = poly_itn(s, vertex_map, coeffs)

  valid_points = [i / 2^L for i in 0:(2^L - 1)]
  values = []
  for point in valid_points
    xis = calculate_xis(point, vertex_map; print_x=false)
    eval_point = calculate_x(xis, vertex_map)
    @assert eval_point ≈ point
    ψ12proj = get_bitstring_network(ψ, s, xis)
    push!(values, ITensors.contract(ψ12proj)[])
  end

  ans = 0 * valid_points
  for (i, c) in enumerate(coeffs)
    ans .+= c * (valid_points) .^ (i - 1)
  end

  @show ans ≈ values
end

main()
