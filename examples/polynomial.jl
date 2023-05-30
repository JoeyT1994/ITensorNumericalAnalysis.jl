using ITensors
using ITensorNetworks
using NamedGraphs
using EllipsisNotation
using Graphs

using ITensorNetworks: delta_network

using NamedGraphs: add_edges, rename_vertices

include("../src/QTT_utils.jl")

function main()
  L = 10
  #g = named_grid((L, 1))
  g = NamedGraph(Graphs.SimpleGraph(uniform_tree(L)))
  #Rename vertices to make a_star work (need to fix this bug in NamedGraphs)
  g = rename_vertices(g, Dict(zip(vertices(g), [(v,1) for v in vertices(g)])))
  s = siteinds("S=1/2", g)

  vertex_map = Dict(vertices(g) .=> [i for i in 1:length(vertices(g))])
  coeffs = [1.0, 0.643, 1.0, -0.1, 0.567, -0.2, 0.3]


  ψ = poly_itn(s, vertex_map, coeffs)
  x = 0.875
  xis = calculate_xis(x, vertex_map)
  ψproj = get_bitstring_network(ψ, s, xis)
  fx = ITensors.contract(ψproj)[]
  @show fx
  n = length(coeffs) - 1
  @show sum([coeffs[i]*(x^(n + 1 - i)) for i in 1:n+1])

  
end

main()
