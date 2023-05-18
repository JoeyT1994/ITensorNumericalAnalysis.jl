using ITensors
using ITensorNetworks
using NamedGraphs
using EllipsisNotation
using Graphs

using ITensorNetworks: delta_network
using NamedGraphs: add_edges

include("../src/QTT_utils.jl")

function main()
  L = 10
  #g = NamedGraph(Graphs.random_regular_graph(L, 3))
  g = named_grid((L, 1))
  a = 1.0
  k = 0.5
  nterms = 50
  s = siteinds("S=1/2", g)

  #Define a map which determines a canonical ordering of the vertices of the network
  vertex_map = Dict(vertices(g) .=> [i for i in 1:L])

  x = 0.25
  xis = calculate_xis(x, vertex_map; print_x=true)

  cutoff = 1e-3
  suggesteddim = round(Int64, -log(cutoff) / (2 * x * k))
  @show suggesteddim, nterms

  maxdim = 5
  for n in 1:nterms
    ψ12 = tanh_itn(s, vertex_map, n; a, k)
    if maxlinkdim(ψ12) > maxdim && isa(ψ12, TreeTensorNetwork)
      ψ12 = truncate(ψ12; maxdim)
    end

    ψ12proj = get_bitstring_network(ψ12, s, xis)
    @show (n, ITensors.contract(ψ12proj)[])
  end
  @show tanh(k * x + a)
  eval_point = calculate_x(xis, vertex_map)
  expansion =
    1 + sum([2 * (-1)^n * exp(-2 * n * eval_point * k - 2 * n * a) for n in 1:nterms])
  @show expansion
end

main()
