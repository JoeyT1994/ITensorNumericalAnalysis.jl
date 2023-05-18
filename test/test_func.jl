using ITensors
using ITensorNetworks
using NamedGraphs
using EllipsisNotation
using Graphs
using Test

using ITensorNetworks: delta_network
using NamedGraphs: add_edges

include("../src/QTT_utils.jl")

funcs = [
  ("cosh", cosh_itn, cosh),
  ("sinh", sinh_itn, sinh),
  ("exp", exp_itn, exp),
  ("cos", cos_itn, cos),
  ("sin", sin_itn, sin),
]
for (name, net_func, func) in funcs
  @testset "test $name" begin
    L = 10
    #g = NamedGraph(Graphs.random_regular_graph(L, 3))
    g = named_grid((L, 1))
    a = 1.2
    k = 0.125
    s = siteinds("S=1/2", g)

    #Define a map which determines a canonical ordering of the vertices of the network
    vertex_map = Dict(vertices(g) .=> [i for i in 1:L])

    x = 0.25
    xis = calculate_xis(x, vertex_map; print_x=true)

    ψ12 = net_func(s, vertex_map; k=k, a=a)
    ψ12proj = get_bitstring_network(ψ12, s, xis)

    network_ans = ITensors.contract(ψ12proj)[]
    eval_point = calculate_x(xis, vertex_map)
    exact = func(k * eval_point + a)
    @test exact ≈ network_ans
  end
end

@testset "test const" begin
  L = 10
  #g = NamedGraph(Graphs.random_regular_graph(L, 3))
  g = named_grid((L, 1))
  a = 1.2
  k = 0.125
  nterms = 115
  s = siteinds("S=1/2", g)

  #Define a map which determines a canonical ordering of the vertices of the network
  vertex_map = Dict(vertices(g) .=> [i for i in 1:L])

  x = 0.25
  xis = calculate_xis(x, vertex_map; print_x=true)

  ψ12 = const_itn(s; c=k)
  ψ12proj = get_bitstring_network(ψ12, s, xis)

  network_ans = ITensors.contract(ψ12proj)[]
  eval_point = calculate_x(xis, vertex_map)
  exact = k
  @test exact ≈ network_ans
end
@testset "test tanh" begin
  L = 10
  #g = NamedGraph(Graphs.random_regular_graph(L, 3))
  g = named_grid((L, 1))
  a = 1.2
  k = 0.125
  nterms = 115
  s = siteinds("S=1/2", g)

  #Define a map which determines a canonical ordering of the vertices of the network
  vertex_map = Dict(vertices(g) .=> [i for i in 1:L])

  x = 0.25
  xis = calculate_xis(x, vertex_map; print_x=true)

  cutoff = 1e-3
  suggesteddim = round(Int64, -log(cutoff) / (2 * x * k))
  @show suggesteddim, nterms

  maxdim = 5
  n = nterms
  ψ12 = tanh_itn(s, vertex_map, n; a, k)
  ψ12proj = get_bitstring_network(ψ12, s, xis)

  network_ans = ITensors.contract(ψ12proj)[]
  eval_point = calculate_x(xis, vertex_map)
  expansion =
    1 + sum([2 * (-1)^n * exp(-2 * n * eval_point * k - 2 * n * a) for n in 1:nterms])
  @test expansion ≈ network_ans
end
