using ITensors
using ITensorNetworks
using NamedGraphs
using EllipsisNotation
using Graphs

using ITensorNetworks: delta_network
using NamedGraphs: add_edges

include("../QTT_utils.jl")

funcs = [
  ("cosh", cosh_itensornetwork, cosh),
  ("sinh", sinh_itensornetwork, sinh),
  ("exp", exp_itensornetwork, exp),
  ("cos", cos_itensornetwork, cos),
  ("sin", sin_itensornetwork, sin),
]
for (name, net_func, func) in funcs
  @testset "test $name" begin
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
    xis = calculate_xis(x, vertex_map; a, print_x=true)

    ψ12 = net_func(s, vertex_map; k=k, a=a)
    ψ12proj = get_bitstring_network(ψ12, s, xis)

    network_ans = ITensors.contract(ψ12proj)[]
    eval_point = calculate_x(xis, vertex_map; a)
    exact = func(k * eval_point)
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
  xis = calculate_xis(x, vertex_map; a, print_x=true)

  ψ12 = const_itensornetwork(s; c=k)
  ψ12proj = get_bitstring_network(ψ12, s, xis)

  network_ans = ITensors.contract(ψ12proj)[]
  eval_point = calculate_x(xis, vertex_map; a)
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
  xis = calculate_xis(x, vertex_map; a, print_x=true)

  cutoff = 1e-3
  suggesteddim = round(Int64, -log(cutoff) / (2 * x * k))
  @show suggesteddim, nterms

  maxdim = 5
  n = nterms
  ψ12 = tanh_itensornetwork(s, vertex_map, n; a, k)
  ψ12proj = get_bitstring_network(ψ12, s, xis)

  network_ans = ITensors.contract(ψ12proj)[]
  eval_point = calculate_x(xis, vertex_map; a)
  expansion = 1 + sum([2 * (-1)^n * exp(-2 * n * eval_point * k) for n in 1:nterms])
  @test expansion ≈ network_ans
end
