using ITensors
using ITensorNetworks
using NamedGraphs
using EllipsisNotation
using Graphs

using ITensorNetworks: delta_network
using NamedGraphs: add_edges

include("../QTT_utils.jl")

  @testset "test addition" begin
    L = 10
    #g = NamedGraph(Graphs.random_regular_graph(L, 3))
    g = named_grid((L, 1))
    a = 1.2
    s = siteinds("S=1/2", g)

    #Define a map which determines a canonical ordering of the vertices of the network
    vertex_map = Dict(vertices(g) .=> [i for i in 1:L])

    x = 0.32
    xis = calculate_xis(x, vertex_map; a, print_x=true)
    eval_point = calculate_x(xis, vertex_map; a)

    k = 0.3
    ψ12 = exp_itensornetwork(s, vertex_map; k=k, a=a)
    exact = exp(k * eval_point)
    for (net_func,k,func) in [(exp_itensornetwork,-0.4,exp),(cosh_itensornetwork,0.3,cosh)]
        exact += func(k * eval_point)
        ψ12 = ψ12+net_func(s, vertex_map; k=k, a=a)
    end
    ψ12proj = get_bitstring_network(ψ12, s, xis)

    network_ans = ITensors.contract(ψ12proj)[]
    @test exact ≈ network_ans
  end
