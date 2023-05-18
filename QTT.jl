using ITensors
using ITensorNetworks
using NamedGraphs
using EllipsisNotation
using Graphs

using ITensorNetworks: delta_network, symmetrise_itensornetwork
using NamedGraphs: add_edges, random_bfs_tree

include("QTT_utils.jl")
        

function main()
    L = 10
    g = NamedGraph(Graphs.random_regular_graph(L, 3))
    gt = random_bfs_tree(g, 1)
    g = NamedGraph(vertices(gt))
    g = add_edges(g, edges(gt))
    g = named_grid((L, 1))
    a = 1.0
    k = 0.5
    nterms = 10
    s = siteinds("S=1/2", g)

    #Define a map which determines a canonical ordering of the vertices of the network
    vertex_map = Dict(vertices(g) .=> [i for i in 1:L])

    x = 0.25
    xis = calculate_xis(x, vertex_map; a, print_x = true)

    ψ =x_itensornetwork(s, vertex_map; k = 1.0)
    ψ = symmetric_gauge(ψ)
    @show normalize!(ψ[(2,1)])

    # cutoff = 1e-3
    # suggesteddim = round(Int64, -log(cutoff)/(2*x*k))
    # @show suggesteddim, nterms
    
    # maxdim = 5
    # for n = 1:nterms
    #     ψ12 = tanh_itensornetwork(s, vertex_map, n; a, k)
    #     if maxlinkdim(ψ12) > maxdim && typeof(ψ12) == TreeTensorNetwork
    #         ψ12 = truncate(ψ12; maxdim)
    #     end

    #     @show maxlinkdim(ψ12)
    #     ψ12proj = get_bitstring_network(ψ12, s, xis)
    #     @show (n, ITensors.contract(ψ12proj)[])
    # end
    # @show tanh(k*x)
    # eval_point = calculate_x(xis,vertex_map;a)
    # expansion = 1+sum([2*(-1)^n*exp(-2*n*eval_point*k) for n=1:nterms])
    # @show expansion

end

main()
