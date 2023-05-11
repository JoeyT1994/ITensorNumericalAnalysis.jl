using ITensors
using ITensorNetworks
using NamedGraphs
using EllipsisNotation
using Graphs

using ITensorNetworks: delta_network
using NamedGraphs: add_edges

include("QTT_utils.jl")
        

function main()
    L = 10
    #g = NamedGraph(Graphs.random_regular_graph(L, 3))
    g = named_grid((L, 1))
    a = 1.0
    k = 0.1
    nterms = 115
    s = siteinds("S=1/2", g)

    #Define a map which determines a canonical ordering of the vertices of the network
    vertex_map = Dict(vertices(g) .=> [i for i in 1:L])

    x = 0.25
    xis = calculate_xis(x, vertex_map; a, print_x = true)

    cutoff = 1e-3
    suggesteddim = round(Int64, -log(cutoff)/(2*x*k))
    @show suggesteddim, nterms
    
    maxdim = 5
    for n = 1:nterms
        ψ12 = tanh_itensornetwork(s, vertex_map, n; a, k)
        ψ12 = TTN(ψ12)
        if maxlinkdim(ψ12) > maxdim
            ψ12 = truncate(ψ12; maxdim)
        end

        @show maxlinkdim(ψ12)
        ψ12proj = get_bitstring_network(ψ12, s, xis)
        @show (n, ITensors.contract(ψ12proj)[])
    end
    @show tanh(k*x)
    eval_point = calculate_x(xis,vertex_map;a)
    expansion = 1+sum([2*(-1)^n*exp(-2*n*eval_point*k) for n=1:nterms])
    @show expansion

end

main()