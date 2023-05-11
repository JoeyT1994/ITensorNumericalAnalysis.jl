using ITensors
using ITensorNetworks
using NamedGraphs
using EllipsisNotation
using Graphs

using ITensorNetworks: delta_network
using NamedGraphs: add_edges

include("QTT_utils.jl")
        

function main()
    L = 8
    #g = NamedGraph(Graphs.random_regular_graph(L, 3))
    g = named_grid((L, 1))
    a = 2.0
    s = siteinds("S=1/2", g)

    #Define a map which determines a canonical ordering of the vertices of the network
    vertex_map = Dict(vertices(g) .=> [i for i in 1:L])

    ψ12 = cos_itensornetwork(s, vertex_map; a)
    ϕ12 = cos_itensornetwork(s, vertex_map; a)
    ψ12 = ψ12+ϕ12
    #ψ12 = treetensornetwork(ψ12)

    x = 1.9
    xis = calculate_xis(x, vertex_map; a)

    ψ12proj = get_bitstring_network(ψ12, s, xis)


    #Note the answer here is exact because our binary representation of x is exact, (only need 2 bits)
    @show ITensors.contract(ψ12proj)[]
    @show 2*cos(x)

end

main()
