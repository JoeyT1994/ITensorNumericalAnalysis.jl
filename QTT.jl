using ITensors
using ITensorNetworks
using NamedGraphs
using EllipsisNotation
using Graphs

using ITensorNetworks: delta_network
using NamedGraphs: add_edges

include("itensornetwork_functions.jl")
        

function main()
    L = 8
    g = NamedGraph(Graphs.random_regular_graph(L, 3))
    a = 2.0
    s = siteinds("S=1/2", g)

    #Define a map which determines a canonical ordering of the vertices of the network
    vertex_map = Dict(vertices(g) .=> [i for i in 1:L])

    ψ12 = cos_itensornetwork(s, vertex_map; a)


    #A Binary representation of a*0.75 with 10 bits
    xis = Dict(vertices(ψ12) .=> [i <= 2 ? 1 : 0 for i in 1:L])
    x = sum([a*xis[v]/2^vertex_map[v] for v in vertices(ψ12)])

    ψ12proj = get_bitstring_network(ψ12, s, xis)


    #Note the answer here is exact because our binary representation of x is exact, (only need 2 bits)
    @show ITensors.contract(ψ12proj)[]
    @show cos(x)

end

main()