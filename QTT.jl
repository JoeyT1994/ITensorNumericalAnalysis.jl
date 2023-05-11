using ITensors
using ITensorNetworks
using NamedGraphs
using EllipsisNotation
using Graphs

using ITensorNetworks: delta_network
using NamedGraphs: add_edges

"""Given a bitstring collapse the relevant tensors of ψ down to get the TN which represents ψ[bitstring]"""
function get_bitstring_network(ψ::ITensorNetwork, s::IndsNetwork,  bitstring::Dict)
    ψ = copy(ψ)
    for v in keys(bitstring)
        proj = ITensor([i != bitstring[v] ? 0 : 1 for i in 0:(dim(s[v])-1)], s[v])
        ψ[v] = ψ[v]*proj
    end

    return ψ
end

"""Add two ITensors with different index sets. Currently we require that they have the same number of indices and the same
number of unique indices"""
function add_itensors(A::ITensor, B::ITensor)
    @assert length(inds(A)) == length(inds(B))

    cinds = commoninds(A, B)
    Auinds = uniqueinds(A, B)
    Buinds = uniqueinds(B, A)
    @assert length(Auinds) == length(Buinds)

    Ap, Bp = permute(A, vcat(cinds, Auinds)), permute(B, vcat(cinds, Buinds))

    Ainds = inds(Ap)
    Binds = inds(Bp)


    A_array = Array(Ap, Ainds)
    B_array = Array(Bp, Binds)

    A_type = eltype(A_array)
    B_type = eltype(B_array)
    @assert A_type == B_type
    extended_dims = [dim(Auinds[i]) + dim(Buinds[i]) for i in 1:length(Auinds)]
    Apb_newdims = Tuple(vcat([dim(i) for i in cinds], extended_dims))
    out_array = zeros(A_type, Apb_newdims)
    A_ind = Tuple(vcat([..], [1:dim(i) for i in Auinds]))
    out_array[A_ind...] = A_array

    B_ind = Tuple(vcat([..], [(dim(Auinds[i]) + 1):(dim(Auinds[i]) + dim(Buinds[i])) for i in 1:length(Auinds)]))
    out_array[B_ind...] = B_array

    out_inds = vcat(cinds, [Index(extended_dims[i], tags(Auinds[i])) for i in 1:length(Auinds)])

    return ITensor(out_array, out_inds)

end

"""Add two itensornetworks together by growing the bond dimension. The network structures need to be identical (same edges, vertex names) and the local
tensors must have the same site indices but can have different link indices"""
function add_itensornetworks(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork)

    @assert issetequal(vertices(tn1), vertices(tn2))

    tn_1p2 = copy(tn1)
    for v in vertices(tn1)

        @assert issetequal(siteinds(tn1, v), siteinds(tn2, v))
        tn1_linds = setdiff(inds(tn1[v]), siteinds(tn1, v)) 
        tn2_linds = setdiff(inds(tn2[v]), siteinds(tn2, v)) 

        @assert length(tn1_linds) == length(tn2_linds)

        tn_1p2[v] = add_itensors(tn1[v], tn2[v])
    end

    tn_1p2 = add_edges(tn_1p2, edges(tn1))

    for e in edges(tn_1p2)
        tsrc, tdst = tn_1p2[src(e)], tn_1p2[dst(e)]
        tsrc_link_index = inds(tsrc)[findall(i -> tags(i) ∈ tags.(inds(tdst)), inds(tsrc))]
        tdst_link_index = inds(tdst)[findall(i -> tags(i) ∈ tags.(inds(tsrc)), inds(tdst))]

        replaceinds!(tn_1p2[dst(e)], tdst_link_index, tsrc_link_index)
    end

    return tn_1p2
end

"""Construct the product state representation of the function f(x) = const."""
function const_itensornetwork(s::IndsNetwork; c::Union{Float64, ComplexF64} = 1.0)
    ψ =  delta_network(s; link_space = 1)
    ψ[first(vertices(ψ1))] *= c
end

"""Construct the product state representation of the exp(kx) function for x ∈ [0,a] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function exp_itensornetwork(s::IndsNetwork, vertex_map::Dict; k::Union{Float64, ComplexF64} = Float64(1.0), a::Float64 = 1.0)
    ψ =  delta_network(s; link_space = 1)
    for v in vertices(ψ)
        ψ[v] = ITensor([1.0, exp(k*a/(2^vertex_map[v]))], inds(ψ[v]))
    end

    return ψ
end

"""Construct the bond dim 2 representation of the cosh(kx) function for x ∈ [0,a] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function cosh_itensornetwork(s::IndsNetwork, vertex_map::Dict; k::Union{Float64, ComplexF64} = Float64(1.0), a::Float64 = 1.0)
    ψ1 =  exp_itensornetwork(s, vertex_map; a)
    ψ2 =  exp_itensornetwork(s, vertex_map; a, k = -1.0)

    ψ1[first(vertices(ψ1))] *= 0.5
    ψ2[first(vertices(ψ1))] *= 0.5

    return add_itensornetworks(ψ1, ψ2)
end

"""Construct the bond dim 2 representation of the sinh(kx) function for x ∈ [0,a] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function sinh_itensornetwork(s::IndsNetwork, vertex_map::Dict; k::Union{Float64, ComplexF64} = Float64(1.0), a::Float64 = 1.0)
    ψ1 =  exp_itensornetwork(s, vertex_map; a)
    ψ2 =  exp_itensornetwork(s, vertex_map; a, k = -1.0)

    ψ1[first(vertices(ψ1))] *= 0.5
    ψ2[first(vertices(ψ1))] *= -0.5

    return add_itensornetworks(ψ1, ψ2)
end

"""Construct the bond dim 2 representation of the cos(kx) function for x ∈ [0,a] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function cos_itensornetwork(s::IndsNetwork, vertex_map::Dict; k::Float64 = 1.0, a::Float64 = 1.0)
    ψ1 =  exp_itensornetwork(s, vertex_map; a, k = 1.0*im)
    ψ2 =  exp_itensornetwork(s, vertex_map; a, k = -1.0*im)

    ψ1[first(vertices(ψ1))] *= 0.5
    ψ2[first(vertices(ψ1))] *= 0.5

    return add_itensornetworks(ψ1, ψ2)
end

"""Construct the bond dim 2 representation of the sin(kx) function for x ∈ [0,a] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function sin_itensornetwork(s::IndsNetwork, vertex_map::Dict; k::Float64 = 1.0, a::Float64 = 1.0)
    ψ1 =  exp_itensornetwork(s, vertex_map; a, k = 1.0*im)
    ψ2 =  exp_itensornetwork(s, vertex_map; a, k = -1.0*im)

    ψ1[first(vertices(ψ1))] *= -0.5*im
    ψ2[first(vertices(ψ1))] *= 0.5*im

    return add_itensornetworks(ψ1, ψ2)
end
        

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