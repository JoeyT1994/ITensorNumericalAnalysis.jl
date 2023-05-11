using ITensorNetworks

include("itensorutils.jl")

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

Base.:+(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork) = add_itensornetworks(tn1, tn2)

"""Given a bitstring collapse the relevant tensors of ψ down to get the TN which represents ψ[bitstring]"""
function get_bitstring_network(ψ::AbstractITensorNetwork, s::IndsNetwork,  bitstring::Dict)
    ψ = copy(ψ)
    for v in keys(bitstring)
        proj = ITensor([i != bitstring[v] ? 0 : 1 for i in 0:(dim(s[v])-1)], s[v])
        ψ[v] = ψ[v]*proj
    end

    return ψ
end
