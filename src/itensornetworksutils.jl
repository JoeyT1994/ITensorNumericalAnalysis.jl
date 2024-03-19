#Imports
using ITensors
using Graphs
using NamedGraphs
using ITensorNetworks
using EllipsisNotation
using Dictionaries
using DataGraphs

using ITensorNetworks: delta_network, data_graph, data_graph_type
using NamedGraphs: add_edges, random_bfs_tree, rem_edges

using SplitApplyCombine

using Random
using Distributions

include("itensornetworks_elementary_functions.jl")

"""Given a bitstring collapse the relevant tensors of ψ down to get the TN which represents ψ[bitstring]"""
function get_bitstring_network(ψ::AbstractITensorNetwork, s::IndsNetwork, bitstring::Dict)
  ψ = copy(ψ)
  for v in keys(bitstring)
    proj = ITensor([i != bitstring[v] ? 0 : 1 for i in 0:(dim(s[v]) - 1)], s[v])
    ψ[v] = ψ[v] * proj
  end

  return ψ
end

"""Build the order L tensor corresponding to fx(x): x ∈ [0,1]."""
function build_full_rank_tensor(L::Int64, fx::Function)
  inds = [Index(2, "$i") for i in 1:L]
  dims = Tuple([2 for i in 1:L])
  array = zeros(dims)
  for i in 0:(2^(L) - 1)
    xis = digits(i; base=2, pad=L)
    x = sum([xis[i] / (2^i) for i in 1:L])
    array[Tuple(xis + ones(Int64, (L)))...] = fx(x)
  end

  return ITensor(array, inds)
end
