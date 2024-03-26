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

function c_tensor(phys_ind::Index, virt_inds::Vector)
  inds = vcat(phys_ind, virt_inds)
  @assert allequal(ITensors.dim.(virt_inds))
  #Build tensor to be delta on inds and independent of phys_ind
  T = ITensor(0.0, inds...)
  for i in 1:ITensors.dim(phys_ind)
    for j in 1:ITensors.dim(first(virt_inds))
      ind_array = [v => j for v in virt_inds]
      T[phys_ind => i, ind_array...] = 1.0
    end
  end

  return T
end

function copy_tensor_network(s::IndsNetwork, linkdim::Int64)
  tn = randomITensorNetwork(s; link_space = linkdim)
  for v in vertices(tn)
    virt_inds = setdiff(inds(tn[v]), Index[only(s[v])])
    tn[v] = c_tensor(only(s[v]), virt_inds)
  end

  return tn
end