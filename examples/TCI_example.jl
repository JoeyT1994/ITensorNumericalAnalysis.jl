using Graphs: SimpleGraph, is_tree, uniform_tree
using ITensors: contract, onehot, scalar
using ITensorNetworks: ITensorNetwork, siteinds, vertices
using NamedGraphs: NamedGraph, rename_vertices
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_path_graph, named_grid
using Random
using LinearAlgebra: diagind
using ITensorNumericalAnalysis: indsnetwork, continuous_siteinds, interpolate, calculate_p, evaluate, IndsNetworkMap,
  complex_continuous_siteinds

Random.seed!(123)
let

  # Define graph and indices
  L = 12
  g = NamedGraph(SimpleGraph(uniform_tree(L)))
  g = rename_vertices(v -> (v, 1), g)
  #s = siteinds("Qubit", g)
  s = complex_continuous_siteinds(g; map_dimension = 1)

  f = x -> sech(x)

  # Learn function using TCI
  nsweeps = 3
  cutoff = 1E-10
  maxdim = 25
  tn = interpolate(f, s; nsweeps, maxdim, cutoff, outputlevel=1)

  x = 0.25 + 0.125*im
  @show evaluate(tn, x)
  @show f(x)

  return nothing
end