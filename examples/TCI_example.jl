using Graphs: SimpleGraph, is_tree, uniform_tree
using ITensors: contract, onehot, scalar
using ITensorNetworks: ITensorNetwork, siteinds, vertices
using NamedGraphs: NamedGraph, rename_vertices
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_path_graph, named_grid
using Random
using LinearAlgebra: diagind
using ITensorNumericalAnalysis: indsnetwork, continuous_siteinds, interpolate, calculate_p, evaluate, IndsNetworkMap

Random.seed!(123)
let

  # Define graph and indices
  L = 12
  g = NamedGraph(SimpleGraph(uniform_tree(L)))
  g = rename_vertices(v -> (v, 1), g)
  #s = siteinds("Qubit", g)
  s = continuous_siteinds(g; map_dimension = 2)

  f = x -> sech(x[1] - 5*x[2])

  # Learn function using TCI
  nsweeps = 2
  cutoff = 1E-5
  maxdim = 10
  tn = interpolate(f, s; nsweeps, maxdim, cutoff, outputlevel=1)

  x, y = 0.25, 0.5
  @show evaluate(tn, [x,y])
  @show f([x,y])

  return nothing
end