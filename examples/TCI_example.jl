using Graphs: SimpleGraph, is_tree, uniform_tree
using ITensors: contract, onehot, scalar
using ITensorNetworks: ITensorNetwork, siteinds, vertices
using NamedGraphs: NamedGraph, rename_vertices
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_path_graph, named_grid
using Random
using LinearAlgebra: diagind
using ITensorNumericalAnalysis:
  indsnetwork,
  continuous_siteinds,
  interpolate,
  calculate_p,
  evaluate,
  IndsNetworkMap,
  complex_continuous_siteinds

function two_soliton(x, y)
  num = 12 * (3 + 4 * cosh(2 * x - 8 * y) + cosh(4 * x - 64 * y))
  den = 3 * cosh(x - 28 * y) + cosh(3 * x - 36 * y)
  den = den * den
  return -12 * num / den
end

Random.seed!(123)
let

  # Define graph and indices
  L = 12
  #g = NamedGraph(SimpleGraph(uniform_tree(L)))
  #g = rename_vertices(v -> (v, 1), g)
  g = named_grid((L, 1))
  s = continuous_siteinds(g; map_dimension=2)
  #f = x -> sech(3*(x[1] - 1.5*x[2]))*sech(3*(x[1] - 1.5*x[2]))
  f = x -> two_soliton(x[1], x[2])

  tn = interpolate(f, s; nsweeps=3, maxdim=30, cutoff=1e-10, outputlevel=1)

  xs, ys = [0.0625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875],
  [0.0625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
  err = 0
  for x in xs
    for y in ys
      tn_xy = evaluate(tn, [x, y])
      f_xy = f([x, y])
      err += abs(tn_xy - f_xy) / (f_xy)
    end
  end

  @show err

  return nothing
end
