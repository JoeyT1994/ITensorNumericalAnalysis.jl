using Graphs: SimpleGraph, is_tree, uniform_tree, binary_tree, add_vertex!, add_edge!
using ITensors: contract, onehot, scalar
using ITensorNetworks: ITensorNetwork, siteinds, vertices, edges
using NamedGraphs: NamedGraph, rename_vertices
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_path_graph, named_grid
using Random
using LinearAlgebra: diagind
include("../src/ITensorNumericalAnalysis.jl")
using .ITensorNumericalAnalysis:
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
  dim = 2
  # L = 8

  """
  Generate the BTTN from http://arxiv.org/abs/2410.03572.

  Returns a BTTN representing a function with `dim` input variables
  by representing every input variable as a subtree with depth `tree_depth_per_dimension`.
  """
  function bttn(dim, tree_depth_per_dimension)
    subtree = NamedGraph(SimpleGraph(binary_tree(tree_depth_per_dimension)))
    add_vertex!(subtree, 0)
    add_edge!(subtree, 0, 1)
    g = NamedGraph([(1, i) for i in 1:dim])
    subtrees = [rename_vertices(v -> (v+1, i), subtree) for i in 1:dim]
    # The vertex order is needed for default_dimension_vertices in digit_inds to work
    # (map_dimension=dim).
    for v in zip([vertices(subtree) for subtree in subtrees]...)
      for i in 1:dim
        add_vertex!(g, v[i])
      end
    end
    for (i, subtree) in enumerate(subtrees)
      if i < dim
        add_edge!(g, (1, i), (1, i+1))
      end
      for edge in edges(subtree)
        add_edge!(g, edge)
      end
    end
    return g
  end

  # g = NamedGraph(SimpleGraph(binary_tree(3)))
  # g = # named_comb_tree((2, L ÷ 2)) # named_grid((L, 1))
  # g = rename_vertices(v -> (v+1, 1), g)

  g = bttn(dim, 3)
  s = continuous_siteinds(g; map_dimension=2)
  println(s)
  exit()
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
