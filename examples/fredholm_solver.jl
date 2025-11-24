using ITensorNumericalAnalysis

using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: NamedGraph, NamedEdge, rename_vertices, edges, vertices
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree
using ITensors:
  ITensors,
  ITensor,
  Index,
  dim,
  tags,
  replaceprime!,
  inner,
  Op,
  op
using Dictionaries: Dictionary, set!
using Random: seed!
using TensorNetworkQuantumSimulator: setindex_preserve!, siteinds, tensors, tensornetwork, TensorNetwork, insert_virtualinds!

function main()

  seed!(1234)
  L = 100
  g = named_comb_tree((2, L ÷ 2))

  println(
    "########## Iteratively solve a inhomogeneous Fredholm equation of the second kind ##########",
  )
  println("solve f(x) = eˣ + ∫₀¹ (xy) f(y) dy")
  # solution: f(x) = 3x/2 + eˣ

  # start f(x) = f(x)⊗1_y
  # 1. make g(x,y)
  # 2. f*g 
  # 3. apply operator I or |x>
  # 4. apply shift if any

  s = continuous_siteinds(g, [[(i, j) for j in 1:(L ÷ 2)] for i in 1:2])
  dim_ψ = 2
  s1, s2 = reduced_indexmap(s, 1), reduced_indexmap(s, 2)

  ψ = const_tnf(g, s) # f(x) = 1_x⊗1_y

  # make g(x,y) = x*y
  gxy = poly_tnf(g, s, [0, 1]; dim=1) * poly_tnf(g, s, [0, 1]; dim=2)

  c1, c2 = exp_tnf(g, s; dim=1), exp_tnf(g, s; dim=2)

  niter = 20
  for iter in 1:niter
    ψ = ψ * gxy

    ψ = partial_integrate(ψ, [dim_ψ])

    new_tensors = Dictionary(dimension_vertices(s, dim_ψ), [ITensors.ITensor([1, 1], only(siteinds(s)[v])) for v in dimension_vertices(s, dim_ψ)])
    ψ = TensorNetwork(merge(new_tensors, tensors(tensornetwork(ψ))), g)
    ψ = TensorNetworkFunction(ψ, s)
    insert_virtualinds!(ψ)

    dim_ψ = dim_ψ == 1 ? 2 : 1

    c = dim_ψ == 1 ? c1 : c2

    ψ = ψ + c
  end

  n_grid = 100
  x_vals = grid_points(s, n_grid, 1)
  ψ_vals = dim_ψ == 1 ? [real(evaluate(ψ, [x, 0.5])) for x in x_vals] : [real(evaluate(ψ, [0.5, x])) for x in x_vals]
  correct_vals = (3 / 2) * x_vals + exp.(x_vals)

  avg_err = sum(abs.(correct_vals - ψ_vals)) / n_grid
  @show avg_err
end

main()