using Graphs: SimpleGraph, is_tree, uniform_tree
using ITensors: contract, onehot, scalar
using ITensorNetworks: ITensorNetwork, siteinds, vertices
using NamedGraphs: NamedGraph, rename_vertices, edges, degree
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
  complex_continuous_siteinds,
  integrate
using Elliptic
using SpecialFunctions
using Distributions, Random

include("utils.jl")
include("commonnetworks.jl")
include("commonfunctions.jl")
include("mutualinfo.jl")

Random.seed!(123)
let

  # Define graph and indices
  L = 40
  delta = 2.0^(-L)
  g = named_grid((L,1))
  f =  x -> cos(100*log(1+only(x)))
  nsamples = 1000
  mi_m = generate_mi_matrix(f, nsamples, L, 1)
  g1 = minimize_me(g, mi_m; max_z = 3, alpha = 2)
  s1 = continuous_siteinds(g1, [[(i,1) for i in 1:L]])
  s2 = continuous_siteinds(g, [[(i,1) for i in 1:L]])

  χ =4
  tn1 = interpolate(f, s1; nsweeps=10, maxdim=χ, cutoff=1e-16, outputlevel=1)
  tn2 = interpolate(f, s2; nsweeps=10, maxdim=χ, cutoff=1e-16, outputlevel=1)

  ngrid_points = 250
  xs = [delta * Random.rand(1:(2^L-1)) for i in 1:ngrid_points]
  err1, err2 = 0, 0
  exact_vals = Float64[]
  tn1_vals = Float64[]
  tn2_vals = Float64[]
  for x in xs
    push!(tn1_vals, real(evaluate(tn1, x)))
    push!(tn2_vals, real(evaluate(tn2, x)))
    push!(exact_vals, real(f(x)))
  end

  err1, err2 = calc_error(exact_vals, tn1_vals), calc_error(exact_vals, tn2_vals)
  @show err1, err2


  return nothing
end
