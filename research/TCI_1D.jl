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
using Elliptic
using SpecialFunctions

include("utils.jl")
include("commonnetworks.jl")
include("commonfunctions.jl")

Random.seed!(123)
let

  # Define graph and indices
  L = 33
  delta = 2.0^(-L)
  χ = 15
  #s = continuous_siteinds(g; map_dimension=2)
  s1 = qtt_siteinds_multidimstar_ordered(L, 4; map_dimension = 1, is_complex = false)
  s2 = qtt_siteinds_canonical_sequentialdims(L; map_dimension = 1, is_complex = false)
  ks = weirstrass_coefficients(100, 3)
  #f = x -> calulate_weirstrass(x, ks)
  f = x -> airyai(-100*x)

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
