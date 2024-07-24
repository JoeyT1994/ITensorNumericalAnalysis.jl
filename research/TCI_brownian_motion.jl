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
  complex_continuous_siteinds,
  integrate
using Elliptic
using SpecialFunctions
using Distributions, Random

include("utils.jl")
include("commonnetworks.jl")
include("commonfunctions.jl")

function brownian_coeffs(nterms)
  return [rand(Normal()) for i in 1:nterms]
end

function weiener_process(x, ks)
  return ks[1]*x + sqrt(2)*sum([ks[i] * sin(pi *i * x) / (pi * i) for i in 2:length(ks)])
end

Random.seed!(123)
let

  # Define graph and indices
  L = 51
  delta = 2.0^(-L)
  χ1,χ2  = 8, 100
  #s = continuous_siteinds(g; map_dimension=2)
  s1 = qtt_siteinds_multidimstar_ordered(L, 5; map_dimension = 1, is_complex = false)
  s2 = qtt_siteinds_canonical(L; map_dimension = 1, is_complex = false)

  nterms = 10000
  ks = brownian_coeffs(nterms)
  f = x -> weiener_process(x, ks)

  tn1 = interpolate(f, s1; nsweeps=10, maxdim=χ1, cutoff=1e-16, outputlevel = 1)
  tn2 = interpolate(f, s2; nsweeps=10, maxdim=χ2, cutoff=1e-16, outputlevel=1)

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
