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

include("utils.jl")
include("commonnetworks.jl")
include("commonfunctions.jl")

function integrand(x,y)
  return (sqrt(x^3 + y^4 + 5) / (x^2 + y + 3)) * cos(100*sqrt(x*x+y*y+2))
end

Random.seed!(123)
let

  # Define graph and indices
  L = 32
  Lx = Int(L/2)
  delta = 2.0^(-Lx)
  χ1,χ2  = 10, 10
  #s = continuous_siteinds(g; map_dimension=2)
  s1 = qtt_siteinds_multidimstar_ordered(L, 3; map_dimension = 2, is_complex = false)
  s2 = qtt_siteinds_canonical_sequentialdims(L; map_dimension = 2, is_complex = false)
  s3 = qtt_siteinds_canonical(L; map_dimension = 2, is_complex = false)

  f = x -> x[1] < x[2] ? x[1]*(1-x[2]) : x[2]*(1-x[1])

  tn1 = interpolate(f, s1; nsweeps=10, maxdim=χ1, cutoff=1e-16, outputlevel = 1)
  tn2 = interpolate(f, s2; nsweeps=10, maxdim=χ2, cutoff=1e-16, outputlevel=1)
  tn3 = interpolate(f, s3; nsweeps=10, maxdim=χ2, cutoff=1e-16, outputlevel=1)

  ngrid_points = 10
  xs, ys = [delta * Random.rand(1:(2^Lx-1)) for i in 1:ngrid_points], [delta * Random.rand(1:(2^Lx-1)) for i in 1:ngrid_points]
  err1, err2 = 0, 0
  exact_vals = Float64[]
  tn1_vals = Float64[]
  tn2_vals = Float64[]
  tn3_vals = Float64[]
  for x in xs
    for y in ys
      push!(tn1_vals, real(evaluate(tn1, [x, y])))
      push!(tn2_vals, real(evaluate(tn2, [x, y])))
      push!(tn3_vals, real(evaluate(tn3, [x, y])))
      push!(exact_vals, real(f([x, y])))
    end
  end

  err1, err2, err3 = calc_error(exact_vals, tn1_vals), calc_error(exact_vals, tn2_vals), calc_error(exact_vals, tn3_vals)
  @show err1, err2, err3

  return nothing
end
