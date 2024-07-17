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
using ITensorNetworks: maxlinkdim

function two_soliton(x, y)
  num = 12 * (3 + 4 * cosh(2 * x - 8 * y) + cosh(4 * x - 64 * y))
  den = 3 * cosh(x - 28 * y) + cosh(3 * x - 36 * y)
  den = den * den
  return -12 * num / den
end

include("utils.jl")
include("commonnetworks.jl")
include("commonfunctions.jl")

Random.seed!(123)
let

  # Define graph and indices
  L = 38
  Lx = Int(L/2)
  delta = 2.0^(-Lx)
  χ = 100
  #s = continuous_siteinds(g; map_dimension=2)
  s1 = qtt_siteinds_multidimstar_ordered(L, 3; map_dimension = 2, is_complex = false)
  s2 = qtt_siteinds_canonical_sequentialdims(L; map_dimension = 2, is_complex = false)
  #f = x -> sech(3*(x[1] - 1.5*x[2]))*sech(3*(x[1] - 1.5*x[2]))
  #f = x -> two_soliton(12*x[1] - 6, 2*x[2] / 3 - (1/3))
  #f = x -> in_mandelbrot(2 * x[1] * exp(1 *im * 2 * pi * x[2]))
  f = x -> x[1] <= x[2] ? (1-x[2])*x[1] : (1-x[1])*x[2]

  tn1 = interpolate(f, s1; initial_state = rand_itn(s1; link_space = 2), nsweeps=20, maxdim=χ, cutoff=1e-18, outputlevel=1)
  tn2 = interpolate(f, s2; initial_state = rand_itn(s2; link_space = 2), nsweeps=20, maxdim=χ, cutoff=1e-18, outputlevel=1)

  ngrid_points = 25
  xs, ys = [delta * Random.rand(1:(2^Lx-1)) for i in 1:ngrid_points], [delta * Random.rand(1:(2^Lx-1)) for i in 1:ngrid_points]
  err1, err2 = 0, 0
  exact_vals = Float64[]
  tn1_vals = Float64[]
  tn2_vals = Float64[]
  for x in xs
    for y in ys
      push!(tn1_vals, real(evaluate(tn1, [x, y])))
      push!(tn2_vals, real(evaluate(tn2, [x, y])))
      push!(exact_vals, real(f([x, y])))
    end
  end

  err1, err2 = calc_error(exact_vals, tn1_vals), calc_error(exact_vals, tn2_vals)
  @show err1, err2

  return nothing
end
