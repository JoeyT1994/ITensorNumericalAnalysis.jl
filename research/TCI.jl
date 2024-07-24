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
  rand_itn
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
include("functioncompressioncluster3D.jl")

Random.seed!(123)
let

  # Define graph and indices
  L = 72
  Lx = Int(L/2)
  delta = 2.0^(-Lx)
  χ = 20
  #s = continuous_siteinds(g; map_dimension=2)
  s = siteinds_constructor("CombTree2", L; map_dimension = 4, is_complex = false)
  f = x -> 1 / (x[1] + x[2] + 1)^2

  tn = interpolate(f, s; initial_state = rand_itn(s; link_space = 2), nsweeps=20, maxdim=χ, cutoff=1e-18, outputlevel=1)

  ngrid_points = 50
  xs, ys = [delta * Random.rand(1:(2^Lx-1)) for i in 1:ngrid_points], [delta * Random.rand(1:(2^Lx-1)) for i in 1:ngrid_points]
  err1, err2 = 0, 0
  exact_vals = Float64[]
  tn_vals = Float64[]
  for x in xs
    for y in ys
      push!(tn_vals, real(evaluate(tn, [x, y, 0.5, 0.5])))
      push!(exact_vals, real(f([x, y, 0.5, 0.5])))
    end
  end

  err = calc_error(exact_vals, tn_vals)
  @show maximum(abs.(exact_vals - tn_vals))
  @show err

  return nothing
end
