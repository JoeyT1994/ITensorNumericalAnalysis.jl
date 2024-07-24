include("commonnetworks.jl")
include("commonfunctions.jl")
include("utils.jl")

using ITensorNetworks: maxlinkdim
using NamedGraphs: nv
using NamedGraphs.GraphsExtensions: eccentricity
using NamedGraphs. NamedGraphGenerators: named_comb_tree, named_grid, named_binary_tree
using Random: Random
using SpecialFunctions

using NPZ
using MKL

using ITensorNumericalAnalysis: interpolate

include("mutualinfo.jl")
include("commonnetworks.jl")

Random.seed!(1234)

function siteinds_constructor(mode::String, L::Int64; map_dimension = 1, f = nothing)
  if mode == "CanonicalPath"
    return qtt_siteinds_canonical(L; map_dimension)
  elseif mode == "OrderedPath"
    return continuous_siteinds_ordered(named_grid((L,1)); map_dimension)
  elseif mode[1:(length(mode)-1)] == "OrderedStar"
    npoints = parse(Int64, last(mode))
    pointlength = Int64((L-1) / npoints)
    return continuous_siteinds_ordered(star(npoints, pointlength); map_dimension)
  elseif mode[1:(length(mode)-1)] == "CombTree"
    backbonelength = parse(Int64, last(mode))
    comblength = round(Int, L / backbonelength)
    return continuous_siteinds_ordered(named_comb_tree((backbonelength, comblength)); map_dimension)
  elseif mode == "BinaryTree"
    k = round(Int, log2(0.5*L + 1)) + 1
    return continuous_siteinds_ordered(named_binary_tree(k); map_dimension)
  elseif mode[1:(length(mode)-1)] == "MISearch"
    nsamples = 100
    mi = generate_mi_matrix(f, nsamples, L, map_dimension)
    mi = dropdims(mi, dims = (findall(size(mi) .== 1)...,))
    max_z = parse(Int64, last(mode))
    g, dimension_vertices = order_star_vertices(mi, L, max_z)
    return continuous_siteinds(g, dimension_vertices)
  end
end

function get_function(mode::String)
  if mode == "Weirstrass"
    nterms, a = 100, 3
    ks = weirstrass_coefficients(nterms, a)
    eval_function = x -> calulate_weirstrass(x, ks)
    return eval_function, nterms, ks
  elseif mode == "Airy"
    nterms, omega = 100, 250
    eval_function = x -> airyai(-omega * x)
    return eval_function, nterms, nothing
  elseif mode == "Gaussian"
    nterms, sigma = 100, 0.1
    eval_function = x -> exp(-(x-0.5)*(x-0.5) /(sigma*sigma))
    return eval_function, nterms, nothing
  end
end

function main(; md = nothing, func = nothing, l = nothing, chi = nothing, nsweeps = 10, save = true)
  mode = md == nothing ? ARGS[1] : md
  function_mode = func == nothing ? ARGS[2] : func
  L = l == nothing ? parse(Int64, ARGS[3]) : l
  χ = chi == nothing ? parse(Int64, ARGS[4]) : chi
  map_dimension = 1
  ngrid_points = 1000
  eval_function, _, _ = get_function(function_mode)
  s = siteinds_constructor(mode, L; map_dimension, f = x -> eval_function(only(x)))
  println("Graph is "*mode*" chi is $chi")

  fx, info = interpolate(eval_function, s; nsweeps, maxdim=χ, cutoff=1e-17, outputlevel=0)
  inf_norms = info[:, :error]
  regions = info[:, :region]
  sweeps = info[:,  :sweep]

  Lx = length(dimension_vertices(fx, 1))
  delta = (2^(-1.0*Lx))

  grid_points = Float64[delta * Random.rand(1:(2^Lx-1)) for i in 1:ngrid_points]
  fx_xs_exact = Float64[eval_function(x) for x in grid_points]
  fx_xs = Float64[real(evaluate(fx, x)) for x in grid_points]
  memory_req = no_elements(fx)
  error = 1 - calc_error_V2(fx_xs_exact, fx_xs)
  error_og = calc_error(fx_xs_exact, fx_xs)
  vectorized_regions = [[[first(r), last(r)] for r in region] for region in regions]


  println("Function constructed with an error of $error, an error_og of $error_og and a memory req of $memory_req")

  file_root = "/mnt/home/jtindall/Documents/Data/ITensorNumericalAnalysis/TCI/1D/"
  file_name = file_root * "L"*string(L)*"GRAPH"*mode*"FUNCTION"*function_mode*"CHI"*string(χ)*"NGRIDPOINTS"*string(ngrid_points)*"nsweeps"*string(nsweeps)*".npz"
  if save
    npzwrite(file_name, grid_points = grid_points, fx_xs = fx_xs, fx_xs_exact = fx_xs_exact, L = L, memory_req = memory_req, error = error, error_og = error_og, inf_norms = inf_norms, sweeps = sweeps)
  end
end

main(; )