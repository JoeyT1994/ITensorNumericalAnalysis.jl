include("commonnetworks.jl")
include("commonfunctions.jl")
include("laughlinfunctions.jl")
include("utils.jl")
include("mutualinfo.jl")

using ITensorNetworks: maxlinkdim
using NamedGraphs.GraphsExtensions: add_edges, nv, eccentricity, disjoint_union, degree
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_grid, named_binary_tree
using Random: Random, rand
using LinearAlgebra: diagind, diagm
using NPZ
using MKL
using Distributions: Uniform

using ITensorNumericalAnalysis: interpolate

Random.seed!(1234)

function siteinds_constructor(mode::String, L::Int64; map_dimension = 3, is_complex = false, f = nothing)
  if mode == "CanonicalPath"
    return qtt_siteinds_canonical(L; map_dimension, is_complex)
  elseif mode == "SequentialPath"
    return qtt_siteinds_canonical_sequentialdims(L; map_dimension, is_complex)
  elseif mode == "OrderedPath"
    return continuous_siteinds_ordered(named_grid((L,1)); map_dimension, is_complex)
  elseif mode[1:(length(mode)-1)] == "OrderedStar"
    npoints = parse(Int64, last(mode))
    pointlength = Int64((L-1) / npoints)
    return continuous_siteinds_ordered(star(npoints, pointlength); map_dimension, is_complex)
  elseif mode[1:(length(mode)-1)] == "MultiDimOrderedStar"
      npoints = parse(Int64, last(mode))
      return qtt_siteinds_multidimstar_ordered(L, npoints; map_dimension, is_complex)
  elseif mode[1:(length(mode)-1)] == "CombTree"
    backbonelength = parse(Int64, last(mode))
    comblength = round(Int, L / backbonelength)
    return continuous_siteinds_ordered(named_comb_tree((backbonelength, comblength)); map_dimension, is_complex)
  elseif mode == "BinaryTree"
    k = round(Int, log2(0.5*L + 1)) + 1
    return continuous_siteinds_ordered(named_binary_tree(k); map_dimension, is_complex)
  elseif mode[1:(length(mode)-1)] == "MISearch"
    nsamples, max_z = 1000, parse(Int64, last(mode))
    mi = generate_mi_matrix(f, nsamples, round(Int, L / max_z), map_dimension)
    g = minimize_me(mi; max_z, alpha = 1)
    return continuous_siteinds(g, [[(i,j) for i in 1:round(Int, L / max_z)] for j in 1:map_dimension])
  end
end

function get_function(mode::String)
  if mode == "RandPlaneWaves"
    nterms = 40
    As = [1.0 for i in 1:nterms]
    kxs = [i*randn() for i in 1:nterms]
    kys = [i*randn() for i in 1:nterms]
    kzs = [i*randn() for i in 1:nterms]
    eval_function = (x, y, z) -> sum([As[i]*cos(kxs[i]*x + kys[i]*y + kzs[i]*z) for i in 1:nterms])
    return eval_function, nterms, (; nterms, As, kxs, kys, kzs)
  elseif mode == "PoissonKernel"
    nterms = 10
    eval_function = (r, α, θ) -> (1-r*r) / (1 + r*r - 2*r*cos(2*pi*(θ - α)))
    return eval_function, nterms, (;)
  elseif  mode == "PoissonGreensFunction"
    nterms = 10
    eval_function = (r, θ, rp, θp) -> log((r*r + rp*rp - 2*r*rp * cos(2*pi*(θ - θp)))/ (1+r*r*rp*rp - 2*r*rp * cos(2*pi*(θ - θp))))
    return eval_function, nterms, (;)
  elseif  mode == "Sphere"
    nterms = 10
    eval_function = (x, y, z) -> x*x + y*y + z*z < 1 ? 1.0 : 0.0
    return eval_function, nterms, (;)
  elseif mode[1:(length(mode)-1)] == "Gaussian"
    Random.seed!(1243)
    nterms = 50
    ndims = parse(Int64, last(mode))
    Ms = []
    centres = []
    for i in 1:nterms
      push!(centres, [rand(Uniform(.2, .8)) for i in 1:ndims])
      M = 0.1*randn((ndims,ndims))
      M = M * M'
      push!(Ms, inv(M))
    end
    eval_function = x -> sum([exp(-(x - c)' * M * (x - c)) for (c,M) in zip(centres, Ms)])
    return eval_function, nterms, (;)
  end
end

function main(; md = nothing, func = nothing, l = nothing, chi = nothing, nsweeps = 10, save = true)
  mode = md == nothing ? ARGS[1] : md
  function_mode = func == nothing ? ARGS[2] : func
  L = l == nothing ? parse(Int64, ARGS[3]) : l
  χ = chi == nothing ? parse(Int64, ARGS[4]) : chi
  map_dimension = 3
  eval_function, _, _ = get_function(function_mode)
  s = siteinds_constructor(mode, L; map_dimension, f = eval_function)
  println("Graph is "*mode*" chi is $χ")

  init_state = rand_itn(s; link_dim = 2)
  fxyz, info = interpolate(eval_function, s; initial_state = init_state, maxdim = χ, nsweeps,cutoff = 1e-20, outputlevel=1)
  inf_norms = info[:, :error]
  regions = info[:, :region]
  sweeps = info[:,  :sweep]

  Lx = length(dimension_vertices(fxyz, 1))
  delta = (2^(-1.0*Lx))

  alg = maximum([degree(s, v) for v in vertices(s)]) >= 4 ? "bp" : "ttn"
  z_fxyz = inner(fxyz, fxyz; alg)
  χmax = maxlinkdim(fxyz)
  println("Function built with χmax = $χmax")

  bond_dims = [χ for χ in χmax:-1:1]
  no_bds = length(bond_dims)

  memory_req = zeros(Int64, (no_bds))
  overlaps = zeros(Float64, (no_bds))
  ngrid_points = 100
  delta = 2.0^(-Lx)
  grid_points = zeros(Float64, (ngrid_points, map_dimension))
  for i in 1:ngrid_points
    grid_points[i, :] = [delta * Random.rand(1:(2^Lx-1)) for d in 1:map_dimension]
  end
  exact_vals = Float64[real(eval_function(grid_points[i, :])) for i in 1:ngrid_points]
  trunc_vals = zeros(Float64, (no_bds, ngrid_points))
  l2_errors = zeros(Float64, (no_bds))

  for (i, χ) in enumerate(bond_dims)
    println("Truncating down to chi = $χ")
    fxyz_trunc = truncate(fxyz; maxdim = χ)
    z_trunc = inner(fxyz, fxyz; alg)
    println("Evaluating function")

    f = inner(fxyz_trunc, fxyz_trunc; alg)
    err= (f * conj(f)) / (z_trunc * z_fxyz)
    overlaps[i] = real(err)
    memory_req[i] = no_elements(fxyz_trunc)
    trunc_vals[i, :] = Float64[real(evaluate(fxyz_trunc, grid_points[i, :])) for i in 1:ngrid_points]
    l2_errors[i] = calc_error(exact_vals, trunc_vals[i, :])
    println("Achieved an overlap error of $(1.0 - overlaps[i])")
    println("Memory req was $(memory_req[i])")
    println("Error val is $(l2_errors[i])")
    flush(stdout)
  end


  println("Function constructed with an error of $error, and a memory req of $memory_req")

  file_root = "/mnt/home/jtindall/Documents/Data/ITensorNumericalAnalysis/TCI/3D/"
  file_name = file_root * "L"*string(L)*"GRAPH"*mode*"FUNCTION"*function_mode*"CHI"*string(χ)*"NGRIDPOINTS"*string(ngrid_points)*"nsweeps"*string(nsweeps)*".npz"
  if save
    npzwrite(file_name, grid_points =grid_points, exact_vals = exact_vals, L = L, memory_req = memory_req, l2_errors = l2_errors, overlaps = overlaps, trunc_vals = trunc_vals, inf_norms = inf_norms, sweeps = sweeps, bond_dims = bond_dims)
  end
end

χ =20
main(; chi = χ, func = "Gaussian3", md = "CanonicalPath", l = 30, save = true)
main(; chi = χ, func = "Gaussian3", md = "SequentialPath", l = 30, save = true)
main(; chi = χ, func = "Gaussian3", md = "CombTree3", l = 30, save = true)