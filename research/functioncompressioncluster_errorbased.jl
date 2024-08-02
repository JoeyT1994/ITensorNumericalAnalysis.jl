include("commonnetworks.jl")
include("commonfunctions.jl")
include("utils.jl")

using ITensorNetworks: maxlinkdim
using NamedGraphs: nv
using NamedGraphs.GraphsExtensions: eccentricity
using NamedGraphs. NamedGraphGenerators: named_comb_tree, named_grid, named_binary_tree
using Random: Random

using NPZ
using MKL

using ITensorNumericalAnalysis

include("mutualinfo.jl")

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
    nsamples, max_z = 100, parse(Int64, last(mode))
    mi = generate_mi_matrix(f, nsamples, L, map_dimension)
    g = minimize_me(mi; max_z, alpha = 1)
    return continuous_siteinds(g, [[(i,1) for i in 1:L]])
  end
end

function get_function(mode::String)
  if mode == "Weirstrass"
    nterms, a = 25, 3
    ks = weirstrass_coefficients(nterms, a)
    eval_function = x -> calulate_weirstrass(x, ks)
    return eval_function, nterms, ks
  elseif mode == "Bessel"
    nterms, a, k = 20, 1, 25
    cs = bessel_coefficients(nterms, a, k)
    eval_function = x -> evaluate_polynomial(x, cs)
    return eval_function, nterms, cs
  elseif mode == "Laguerre"
    nterms = 40
    cs = laguerre_coefficients(nterms)
    eval_function = x -> evaluate_polynomial(x, cs)
    return eval_function, nterms, cs
  end
end

function construct_itn(s::IndsNetworkMap, mode::String, args...)
  if mode == "Weirstrass"
    return weirstrass_itn(s, args...)
  elseif mode == "Bessel"
    return poly_itn(s, args...)
  elseif mode == "Laguerre"
    return poly_itn(s, args...)
  end
end

function main(; md = nothing, func = nothing, l = nothing, save = true)
  mode = md == nothing ? ARGS[1] : md
  function_mode = func == nothing ? ARGS[2] : func
  L = l == nothing ? parse(Int64, ARGS[3]) : l
  map_dimension = 1
  eval_function, nterms, args = get_function(function_mode)
  s = siteinds_constructor(mode, L; map_dimension, f = x -> eval_function(only(x)))
  alg = maximum([degree(s, v) for v in vertices(s)]) >= 4 ? "bp" : "ttn"
  L, Lx = nv(s), nv(s)

  fx_exact = construct_itn(s, function_mode, args)
  z_fx_exact = inner(fx_exact, fx_exact; alg)
  χmax = maxlinkdim(fx_exact)
  println("Function built with χmax = $χmax")

  epsilons = [i for i in 1:30]
  no_eps = length(epsilons)

  memory_req = zeros(Int64, (no_eps))
  bond_dims = zeros(Int64, (no_eps))
  overlaps = zeros(Float64, (no_eps))
  ngrid_points = 100
  delta = 2.0^(-Lx)
  grid_points = Float64[delta * Random.rand(1:(2^Lx-1)) for i in 1:ngrid_points]
  exact_vals = Float64[real(eval_function(x)) for x in grid_points]
  trunc_vals = zeros(Float64, (no_eps, ngrid_points))
  l2_errors = zeros(Float64, (no_eps))


  for (i, eps) in enumerate(epsilons)
    println("Truncating down to eps = $(10^(-Float64(eps)))")
    fx_trunc = truncate(fx_exact; cutoff = 10^(-Float64(eps)))
    z_trunc = inner(fx_trunc, fx_trunc; alg)

    f= inner(fx_exact, fx_trunc; alg)

    err = (f * conj(f)) / (z_fx_exact*z_trunc)
    overlaps[i] = real(err)
    memory_req[i] = no_elements(fx_trunc)
    bond_dims[i] = maxlinkdim(fx_trunc)
    trunc_vals[i, :] = Float64[real(evaluate(fx_trunc, x)) for x in grid_points]
    l2_errors[i] = calc_error(exact_vals, trunc_vals[i, :])
    println("Achieved an overlap error of $(1.0 - overlaps[i])")
    println("Memory req was $(memory_req[i])")
    println("Bond Dim was $(bond_dims[i])")
    println("Error val is $(l2_errors[i])")
    flush(stdout)
  end

  println("Function evaluated")

  file_root = "/mnt/home/jtindall/Documents/Data/ITensorNumericalAnalysis/1DFunctionCompression/"
  file_name = file_root * "L"*string(L)*"GRAPH"*mode*"FUNCTION"*function_mode*"NTERMS"*string(nterms)*".npz"
  if save
    npzwrite(file_name,L = L, memory_req = memory_req, overlaps = overlaps, bond_dims = bond_dims, epsilons = epsilons, grid_points = grid_points, exact_vals = exact_vals, trunc_vals = trunc_vals, l2_errors = l2_errors)
  end
end

#main()
#main(; func = "Weirstrass", md = "CanonicalPath", l = 61, save = true)
#main(; func = "Weirstrass", md = "OrderedStar4", l = 61, save = true)
main(; func = "Weirstrass", md = "MISearch2", l = 61, save = true)
#main(; func = "Weirstrass", md = "OrderedStar4", l = 61, save = true)
#main(; func = "Weirstrass", md = "MISearch4", l = 61, save = true)
# main(; func = "Weirstrass", md = "MISearch2", l = 33, save = true)
# main(; func = "Weirstrass", md = "MISearch3", l = 33, save = true)
#main(; func = "Weirstrass", md = "MISearch5", l = 61, save = true)
# main(; func = "Weirstrass", md = "MISearch5", l = 33, save = true)


