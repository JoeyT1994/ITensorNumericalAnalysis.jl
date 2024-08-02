include("commonnetworks.jl")
include("commonfunctions.jl")
include("laughlinfunctions.jl")
include("utils.jl")
include("mutualinfo.jl")

using ITensorNetworks: maxlinkdim
using NamedGraphs.GraphsExtensions: add_edges, nv, eccentricity, disjoint_union
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_grid, named_binary_tree
using Random: Random, rand

using NPZ
using MKL

using ITensorNumericalAnalysis

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
    e = generate_entanglements(f, nsamples, round(Int, L / max_z), map_dimension)
    g = minimize_entanglement(e; max_z, alpha = 1)
    return continuous_siteinds(g, [[(i,j) for i in 1:round(Int, L / max_z)] for j in 1:map_dimension])
  end
end

function get_function(mode::String)
  if mode == "RandPlaneWaves"
    nterms = 30
    As = [1.0 for i in 1:nterms]
    kxs = [i*randn() for i in 1:nterms]
    kys = [i*randn() for i in 1:nterms]
    kzs = [i*randn() for i in 1:nterms]
    eval_function = (x, y, z) -> sum([As[i]*cos(kxs[i]*x + kys[i]*y + kzs[i]*z) for i in 1:nterms])
    return eval_function, nterms, (; nterms, As, kxs, kys, kzs)
  end
end

function construct_itn(s::IndsNetworkMap, mode::String; construction_params...)
  if mode == "RandPlaneWaves"
    fxy = build_random_planewaves(s; construction_params...)
    return fxy
  end
end

function main(; md = nothing, func = nothing, l = nothing, save = true)
  mode = md == nothing ? ARGS[1] : md
  function_mode = func == nothing ? ARGS[2] : func
  L = l == nothing ? parse(Int64, ARGS[3]) : l
  map_dimension = 3
  eval_function, nterms, construction_params = get_function(function_mode)
  s = siteinds_constructor(mode, L; map_dimension, f = x -> eval_function(x[1], x[2], x[3]))
  L, Lx = nv(s), Int(nv(s) / map_dimension)
  delta = 2.0^(-Lx)
  Random.seed!(1234)
  alg = maximum([degree(s, v) for v in vertices(s)]) >= 4 ? "bp" : "ttn"
  fxy_exact = construct_itn(s, function_mode; construction_params...)
  z_fxy_exact = inner(fxy_exact, fxy_exact; alg)
  χmax = maxlinkdim(fxy_exact)
  println("Function built with χmax = $χmax")

  epsilons = [i for i in 1:30]
  no_eps = length(epsilons)
  ngrid_points = 100

  memory_req = zeros(Int64, (no_eps))
  bond_dims = zeros(Int64, (no_eps))
  overlaps = zeros(Float64, (no_eps))
  grid_points = Vector{Float64}[[delta * Random.rand(1:(2^Lx-1)) for d in 1:map_dimension] for i in 1:ngrid_points]
  exact_vals = Float64[real(eval_function(Tuple(p)...)) for p in grid_points]
  trunc_vals = zeros(Float64, (no_eps, ngrid_points))
  l2_errors = zeros(Float64, (no_eps))

  for (i, eps) in enumerate(epsilons)
    println("Truncating down to eps = $(10^(-Float64(eps)))")
    fxy_trunc = truncate(fxy_exact; cutoff = 10^(-Float64(eps)))
    z_trunc = inner(fxy_trunc, fxy_trunc; alg)
    println("Evaluating function")

    f = inner(fxy_exact, fxy_trunc; alg)
    err= (f * conj(f)) / (z_trunc * z_fxy_exact)
    overlaps[i] = real(err)
    memory_req[i] = no_elements(fxy_trunc)
    bond_dims[i] = maxlinkdim(fxy_trunc)
    trunc_vals[i, :] = Float64[real(evaluate(fxy_trunc, p)) for p in grid_points]
    l2_errors[i] = calc_error(exact_vals, trunc_vals[i, :])
    println("Achieved an overlap error of $(1.0 - overlaps[i])")
    println("Memory req was $(memory_req[i])")
    println("Bond Dim was $(bond_dims[i])")
    println("Error val is $(l2_errors[i])")
    flush(stdout)
  end

  file_root = "/mnt/home/jtindall/Documents/Data/ITensorNumericalAnalysis/3DFunctionCompression/"
  file_name = file_root * "L"*string(L)*"GRAPH"*mode*"FUNCTION"*function_mode*"CHI"*string(χmax)*"NTERMS"*string(nterms)*".npz"
  npzwrite(file_name, L = L, memory_req = memory_req, overlaps = overlaps, epsilons = epsilons, bond_dims = bond_dims)
end

#main()
#main(; func = "RandPlaneWaves", md = "CanonicalPath", l = 45, save = true)
#main(; func = "RandPlaneWaves", md = "SequentialPath", l = 45, save = true)
main(; func = "RandPlaneWaves", md = "MISearch3", l = 45, save = true)
#main(; func = "RandPlaneWaves", md = "CombTree3", l = 45, save = true)


