include("commonnetworks.jl")
include("commonfunctions.jl")
include("utils.jl")

using ITensorNetworks: maxlinkdim
using NamedGraphs: nv, named_comb_tree, named_grid, named_binary_tree, eccentricity
using Random: Random

using NPZ
using MKL

using ITensorNumericalAnalysis

Random.seed!(1234)

function siteinds_constructor(mode::String, L::Int64; map_dimension = 1)
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
  end
end

function construct_itn(s::IndsNetworkMap, mode::String)
  if mode == "Weirstrass"
    nterms, a = 20, 3
    ks = weirstrass_coefficients(nterms, a)
    eval_function = x -> calulate_weirstrass(x, ks)
    fx = weirstrass_itn(s, ks)
    return fx, eval_function, nterms
  elseif mode == "Bessel"
    nterms, a, k = 20, 1, 25
    cs = bessel_coefficients(nterms, a, k)
    eval_function = x -> evaluate_polynomial(x, cs)
    fx = poly_itn(s, cs)
    return fx, eval_function, nterms
  elseif mode == "Laguerre"
    nterms = 40
    cs = laguerre_coefficients(nterms)
    eval_function = x -> evaluate_polynomial(x, cs)
    fx = poly_itn(s, cs)
    return fx, eval_function, nterms
  end
end

function main()
  mode = ARGS[1]
  function_mode = ARGS[2]
  L = parse(Int64, ARGS[3])
  map_dimension = 1
  s = siteinds_constructor(mode, L; map_dimension)
  ngrid_points = 1000
  L = nv(s)

  fx, eval_function, nterms = construct_itn(s, function_mode)
  χmax = maxlinkdim(fx)
  println("Function built with χmax = $χmax")

  Lx = length(dimension_vertices(fx, 1))
  delta = (2^(-1.0*Lx))

  fx_xs = zeros(Float64, (χmax, ngrid_points))
  memory_req = zeros(Int64, (χmax))
  errors = zeros(Float64, (χmax))
  fx_xs_exact =zeros(Float64, (ngrid_points))
  grid_points = zeros(Float64, (ngrid_points, map_dimension))
  for i in 1:ngrid_points
    grid_points[i,:] = Float64[delta * Random.rand(1:(2^Lx-1)) for i in 1:map_dimension]
  end

  for χ in χmax:-1:1
    println("Truncating down to χ = $χ")
    fx = truncate(fx; maxdim=χ)
    println("Evaluating function")

    for i in 1:ngrid_points
      x = grid_points[i, 1]
      fx_xs[χ, i] = real(calculate_fx(fx, x))
      if χ == χ
        fx_xs_exact[i] = eval_function(x)
      end
    end

    errors[χ] = calc_error(reduce(vcat, fx_xs_exact), reduce(vcat, fx_xs[χ, :]))
    memory_req[χ] = no_elements(fx)
    println("Achieved an error of $(errors[χ])")
    flush(stdout)
  end

  println("Function evaluated")

  file_root = "/mnt/home/jtindall/Documents/Data/ITensorNumericalAnalysis/1DFunctionCompression/"
  file_name = file_root * "L"*string(L)*"GRAPH"*mode*"FUNCTION"*function_mode*"CHI"*string(χmax)*"NGRIDPOINTS"*string(ngrid_points)*"NTERMS"*string(nterms)*".npz"
  npzwrite(file_name, grid_points = grid_points, fx_xs = fx_xs, fx_xs_exact = fx_xs_exact, L = L, memory_req = memory_req, errors = errors)
end

main()


