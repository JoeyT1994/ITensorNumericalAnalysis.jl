include("commonnetworks.jl")
include("commonfunctions.jl")
include("utils.jl")

using ITensorNetworks: maxlinkdim
using NamedGraphs: add_edges, nv, named_comb_tree, named_grid, named_binary_tree, eccentricity
using NamedGraphs: disjoint_union
using Random: Random, rand

using NPZ
using MKL

using ITensorNumericalAnalysis
using ITensorNumericalAnalysis:dimension_indices

Random.seed!(1234)

function siteinds_constructor(mode::String, L::Int64; map_dimension = 3)
  if mode == "CanonicalPath"
    return qtt_siteinds_canonical(L; map_dimension)
  elseif mode == "SequentialPath"
    return qtt_siteinds_canonical_sequentialdims(L; map_dimension)
  elseif mode == "OrderedPath"
    return continuous_siteinds_ordered(named_grid((L,1)); map_dimension)
  elseif mode[1:(length(mode)-1)] == "OrderedStar"
    npoints = parse(Int64, last(mode))
    pointlength = Int64((L-1) / npoints)
    return continuous_siteinds_ordered(star(npoints, pointlength); map_dimension)
  elseif mode[1:(length(mode)-1)] == "MultiDimOrderedStar"
      npoints = parse(Int64, last(mode))
      return qtt_siteinds_multidimstar_ordered(L, npoints; map_dimension)
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
  if mode == "RandPlaneWaves"
    nterms = 30
    As = [1.0 for i in 1:nterms]
    kxs = [i*randn() for i in 1:nterms]
    kys = [i*randn() for i in 1:nterms]
    kzs = [i*randn() for i in 1:nterms]
    eval_function = (x, y, z) -> sum([As[i]*cos(kxs[i]*x + kys[i]*y + kzs[i]*z) for i in 1:nterms])
    fxy = build_random_planewaves(s, nterms, As, kxs, kys, kzs)
    return fxy, eval_function, nterms
  elseif mode == "SphericalLaplacian"
    coeffs = ComplexF64[-(3.0/16.0)*pi*pi, -(21.0/128)*pi*pi, 5.0/3.0, -(5.0/4.0)*pi, (315.0/256.0)*pi, -(105.0/256.0)*pi*pi]
    nterms = length(coeffs)
    fxy = build_spherical_laplacian_solution(s, coeffs)
    eval_function = (x,y,z) -> calculate_spherical_laplacian_solution(x,y,z,coeffs)
    return fxy, eval_function, nterms
  end
end

function main()
  mode = ARGS[1]
  function_mode = ARGS[2]
  L = parse(Int64, ARGS[3])
  #mode = "SequentialPath"
  #function_mode = "RandPlaneWaves"
  #L = 45
  map_dimension = 3
  s = siteinds_constructor(mode, L; map_dimension)
  ngrid_points = 1000
  L = nv(s)
  Random.seed!(1234)

  fxy, eval_function, nterms = construct_itn(s, function_mode)

  if function_mode == "SphericalLaplacian"
    if mode == "SequentialPath" || mode == "MultiDimOrderedStar1"
      fxy = truncate(fxy; maxdim = 10)
    elseif mode == "CanonicalPath"
      fxy = truncate(fxy; maxdim = 150)
    end
  end

  χmax = maxlinkdim(fxy)
  println("Function built with χmax = $χmax")

  Lx = length(dimension_vertices(fxy, 1))
  delta = (2^(-1.0*Lx))

  fxy_xys = zeros(Float64, (χmax, ngrid_points))
  fxy_xys_exact = zeros(Float64, (ngrid_points))
  memory_req = zeros(Int64, (χmax))
  errors = zeros(Float64, (χmax))
  grid_points = zeros(Float64, (ngrid_points, map_dimension))
  for i in 1:ngrid_points
    grid_points[i,:] = Float64[delta * Random.rand(1:(2^Lx-1)) for i in 1:map_dimension]
  end

  for χ in χmax:-1:1
    println("Truncating down to χ = $χ")
    fxy = truncate(fxy; maxdim=χ)
    println("Evaluating function")
    for i in 1:ngrid_points
      x, y, z = grid_points[i, 1], grid_points[i, 2], grid_points[i, 3]
      fxy_xys[χ, i] = real(calculate_fxyz(fxy, [x, y, z]))
      if χ == χ
        fxy_xys_exact[i] = eval_function(x, y, z)
      end
    end
    errors[χ] = calc_error(reduce(vcat, fxy_xys_exact), reduce(vcat, fxy_xys[χ, :]))
    memory_req[χ] = no_elements(fxy)
    println("Achieved an error of $(errors[χ])")
    flush(stdout)
  end

  file_root = "/mnt/home/jtindall/Documents/Data/ITensorNumericalAnalysis/3DFunctionCompression/"
  file_name = file_root * "L"*string(L)*"GRAPH"*mode*"FUNCTION"*function_mode*"CHI"*string(χmax)*"NGRIDPOINTS"*string(ngrid_points)*"NTERMS"*string(nterms)*".npz"
  npzwrite(file_name, grid_points = grid_points, fxy_xys = fxy_xys, fxy_xys_exact = fxy_xys_exact, L = L, memory_req = memory_req, errors = errors)
end

main()


