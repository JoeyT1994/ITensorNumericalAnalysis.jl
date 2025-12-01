include("commonnetworks.jl")
include("commonfunctions.jl")
include("laughlinfunctions.jl")
include("utils.jl")
include("mutualinfo.jl")

using ITensorNetworks: maxlinkdim
using NamedGraphs.GraphsExtensions: add_edges, nv, eccentricity, disjoint_union, degree
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_grid, named_binary_tree
using Random: Random, rand
using LinearAlgebra: diagind, diagm, det, BLAS
using NPZ
using MKL
using Distributions: Uniform, LKJ

using ITensorNumericalAnalysis: interpolate, integrate

using Base.Threads

BLAS.set_num_threads(min(64, Sys.CPU_THREADS))
println("Julia is using "*string(nthreads()))
println("BLAS is using "*string(BLAS.get_num_threads()))
@show BLAS.get_config()

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

function get_function(mode::String; η)
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
  elseif mode[1:(length(mode)-1)] == "CentredGaussian"
    nterms = 1
    ndims = parse(Int64, last(mode))
    M = rand(LKJ(ndims, η))
    @show M
    k = 10
    scale_fac = ((k^ndims)/sqrt((2*pi)^ndims * det(M)))
    eval_function = x -> scale_fac * exp(-0.5*k*k*(x .- 0.5)' * inv(M) * (x .- 0.5))
    return eval_function, nterms, (;)
  end
end

function main(; eta = nothing, md = nothing, func = nothing, l = nothing, chi = nothing, nsweeps = 10, save = true, dn = nothing)
  mode = md == nothing ? ARGS[1] : md
  function_mode = func == nothing ? ARGS[2] : func
  L = l == nothing ? parse(Int64, ARGS[3]) : l
  χ = chi == nothing ? parse(Int64, ARGS[4]) : chi
  dis_no = dn == nothing ? parse(Int64, ARGS[5]) : dn
  η = eta == nothing ? parse(Int64, ARGS[6]) : eta
  map_dimension = parse(Int64, last(function_mode))
  Random.seed!(dis_no*183 + 54)
  eval_function, _, _ = get_function(function_mode; η)
  s = siteinds_constructor(mode, L; map_dimension, f = eval_function)
  vertices_dict = Dictionary(collect(vertices(s)), [(vertex_dimension(s, v), vertex_digit(s,v)) for v in collect(vertices(s))])
  f = input -> eval_function(calculate_point(vertices_dict, input; ndim = map_dimension))
  println("Graph is "*mode*" chi is $χ")

  fxyz, info = interpolate(f, s; maxdim = χ, nsweeps,cutoff = 1e-32, outputlevel=1)
  inf_norms = info[:, :error]
  sweeps = info[:,  :sweep]

  Lx = length(dimension_vertices(fxyz, 1))
  delta = (2^(-1.0*Lx))

  χmax = maxlinkdim(fxyz)
  println("Function built with χmax = $χmax")


  ngrid_points = 1000
  delta = 2.0^(-Lx)
  grid_points = zeros(Float64, (ngrid_points, map_dimension))
  for i in 1:ngrid_points
    grid_points[i, :] = [delta * Random.rand(1:(2^Lx-1)) for d in 1:map_dimension]
  end
  exact_vals = Float64[real(eval_function(grid_points[i, :])) for i in 1:ngrid_points]
  @show sum(exact_vals) / length(exact_vals)
  @show integrate(fxyz)
  trunc_vals = Float64[real(evaluate(fxyz, grid_points[i, :])) for i in 1:ngrid_points]
  error = calc_error(exact_vals, trunc_vals)
  memory_req = no_elements(fxyz)

  println("Function constructed with an error of $error, and a memory req of $memory_req")

  file_root = "/mnt/home/jtindall/ceph/Data/ITensorNumericalAnalysis/TCI/MultiD/"
  file_name = file_root * "L"*string(L)*"GRAPH"*mode*"FUNCTION"*function_mode*"Eta"*string(η)*"CHI"*string(χ)*"NGRIDPOINTS"*string(ngrid_points)*"nsweeps"*string(nsweeps)*"DisNo"*string(dis_no)*".npz"
  if save
    npzwrite(file_name, grid_points =grid_points, exact_vals = exact_vals, L = L, memory_req = memory_req, error = error,trunc_vals = trunc_vals, inf_norms = inf_norms, sweeps = sweeps)
  end
end

main()