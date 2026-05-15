using NamedGraphs: vertextype, nv, vertices, edges, NamedGraph,rename_vertices, NamedEdge
using NamedGraphs.GraphsExtensions: add_edges, nv, eccentricity, disjoint_union, degree,  add_vertices!, add_edges!, add_edge!, add_vertex!, add_vertex
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_grid, named_binary_tree
using Random: Random, rand
using ITensors: dim, inds
using NPZ
using Adapt: adapt

using ITensorNumericalAnalysis
using TensorNetworkQuantumSimulator: graph, inner, tensornetwork, TensorNetworkQuantumSimulator, virtualinds

Random.seed!(1234)

function qtt_siteinds_canonical(L::Int64; map_dimension, is_complex = false)
    g = named_grid((L, 1))
    dimension_vertices = Vector{vertextype(g)}[]
    for d in 1:map_dimension
      vertices = [(i, 1) for i in d:map_dimension:L]
      push!(dimension_vertices, vertices)
    end
    s = is_complex ? continuous_siteinds(g, dimension_vertices, dimension_vertices; is_complex) : continuous_siteinds(g, dimension_vertices)
    return s, g
end
  
  function qtt_siteinds_canonical_sequentialdims(L::Int64; map_dimension, is_complex = false)
    g = named_grid((L, 1))
    dimension_vertices = Vector{vertextype(g)}[]
    dim_length = Int64(L/ map_dimension)
    for d in 1:map_dimension
      vertices = [(i, 1) for i in (1+(d-1)*dim_length):((d)*dim_length)]
      push!(dimension_vertices, vertices)
    end
    s = is_complex ? continuous_siteinds(g, dimension_vertices, dimension_vertices; is_complex) : continuous_siteinds(g, dimension_vertices)
    return s, g
end
  
function continuous_siteinds_ordered(g; map_dimension = 1, is_complex = false)
    sorted_vertices = sort(collect(vertices(g)); by = v -> eccentricity(g, v))
    L = length(sorted_vertices)
    dimension_vertices = Vector{vertextype(g)}[]
    for d in 1:map_dimension
        push!(dimension_vertices, sorted_vertices[d:map_dimension:L])
    end
    s = is_complex ? continuous_siteinds(g, dimension_vertices, dimension_vertices; is_complex) : continuous_siteinds(g, dimension_vertices)
    return s, g
end

function binary_tree_siteinds(k, map_dimension)
    g = NamedGraph()
    dimension_vertices = Vector{vertextype(g)}[]
    prev_parent = nothing
    for i in 1:map_dimension
        gi = named_binary_tree(k)
        gi = rename_vertices(v -> (v, i), gi)
        sorted_vertices_i = sort(collect(vertices(gi)); by = v -> eccentricity(gi, v))
        @assert first(sorted_vertices_i) == ((1, ), i)
        gi = add_vertex(gi, ((0, ), i))
        gi = add_edge!(gi, NamedEdge(((0,), i) => ((1,), i)))
        g = add_vertices!(g, collect(vertices(gi)))
        g = add_edges!(g, edges(gi))
        sorted_vertices_i = [[((0, ), i)]; sorted_vertices_i]
        @assert first(sorted_vertices_i) == ((0, ), i)
        if i > 1
            g = add_edge!(g, first(sorted_vertices_i) => prev_parent)
        end
        prev_parent = first(sorted_vertices_i)
        push!(dimension_vertices, sorted_vertices_i)
    end
    return continuous_siteinds(g, dimension_vertices), g
end

function siteinds_constructor(mode::String, L::Int64; k, map_dimension = 3, is_complex = false, f = nothing)
  if mode == "CanonicalPath"
    return qtt_siteinds_canonical(L; map_dimension, is_complex)
  elseif mode == "SequentialPath"
    return qtt_siteinds_canonical_sequentialdims(L; map_dimension, is_complex)
  elseif mode == "OrderedPath"
    return continuous_siteinds_ordered(named_grid((L,1)); map_dimension, is_complex)
  elseif mode[1:(length(mode)-1)] == "CombTree"
    backbonelength = parse(Int64, last(mode))
    comblength = round(Int, L / backbonelength)
    return continuous_siteinds_ordered(named_comb_tree((backbonelength, comblength)); map_dimension, is_complex)
  elseif mode == "BinaryTree"
    return binary_tree_siteinds(k, map_dimension)
  end
end

function get_function(mode::String)
  if mode == "RandPlaneWaves"
    nterms = 30
    As = [1.0 for i in 1:nterms]
    kxs = Float64[i*randn() for i in 1:nterms]
    kys = Float64[i*randn() for i in 1:nterms]
    kzs = Float64[i*randn() for i in 1:nterms]
    eval_function = (x, y, z) -> sum([As[i]*cos(kxs[i]*x + kys[i]*y + kzs[i]*z) for i in 1:nterms])
    return eval_function, nterms, (; nterms, As, kxs, kys, kzs)
  end
end

function construct_itn(s, g, mode::String; construction_params...)
  if mode == "RandPlaneWaves"
    fxy = build_random_planewaves(s, g; construction_params...)
    return fxy
  end
end

function build_random_planewaves(s, g; nterms, As, kxs, kys, kzs, dimension = 1)
    itns = [exp_tnf(g, s; k = 1.0im * kxs[i], c = As[i], dim = 1)*exp_tnf(g, s; k = 1.0im * kys[i], dim = 2)*exp_tnf(g, s; k = 1.0im * kzs[i], dim = 3) for i in 1:nterms]
    return reduce(+, reduce(vcat,itns))
end


function calc_error(exact_vals::Vector, approx_vals::Vector)
    @assert length(exact_vals) == length(approx_vals)
  
    eps = 0
    for (i, e) in enumerate(exact_vals)
      eps += abs((e - approx_vals[i]))
    end
    return eps / length(exact_vals)
  end
  
function no_elements(tn)
      no_elements = 0
      for v in vertices(tn)
          dims = dim.(inds(tn[v]))
          no_elements += prod(dims)
      end
      return no_elements
end

function main(; md = nothing, func = nothing, l = nothing, save = true)
  mode = md == nothing ? ARGS[1] : md
  function_mode = func == nothing ? ARGS[2] : func
  L = l == nothing ? parse(Int64, ARGS[3]) : l
  map_dimension = 3
  eval_function, nterms, construction_params = get_function(function_mode)
  kxs, kys, kzs = construction_params.kxs, construction_params.kys, construction_params.kzs
  s, g = siteinds_constructor(mode, L; k = 4, map_dimension, f = x -> eval_function(x[1], x[2], x[3]))
  L, Lx = nv(g), Int(nv(g) / map_dimension)
  delta = 2.0^(-Lx)
  Random.seed!(1234)
  trunc_alg = "ttn_svd"
  contract_alg = "bp"
  fxy_exact = construct_itn(s, g, function_mode; construction_params...)
  fxy_exact = adapt(Vector{ComplexF64})(fxy_exact)
  #z_fxy_exact = inner(tensornetwork(fxy_exact), fxy_exact; alg)
  χmax = TensorNetworkQuantumSimulator.maxvirtualdim(fxy_exact)
  println("Function built with χmax = $χmax")

  bond_dims = [χ for χ in χmax:-1:1]
  no_bds = length(bond_dims)

  memory_req = zeros(Int64, (no_bds))
  overlaps = zeros(Float64, (no_bds))
  ngrid_points = 1024
  delta = 2.0^(-Lx)
  grid_points = [[delta * Random.rand(1:(2^Lx-1)) for d in 1:map_dimension] for i in 1:ngrid_points]
  grid_points_x = Float64[(2^-10)*(i-1) for i in 1:ngrid_points]
  grid_points_y = Float64[(2^-10)*(i-1) for i in 1:ngrid_points]
  grid_points_z = Float64[(2^-10)*(i-1) for i in 1:ngrid_points]
  exact_vals = Float64[real(eval_function(Tuple(p)...)) for p in grid_points]
  trunc_vals = zeros(Float64, (no_bds, ngrid_points))
  exact_vals_x = Float64[real(eval_function(x, 0.5, 0.5)) for x in grid_points_x]
  exact_vals_y = Float64[real(eval_function(0.5, y, 0.5)) for y in grid_points_y]
  exact_vals_z = Float64[real(eval_function(0.5, 0.5, z)) for z in grid_points_z]
  trunc_vals_x = zeros(Float64, (no_bds, ngrid_points))
  trunc_vals_y = zeros(Float64, (no_bds, ngrid_points))
  trunc_vals_z = zeros(Float64, (no_bds, ngrid_points))
  l2_errors = zeros(Float64, (no_bds))

  for (i, χ) in enumerate(bond_dims)
    if χ <= χmax
        println("Truncating down to chi = $χ")
        fxy_trunc = truncate(fxy_exact; cutoff = 1e-28, maxdim = χ, alg = trunc_alg)
    else
        fxy_trunc = copy(fxy_exact)
    end
    fxy_trunc = adapt(Vector{ComplexF64})(fxy_trunc)
    #z_trunc = inner(tensornetwork(fxy_trunc), tensornetwork(fxy_trunc); alg)
    println("Evaluating function")

    #f = inner(fxy_exact, fxy_trunc; alg)
    #err= (f * conj(f)) / (z_trunc * z_fxy_exact)
    #overlaps[i] = real(err)
    memory_req[i] = no_elements(fxy_trunc)
    trunc_vals[i, :] = Float64[real(evaluate(fxy_trunc, p)) for p in grid_points]
    trunc_vals_x[i, :] = Float64[real(evaluate(fxy_trunc, [p, 0.5, 0.5])) for p in grid_points_x]
    trunc_vals_y[i, :] = Float64[real(evaluate(fxy_trunc, [0.5, p, 0.5])) for p in grid_points_y]
    trunc_vals_z[i, :] = Float64[real(evaluate(fxy_trunc, [0.5, 0.5, p])) for p in grid_points_z]
    l2_errors[i] = calc_error(exact_vals, trunc_vals[i, :])
    #println("Achieved an overlap error of $(1.0 - overlaps[i])")
    println("Memory req was $(memory_req[i])")
    println("Error val is $(l2_errors[i])")
    flush(stdout)
  end

  file_root = "/Users/jtindall/Files/Data/ITensorNumericalAnalysis/3DFunctionCompressionRevised/"
  file_name = file_root * "L"*string(L)*"GRAPH"*mode*"FUNCTION"*function_mode*"CHI"*string(χmax)*"NTERMS"*string(nterms)*".npz"
  npzwrite(file_name, L = L, kxs = kxs, kys= kys, kzs = kzs, grid_points = grid_points_x, memory_req = memory_req, bond_dims = bond_dims, l2_errors = l2_errors, exact_vals = exact_vals, trunc_vals = trunc_vals,
    exact_vals_x = exact_vals_x, exact_vals_y = exact_vals_y, exact_vals_z = exact_vals_z, trunc_vals_x = trunc_vals_x, trunc_vals_y = trunc_vals_y, trunc_vals_z = trunc_vals_z)
end

#main()
main(; func = "RandPlaneWaves", md = "CanonicalPath", l = 48, save = true)
main(; func = "RandPlaneWaves", md = "SequentialPath", l = 48, save = true)
main(; func = "RandPlaneWaves", md = "CombTree3", l = 48, save = true)
main(; func = "RandPlaneWaves", md = "BinaryTree", l = 48, save = true)


