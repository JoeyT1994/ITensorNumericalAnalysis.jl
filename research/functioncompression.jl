include("commonnetworks.jl")
include("commonfunctions.jl")
include("utils.jl")

using Plots
using ITensorNetworks: maxlinkdim
using NamedGraphs: nv, named_comb_tree, named_binary_tree, eccentricity
using Random: Random

Random.seed!(1234)

function siteinds_constructor(mode::String, L::Int64; map_dimension = 1)
  if mode == "CanoncialPath"
    return qtt_siteinds_canonical(L)
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
    ks = weirstrass_coefficients(nterms, a = 20, 3)
    eval_function = x -> calulate_weirstrass(x, ks)
    fx = weirstrass_itn(s, ks)
    return fx, eval_function
  elseif mode == "Bessel"
    nterms, a, k = 20, 1, 25
    cs = bessel_coefficients(nterms, a, k)
    eval_function = x -> evaluate_polynomial(x, cs)
    fx = poly_itn(s, cs)
    return fx, eval_function
  elseif mode == "Laguerre"
    nterms = 40
    cs = laguerre_coefficients(nterms)
    eval_function = x -> evaluate_polynomial(x, cs)
    fx = poly_itn(s, cs)
    return fx, eval_function
  end
end

mode = "OrderedStar4"
function_mode = "Laguerre"
L = 41
χ = 41
s = siteinds_constructor(mode, L)
ngrid_points = 1000

fx, eval_function = construct_itn(s, function_mode)
gx = truncate(fx; maxdim=χ)

xs = grid_points(fx, ngrid_points, 1)
fx_xs, gx_xs, fx_xs_exact = [], [], []
xs = xs[2:length(xs)]
for x in xs
  append!(fx_xs, real(calculate_fx(fx, x)))
  append!(gx_xs, real(calculate_fx(gx, x)))
  append!(fx_xs_exact, eval_function(x))
end

err_nocutoff = calc_error(fx_xs_exact, fx_xs)
err_cutoff = calc_error(fx_xs_exact, gx_xs)
@show err_nocutoff, err_cutoff, no_elements(gx)

plt = plot(xs, fx_xs_exact; label = "Exact")
plot!(plt, xs, gx_xs; label="Truncated, chi = $χ")
