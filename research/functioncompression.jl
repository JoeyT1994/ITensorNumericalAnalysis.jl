include("commonnetworks.jl")
include("commonfunctions.jl")
include("utils.jl")

using Plots
using ITensorNetworks: maxlinkdim
using NamedGraphs: nv, named_comb_tree
using Random: Random

Random.seed!(1234)

s = continuous_siteinds(named_comb_tree((6, 6)))
s = star_siteinds(3, 10)
L = nv(s)
#s = qtt_siteinds(L)

χ = 3
nterms, a, b = 20, 0.75, 3.0
cs, ks = weirstrass_coefficients(nterms, a, b)
fx = weirstrass_itn(s, cs, ks)
gx = truncate(fx; maxdim=χ)
@show maxlinkdim(fx)
@show maxlinkdim(gx)
xs = grid_points(fx, 500, 1)
fx_xs, gx_xs = [], []
fx_xs_exact = []
xs = xs[2:length(xs)]
for x in xs
  append!(fx_xs, real(calculate_fx(fx, x)))
  append!(gx_xs, real(calculate_fx(gx, x)))
  append!(fx_xs_exact, calulate_weirstrass(x, cs, ks))
end

err_nocutoff = calc_error(fx_xs_exact, fx_xs)
err_cutoff = calc_error(fx_xs_exact, gx_xs)
@show err_nocutoff, err_cutoff

plt = plot(xs, fx_xs)
plot!(plt, xs, gx_xs; name="Truncated")
#plot!(plt, xs, fx_xs_exact, name = "Exact")
