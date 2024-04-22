include("commonnetworks.jl")
include("commonfunctions.jl")
include("utils.jl")

using Plots
using ITensorNetworks: maxlinkdim
using NamedGraphs: nv, named_comb_tree
using Random: Random

Random.seed!(1234)

s1 = star_siteinds(5,8)
L = nv(s1)
s2 = qtt_siteinds(L)

χ1, χ2  = 2, 4
nterms, a = 10, 3.0
ks = weirstrass_coefficients_V2(nterms, a)
fx_1 = weirstrass_itn_V2(s1, ks)
gx_1 = truncate(fx_1; maxdim=χ1)

fx_2 = weirstrass_itn_V2(s2, ks)
gx_2 = truncate(fx_2; maxdim=χ2)

xs = grid_points(fx_1, 250, 1)
fx1_xs, fx2_xs, gx1_xs, gx2_xs = [], [], [], []
fx_xs_exact = []
xs = xs[2:length(xs)]
for x in xs
  append!(fx1_xs, real(calculate_fx(fx_1, x)))
  append!(gx1_xs, real(calculate_fx(gx_1, x)))
  append!(fx2_xs, real(calculate_fx(fx_2, x)))
  append!(gx2_xs, real(calculate_fx(gx_2, x)))
  append!(fx_xs_exact, calulate_weirstrass_V2(x, ks))
end

err1_nocutoff = calc_error(fx_xs_exact, fx1_xs)
err1_cutoff = calc_error(fx_xs_exact, gx1_xs)
@show err1_nocutoff, err1_cutoff, no_elements(gx_1)

err2_nocutoff = calc_error(fx_xs_exact, fx2_xs)
err2_cutoff = calc_error(fx_xs_exact, gx2_xs)
@show err2_nocutoff, err2_cutoff, no_elements(gx_2)

plt = plot(xs, abs(fx_xs_exact - gx1_xs); label = "Exact")
plot!(plt, xs, gx1_xs; label="Truncated Star, chi = $χ1")
plot!(plt, xs, gx2_xs; label="Truncated MPS, chi = $χ2")
#plot!(plt, xs, fx_xs_exact, name = "Exact")
