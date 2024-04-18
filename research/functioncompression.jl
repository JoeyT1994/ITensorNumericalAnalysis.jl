include("commonnetworks.jl")
include("commonfunctions.jl")

using UnicodePlots
using ITensorNetworks: maxlinkdim

L = 30
s = qtt_siteinds(L)

alpha, nterms = 5, 30
bc = bessel_coefficients(alpha, nterms; k = 25)

fx = poly_itn(s, bc)
gx = truncate(fx; cutoff = 1e-1)
@show maxlinkdim(fx)
@show maxlinkdim(gx)
xs = grid_points(fx, 100, 1)
fx_xs, gx_xs = [], []
for x in xs
    append!(fx_xs, calculate_fx(fx, x))
    append!(gx_xs, calculate_fx(gx, x))
end

plt=  lineplot(xs, fx_xs)
lineplot!(plt, xs, gx_xs, name = "Truncated")
