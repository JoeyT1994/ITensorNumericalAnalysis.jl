using Test
using ITensorNumericalAnalysis

using NamedGraphs.NamedGraphGenerators: named_comb_tree
using ITensors: ITensors, inner
using ITensorNetworks: ITensorNetwork, ttn, maxlinkdim
using Random: seed!
using ITensors: Ops

using UnicodePlots

seed!(1234)
L = 12
g = named_comb_tree((2, L ÷ 2))
lastDigit = 1 - 1 / 2^(L ÷ 2)

s = continuous_siteinds(g; map_dimension=2)

ψ_fxy = cos_itn(s; k=π)
#ψ_fxy = const_itn(s; c=3, linkdim=3) # note if you use const, need big linkdim
@show maxlinkdim(ψ_fxy)

# make an operator that applies 0 to the planes in certain dimensions
Zo = zero_point_op(s, [0, lastDigit, 0, lastDigit], [1, 1, 2, 2])
## for test later
#for p1 in [0,lastDigit]
#  for p2 in [0,lastDigit]
#    p = ttn(itensornetwork(delta_xyz(s,[p1,p2])))
#    @show p1,p2
#    @show inner(p',Zo,p)
#  end
#end
@show maxlinkdim(Zo)

maxdim = 10
cutoff = 0e-16
@show cutoff
ϕ_fxy = copy(ψ_fxy)
ϕ_fxy = operate([Zo], ϕ_fxy; cutoff, maxdim, normalize=false)
@show maxlinkdim(ϕ_fxy)

n_grid = 100
x_vals, y_vals = grid_points(s, n_grid, 1)[1:2:(end - 1)],
grid_points(s, n_grid, 2)[1:2:(end - 1)]
# fix for if we don't include all 1s
if x_vals[end] != lastDigit
  push!(x_vals, lastDigit)
end
if y_vals[end] != lastDigit
  push!(y_vals, lastDigit)
end
vals = zeros((length(x_vals), length(y_vals)))
for (i, x) in enumerate(x_vals)
  for (j, y) in enumerate(y_vals)
    vals[i, j] = real(calculate_fxyz(ϕ_fxy, [x, y]; alg="exact"))
  end
end

println("Here is the heatmap of the 2D function")
display(heatmap(vals; xfact=1 / 32, yfact=1 / 32, xoffset=0, yoffset=0, colormap=:inferno))

y = 0.5
vals2 = zeros(length(x_vals))
for (i, x) in enumerate(x_vals)
  vals2[i] = real(calculate_fxyz(ϕ_fxy, [x, y]; alg="exact"))
end

lp = lineplot(x_vals, vals2; name="cut y=$y")

x = 0.5
vals3 = zeros(length(y_vals))
for (i, y) in enumerate(y_vals)
  vals3[i] = real(calculate_fxyz(ϕ_fxy, [x, y]; alg="exact"))
end

println("Here is a cut of the function at x = $x or y = $y")
display(lineplot!(lp, y_vals, vals3; name="cut x=$x"))

@show vals2[1], vals2[end], vals3[1], vals3[end]
