using Test
using ITensorNumericalAnalysis

using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: NamedGraph, named_grid
using ITensors: ITensors, Index, siteinds, dim, tags, replaceprime!, MPO, MPS, inner
using ITensorNetworks: ITensorNetwork, dmrg, ttn, maxlinkdim
using Dictionaries: Dictionary
using Random: seed!

using UnicodePlots

seed!(1234)
L = 12
lastDigit = 1 - 1 / 2^(L ÷ 2)
#g = NamedGraph(SimpleGraph(uniform_tree(L)))
g = named_grid((L, 1))

s = continuous_siteinds(g; map_dimension=2)

ψ_fxy = sin_itn(s; k=10) #const_itn(s; c=1, linkdim=2)
#ψ_sin = sin_itn(s)
#ψ_fxy = delta_xyz(s, [0.0,0.5] )
@show maxlinkdim(ψ_fxy)

Zero_X = zero_point_op(s, [0, lastDigit], 1)
Zero_Y = zero_point_op(s, [0, lastDigit], 2)
Zero_X = truncate(Zero_X; cutoff=1e-14)
Zero_Y = truncate(Zero_Y; cutoff=1e-14)
@show maxlinkdim(Zero_X), maxlinkdim(Zero_Y)

x_bcs = (2, 3)
y_bcs = (4, 5)
Cop_x0 = const_plane_op(s, [0.0], 1)
Cop_x1 = const_plane_op(s, [lastDigit], 1)
Cop_y0 = const_plane_op(s, [0.0], 2)
Cop_y1 = const_plane_op(s, [lastDigit], 2)
@show maxlinkdim(Cop_x0), maxlinkdim(Cop_y0)

maxdim = 34
cutoff = 0e-16 #0e-16
@show cutoff
ϕ_fxy = copy(ψ_fxy)
ϕ_fxy = operate([Zero_X, Zero_Y], ϕ_fxy; cutoff, maxdim, normalize=false)
plane = operate(
  [Cop_x0],
  const_itn(s; c=x_bcs[1], linkdim=2);
  cutoff,
  maxdim,
  normalize=false,
  #[Cop_x0], sin_itn(s;); cutoff, maxdim, normalize=false
)
ϕ_fxy += plane
plane = operate(
  [Cop_x1], -1 * const_itn(s; c=x_bcs[2], linkdim=2); cutoff, maxdim, normalize=false
)
ϕ_fxy += plane

plane = operate(
  [Cop_y0], const_itn(s; c=y_bcs[1], linkdim=2); cutoff, maxdim, normalize=false
)
ϕ_fxy += plane
plane = operate(
  [Cop_y1], -1 * const_itn(s; c=y_bcs[2], linkdim=2); cutoff, maxdim, normalize=false
)
ϕ_fxy += plane

ϕ_fxy = truncate(ϕ_fxy; maxdim=10, cutoff=0)
@show maxlinkdim(ϕ_fxy)

n_grid = 50
x_vals, y_vals = grid_points(s, n_grid, 1), grid_points(s, n_grid, 2)
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
show(heatmap(vals; xfact=0.01, yfact=0.01, xoffset=0, yoffset=0, colormap=:inferno))

#y_vals = grid_points(s, n_grid, 2)
x = 0.0
vals3 = zeros(length(y_vals))
for (i, y) in enumerate(y_vals)
  vals3[i] = real(calculate_fxyz(ϕ_fxy, [x, y]; alg="exact"))
end

lp = (lineplot(y_vals, vals3; name="cut x=$x"))

#n_grid = 50
#x_vals = grid_points(s, n_grid, 1)
y = 1 - 1 / 2^L #0.5
vals2 = zeros(length(x_vals))
for (i, x) in enumerate(x_vals)
  vals2[i] = real(calculate_fxyz(ϕ_fxy, [x, y]; alg="exact"))
end

println("Here is a cut of the function at x = $x or y = $y")
show(lineplot!(lp, x_vals, vals2; name="cut y=$y"))

@show vals2[1], vals2[end], vals3[1], vals3[end]
