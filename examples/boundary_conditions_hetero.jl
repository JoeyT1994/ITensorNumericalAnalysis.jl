using Test
using ITensorNumericalAnalysis

using Graphs: SimpleGraph, uniform_tree
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree
using ITensors: ITensors, Index, siteinds, dim, inner, OpSum
using ITensorNetworks: ITensorNetwork, dmrg, ttn, maxlinkdim
using Dictionaries: Dictionary
using Random: seed!

using UnicodePlots

seed!(1234)
L = 12
lastDigit = 1 - 1 / 2^(L ÷ 2)
#g = NamedGraph(SimpleGraph(uniform_tree(L)))
#g = named_grid((L, 1))
g = named_comb_tree((2, L ÷ 2))

s = continuous_siteinds(g; map_dimension=2)

ψ_fxy = cos_itn(s; k=π)
@show maxlinkdim(ψ_fxy)

Zo = zero_point_op(s, [0, lastDigit, 0, lastDigit], [1, 1, 2, 2])
@show maxlinkdim(Zo)

x_bcs = (2, 3)
y_bcs = (4, 5)
ttn_op = OpSum()
all_ops = []
bc_vals = [
  (x_bcs[1], 0.0, 1), (x_bcs[2], lastDigit, 1), (y_bcs[1], 0.0, 2), (y_bcs[2], lastDigit, 2)
]
for (val, p, dimension) in bc_vals
  b_op = ITensorNumericalAnalysis.point_to_opsum(s, p, dimension)
  push!(all_ops, b_op)
  global ttn_op += val * b_op
end
# if we have (∑P_x)(∑P_y) then
# we have an additional +P_x*P_y term
# this is equivelent to ∑ P_{all overlap spots}
# We allow the y BC to "win"
dims = [1, 1, 2, 2]
for i in 1:length(all_ops)
  v1, p1, d1 = bc_vals[i]
  for j in (i + 1):length(all_ops)
    v2, p2, d2 = bc_vals[j]
    b_op1, b_op2 = all_ops[i], all_ops[j]
    (d1 == d2) && continue
    global ttn_op += -1 * (v1) * Ops.expand(b_op1 * b_op2)
  end
end
Boundary_op = ttn(ttn_op, ITensorNumericalAnalysis.indsnetwork(s);)

maxdim = 34
cutoff = 0e-16 #0e-16
@show cutoff
ϕ_fxy = copy(ψ_fxy)
ϕ_fxy = operate(Zo, ϕ_fxy; cutoff, maxdim, normalize=false)
plane = operate(
  [Boundary_op], const_itn(s; c=1, linkdim=4); cutoff, maxdim, normalize=false
)
ϕ_fxy += plane

ϕ_fxy = truncate(ϕ_fxy; maxdim=10, cutoff=0)
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

@show vals2[1], vals2[end - 1], vals3[1], vals3[end - 1]
