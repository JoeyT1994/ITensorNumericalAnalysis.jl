using ITensorNumericalAnalysis

using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: NamedGraph, rename_vertices
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree
using ITensors:
  ITensors, ITensor, Index, siteinds, dim, tags, replaceprime!, MPO, MPS, inner, Op
using ITensorNetworks: ITensorNetwork, dmrg, ttn, maxlinkdim, siteinds, union_all_inds
using Dictionaries: Dictionary
using Random: seed!
using ITensors: Ops

using UnicodePlots

seed!(1234)
L = 60
#g = NamedGraph(SimpleGraph(uniform_tree(L)))
#g = rename_vertices(v -> (v, 1), g)
g = named_comb_tree((2, L ÷ 2))
#g = named_grid((L,1))
lastDigit = 1 - 1 / 2^(L ÷ 2)

s = continuous_siteinds(g; map_dimension=2)

ψ_fx = exp_itn(s; dimension=1)
ψ_fy = exp_itn(s; dimension=2)
ψ_fxy = ψ_fx * ψ_fy

ans = ITensorNumericalAnalysis.integrate(ψ_fxy)
correct = (exp(1) - 1)^2
@show ans, correct
@show ans - correct

# solve f(x) = x + ∫₀¹ (xy) f(y) dy
# ans = 3/2x

# start f(x) = f(x)⊗1_y
# g(x,y) = xy
# 1. make g(x,y)
# 2. f*g
# 3. apply operator I or |x>
ψ = const_itn(s) # f(x) = 1_x⊗1_y
# make g(x,y) = x*y
g = poly_itn(s, [0, 1]; dimension=1) * poly_itn(s, [0, 1]; dimension=2)

s2 = union_all_inds(s.indsnetwork, s.indsnetwork')
Oo = ITensorNetwork(v -> v ∈ dimension_vertices(s, 1) ? Op("I") : Op("Int"), s2)
Oe = ITensorNetwork(v -> v ∈ dimension_vertices(s, 1) ? Op("Int") : Op("I"), s2)
const_expo = exp_itn(s; dimension=1) #poly_itn(s,[0,1]; dimension=1)
const_expe = exp_itn(s; dimension=2) #poly_itn(s,[0,1]; dimension=1)

niter = 100
for iter in 1:niter
  global ψ = ψ * g

  local O = (iter % 2 == 1) ? Oe : Oo
  local c = (iter % 2 == 1) ? const_expe : const_expo

  global ψ = operate(O, ψ) + c
  #
end

n_grid = 40
x_vals, y_vals = grid_points(s, n_grid, 1)[1:(end - 1)],
grid_points(s, n_grid, 2)[1:(end - 1)]
y = 0.5
vals2 = zeros(length(x_vals))
for (i, x) in enumerate(x_vals)
  vals2[i] = real(calculate_fxyz(ψ, [x, y]; alg="exact"))
end

correct = (3 / 2 * x_vals) .+ exp.(x_vals)
lp = lineplot(x_vals, vals2; name="cut y=$y")
lineplot!(lp, x_vals, correct; name="correct")
show(lp)
@show sum(abs.(correct .- vals2)) / n_grid

# Alternative interface if you have an operator|state> to integrate directly

#s = continuous_siteinds(g; map_dimension=1)
#
##ψ_fxy = const_itn(s; c=3, linkdim=2)
#ψ_fx = exp_itn(s; dimension=1)
#O = operator(ψ_fx)
#correct = 1/2*(-1 + exp(1)^2)
#ans= ITensorNumericalAnalysis.integrate(O,ψ_fx)
#@show ans,correct
#@show ans-correct
