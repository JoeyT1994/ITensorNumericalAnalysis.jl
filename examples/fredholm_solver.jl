using ITensorNumericalAnalysis

using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: NamedGraph, rename_vertices
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree
using ITensors:
  ITensors,
  ITensor,
  Index,
  siteinds,
  dim,
  tags,
  replaceprime!,
  MPO,
  MPS,
  inner,
  Op,
  @OpName_str,
  @SiteType_str,
  op
using ITensorNetworks: ITensorNetwork, dmrg, ttn, maxlinkdim, siteinds, union_all_inds
using Dictionaries: Dictionary
using Random: seed!

using UnicodePlots

function ITensors.op(::OpName"HalfInt", ::SiteType"Digit", s::Index)
  return ITensor(0.5, s, s') # |[1/2 1/2]> <δ(s')|
end

seed!(1234)
L = 60
g = named_comb_tree((2, L ÷ 2))

println(
  "########## Iteratively solve a inhomogeneous Fredholm equation of the second kind ##########",
)
println("solve f(x) = eˣ + ∫₀¹ (xy) f(y) dy")
# solution: f(x) = 3x/2 + eˣ

# start f(x) = f(x)⊗1_y
# 1. make g(x,y)
# 2. f*g 
# 3. apply operator I or |x>
# 4. apply shift if any

s = continuous_siteinds(g; map_dimension=2)
ψ = const_itn(s) # f(x) = 1_x⊗1_y
# make g(x,y) = x*y
g = poly_itn(s, [0, 1]; dimension=1) * poly_itn(s, [0, 1]; dimension=2)

sU = union_all_inds(s.indsnetwork, s.indsnetwork')
∫_odd = ITensorNetwork(v -> v ∈ dimension_vertices(s, 1) ? Op("I") : Op("HalfInt"), sU)
∫_even = ITensorNetwork(v -> v ∈ dimension_vertices(s, 1) ? Op("HalfInt") : Op("I"), sU)
const_exp_odd = exp_itn(s; dimension=1)
const_exp_even = exp_itn(s; dimension=2)

niter = 20
for iter in 1:niter
  global ψ = ψ * g

  local O = (iter % 2 == 1) ? ∫_even : ∫_odd
  local c = (iter % 2 == 1) ? const_exp_even : const_exp_odd

  global ψ = operate(O, ψ) + c
  #
end

n_grid = 40
x_vals = grid_points(s, n_grid, 1)
y = 0.5 # arbitrary
cutvals = zeros(length(x_vals))
for (i, x) in enumerate(x_vals)
  cutvals[i] = real(calculate_fxyz(ψ, [x, y]; alg="bp")) # alg="exact" if you encounter NaNs
end

correct = (3 / 2 * x_vals) .+ exp.(x_vals)
lp = lineplot(x_vals, cutvals; name="≈f(x) (cut at y=$y)")
lineplot!(lp, x_vals, correct; name="correct f(x) = 3x/2 + eˣ")
display(lp)

avg_err = sum(abs.(correct .- cutvals)) / n_grid
@show avg_err
