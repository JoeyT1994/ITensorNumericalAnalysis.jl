using ITensorNumericalAnalysis

using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: NamedGraph, NamedEdge, rename_vertices, edges, vertices
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

using ITensorNumericalAnalysis: partial_integrate, reduced_indsnetworkmap

seed!(1234)
L = 100
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

s = continuous_siteinds(g, [[(i, j) for j in 1:(L÷2)] for i in 1:2])
dim_ψ = 2
s1, s2 = reduced_indsnetworkmap(s, 1), reduced_indsnetworkmap(s, 2)

ψ = const_itn(s2) # f(x) = 1_x⊗1_y

# make g(x,y) = x*y
g = poly_itn(s1, [0, 1]; dim=1) * poly_itn(s2, [0, 1]; dim=2)

c1, c2 = exp_itn(s1; dim=1), exp_itn(s2; dim=2)

niter = 100
for iter in 1:niter
  global ψ = ψ * g

  global ψ = partial_integrate(ψ, [dim_ψ])

  global dim_ψ = dim_ψ == 1 ? 2 : 1

  local c = dim_ψ == 1 ? c1 : c2

  global ψ = ψ + c
end

n_grid = 100
x_vals = grid_points(s, n_grid, 1)
ψ_vals = [real(evaluate(ψ, x)) for x in x_vals]
correct_vals = (3 / 2) * x_vals + exp.(x_vals)

avg_err = sum(abs.(correct_vals - ψ_vals)) / n_grid
@show avg_err
