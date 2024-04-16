using ITensorNumericalAnalysis

using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: NamedGraph
using ITensors: siteinds
using Random: Random

L = 12
Random.seed!(1234)
g = NamedGraph(SimpleGraph(uniform_tree(L)))
s = continuous_siteinds(g)
index_map = IndexMap(s; map_dimension=3)

println(
  "Constructing the 3D function f(x,y,z) = x³(y + y²) + cosh(πz) as a tensor network on a randomly chosen tree with $L vertices",
)
ψ_fx = poly_itn(s, index_map, [0.0, 0.0, 0.0, 1.0]; dimension=1)
ψ_fy = poly_itn(s, index_map, [0.0, 1.0, 1.0, 0.0]; dimension=2)
ψ_fz = cosh_itn(s, index_map; k=Float64(pi), dimension=3)
ψxyz = ψ_fx * ψ_fy + ψ_fz

ψxyz = truncate(ψxyz; cutoff=1e-12)
println("Maximum bond dimension of the network is $(maxlinkdim(ψxyz))")

x, y, z = 0.125, 0.625, 0.5
fxyz_xyz = calculate_fxyz(ψxyz, [x, y, z])
println(
  "Tensor network evaluates the function as $fxyz_xyz at the co-ordinate: (x,y,z) = ($x, $y, $z)",
)
println(
  "Actual value of the function is $(x^3 * (y  + y^2) + cosh(pi * z)) at the co-ordinate: (x,y,z) = ($x, $y, $z)",
)
