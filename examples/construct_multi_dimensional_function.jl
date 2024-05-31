using ITensorNumericalAnalysis

using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: NamedGraph
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using ITensors: siteinds, maxlinkdim
using Random: Random

L = 12
Random.seed!(1234)
g = named_comb_tree((3, 4))
s = continuous_siteinds(g; map_dimension=3)

println(
  "Constructing the 3D function f(x,y,z) = x³(y + y²) + cosh(πz) as a tensor network on a randomly chosen tree with $L vertices",
)
ψ_fx = poly_itn(s, [0.0, 0.0, 0.0, 1.0]; dim=1)
ψ_fy = poly_itn(s, [0.0, 1.0, 1.0, 0.0]; dim=2)
ψ_fz = cosh_itn(s; k=Number(pi), dim=3)
ψxyz = ψ_fx * ψ_fy + ψ_fz

ψxyz = truncate(ψxyz; cutoff=1e-12)
println("Maximum bond dimension of the network is $(maxlinkdim(ψxyz))")

x, y, z = 0.125, 0.625, 0.5
fxyz_xyz = evaluate(ψxyz, [x, y, z])
println(
  "Tensor network evaluates the function as $fxyz_xyz at the co-ordinate: (x,y,z) = ($x, $y, $z)",
)
println(
  "Actual value of the function is $(x^3 * (y  + y^2) + cosh(pi * z)) at the co-ordinate: (x,y,z) = ($x, $y, $z)",
)
