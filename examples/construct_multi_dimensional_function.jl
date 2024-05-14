using ITensorNumericalAnalysis

using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: NamedGraph
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using ITensors: siteinds, maxlinkdim
using Random: Random

L = 12
Random.seed!(1234)
g = named_comb_tree((3, 4))
s = continuous_siteinds(g; map_dimension=3, is_complex = true)

println(
  "Constructing the 3D complex function f(z1,z2,z3) = z1³(z2 + z2²) + cosh(πz3) as a tensor network on a randomly chosen tree with $L vertices",
)
ψ_fz1 = poly_itn(s, [0.0, 0.0, 0.0, 1.0]; dimension=1)
ψ_fz2 = poly_itn(s, [0.0, 1.0, 1.0]; dimension=2)
ψ_fz3 = cosh_itn(s; k=Number(pi), dimension=3)
ψz1z2z3 = ψ_fz1*ψ_fz2 + ψ_fz3

ψz1z2z3 = truncate(ψz1z2z3; cutoff=1e-12)
println("Maximum bond dimension of the network is $(maxlinkdim(ψz1z2z3))")

z1, z2, z3 = 0.125 + 0.5*im, 0.625 + 0.875*im, 0.5
fz1z2z3_z1z2z3 = calculate_fxyz(ψz1z2z3, [z1, z2, z3]; alg = "exact")
println(
  "Tensor network evaluates the function as $fz1z2z3_z1z2z3 at the co-ordinate: (z1,z2,z3) = ($z1, $z2, $z3)",
)
println(
  "Actual value of the function is $(z1^3 * (z2  + z2^2) + cosh(pi * z3)) at the co-ordinate: (z1,z2,z3) =($z1, $z2, $z3)",
)
