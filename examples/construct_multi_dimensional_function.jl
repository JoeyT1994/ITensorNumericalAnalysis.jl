using ITensorNumericalAnalysis

using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: NamedGraph
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using ITensors: siteinds, maxlinkdim, inds
using Random: Random

L = 12
Random.seed!(1234)
g = named_comb_tree((3, 4))
s = complex_continuous_siteinds(g; map_dimension=3)

println(
  "Constructing the 3D complex function f(z1,z2,z3) = z1³(z2 + z2²) + cosh(πz3)^2 as a tensor network on a randomly chosen tree with $L vertices",
)
L = 12
Random.seed!(1234)
g = named_comb_tree((3, 4))
real_dim_vertices = [[(j, i) for i in 1:4] for j in 1:3]
imag_dim_vertices = [[(j, i) for i in 4:-1:1] for j in 3:-1:1]
s = complex_continuous_siteinds(g, real_dim_vertices, imag_dim_vertices)
ψ_fz1 = poly_itn(s, [0.0, 0.0, 0.0, 1.0]; dim=1)
ψ_fz2 = poly_itn(s, [0.0, 1.0, 1.0]; dim=2)
ψ_fz3 = cosh_itn(s; k=Number(pi), dim=3)
ψ_z = ψ_fz1 * ψ_fz2 + ψ_fz3 * ψ_fz3

ψ_z = truncate(ψ_z; cutoff=1e-12)
println("Maximum bond dimension of the network is $(maxlinkdim(ψ_z))")

z1, z2, z3 = 0.125 + 0.5 * im, 0.625 + 0.875 * im, 0.5
z = [z1, z2, z3]
f_at_z = evaluate(ψ_z, z)
println(
  "Tensor network evaluates the function as $f_at_z at the co-ordinate: (z1,z2,z3) = ($z1, $z2, $z3)",
)
println(
  "Actual value of the function is $(z1^3 * (z2  + z2^2) + cosh(pi * z3)^2) at the co-ordinate: (z1,z2,z3) =($z1, $z2, $z3)",
)
