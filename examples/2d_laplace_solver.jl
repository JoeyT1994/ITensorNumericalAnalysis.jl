using Test
using ITensorNumericalAnalysis

using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: NamedGraph
using NamedGraphs.GraphsExtensions: rename_vertices
using ITensors: ITensors, Index, siteinds, dim, tags, replaceprime!, MPO, MPS, inner
using ITensorNetworks: ITensorNetwork, dmrg, ttn, maxlinkdim
using Dictionaries: Dictionary
using Random: seed!

using UnicodePlots

#Solve the 2D Laplace equation on a random tree
seed!(1234)
L = 12
g = NamedGraph(SimpleGraph(uniform_tree(L)))
g = rename_vertices(v -> (v, 1), g)

s = continuous_siteinds(g; map_dimension=2)

ψ_fxy = 0.1 * rand_itn(s; link_space=3)
∇ = laplacian_operator(s; scale=false, cutoff=1e-8)
println("2D Laplacian constructed for this tree, bond dimension is $(maxlinkdim(∇))")

init_energy =
  inner(ttn(itensornetwork(ψ_fxy))', ∇, ttn(itensornetwork(ψ_fxy))) /
  inner(ttn(itensornetwork(ψ_fxy)), ttn(itensornetwork(ψ_fxy)))
println(
  "Starting DMRG to find eigensolution of 2D Laplace operator. Initial energy is $init_energy",
)

dmrg_kwargs = (nsweeps=15, normalize=true, maxdim=30, cutoff=1e-12, outputlevel=1, nsites=2)
final_energy, ϕ_fxy = dmrg(∇, ttn(itensornetwork(ψ_fxy)); dmrg_kwargs...)
ϕ_fxy = ITensorNetworkFunction(ITensorNetwork(ϕ_fxy), s)

ϕ_fxy = truncate(ϕ_fxy; cutoff=1e-10)

println(
  "Finished DMRG. Found solution of energy $final_energy with bond dimension $(maxlinkdim(ϕ_fxy))",
)
println(
  "Note that in 2D, the discrete laplacian with a step size of 1 has a lowest eigenvalue of -8.",
)

n_grid = 100
x_vals, y_vals = grid_points(s, n_grid, 1), grid_points(s, n_grid, 2)
vals = zeros((length(x_vals), length(y_vals)))
for (i, x) in enumerate(x_vals)
  for (j, y) in enumerate(y_vals)
    vals[i, j] = real(evaluate(ϕ_fxy, [x, y]))
  end
end

println("Here is the heatmap of the 2D function")
display(heatmap(vals; xfact=0.01, yfact=0.01, xoffset=0, yoffset=0, colormap=:inferno))

n_grid = 100
x_vals = grid_points(s, n_grid, 1)
y = 0.5
vals = zeros(length(x_vals))
for (i, x) in enumerate(x_vals)
  vals[i] = real(evaluate(ϕ_fxy, [x, y]))
end

println("Here is a cut of the function at y = $y")
display(lineplot(x_vals, vals))
