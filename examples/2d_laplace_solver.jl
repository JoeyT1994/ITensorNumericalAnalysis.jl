using Test
using ITensorNumericalAnalysis

using Graphs: SimpleGraph, uniform_tree, binary_tree, random_regular_graph, is_tree
using NamedGraphs:
  NamedGraph,
  named_grid,
  vertices,
  named_comb_tree,
  rename_vertices,
  random_bfs_tree,
  undirected_graph
using ITensors: ITensors, Index, siteinds, dim, tags, replaceprime!, MPO, MPS, inner
using ITensorNetworks: ITensorNetwork, dmrg, TTN, maxlinkdim
using Dictionaries: Dictionary
using SplitApplyCombine: group
using Random: seed!
using Distributions: Uniform

using UnicodePlots

#Solve the 2D Laplace equation on a random tree
seed!(1234)
L = 14
g = NamedGraph(SimpleGraph(uniform_tree(L)))
s = siteinds("S=1/2", g)

vertex_to_dimension_map = Dictionary(vertices(g), [(v[1] % 2) + 1 for v in vertices(g)])
vertex_to_bit_map = Dictionary(vertices(g), [ceil(Int64, v[1] * 0.5) for v in vertices(g)])
bit_map = BitMap(vertex_to_bit_map, vertex_to_dimension_map)

ψ_fxy = 0.1 * rand_itn(s, bit_map; link_space=2)
∇ = laplacian_operator(s, bit_map; scale=false)
∇ = truncate(∇; cutoff=1e-12)
@show maxlinkdim(∇)

dmrg_kwargs = (nsweeps=25, normalize=true, maxdim=20, cutoff=1e-12, outputlevel=1, nsites=2)
ϕ_fxy = dmrg(∇, TTN(itensornetwork(ψ_fxy)); dmrg_kwargs...)
ϕ_fxy = ITensorNetworkFunction(ITensorNetwork(ϕ_fxy), bit_map)

final_energy = inner(TTN(itensornetwork(ϕ_fxy))', ∇, TTN(itensornetwork(ϕ_fxy)))
#Smallest eigenvalue in this case should be -8
@show final_energy

n_grid = 100
x_vals, y_vals = grid_points(bit_map, n_grid, 1), grid_points(bit_map, n_grid, 2)
vals = zeros((length(x_vals), length(y_vals)))
for (i, x) in enumerate(x_vals)
  for (j, y) in enumerate(y_vals)
    vals[i, j] = real(calculate_fxyz(ϕ_fxy, [x, y]))
  end
end

show(heatmap(vals))
