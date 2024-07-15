include("commonnetworks.jl")
include("commonfunctions.jl")
include("laughlinfunctions.jl")
include("utils.jl")

using ITensorNetworks: maxlinkdim
using NamedGraphs.GraphsExtensions: add_edges, nv, eccentricity, disjoint_union
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_grid, named_binary_tree
using Random: Random, rand

using NPZ
using MKL

using ITensorNumericalAnalysis

g = named_grid((2,2))
#g = named_comb_tree((2,2))
s = continuous_siteinds(g)

psi = exp_itn(s; c = 0.5, k = 1.0) + exp_itn(s; c = 0.5, k = -1.0)
psi1 = exp_itn(s; c = 0.5, k = 1.0)
psi2 = exp_itn(s; c = 0.5, k = -1.0)
x = 0.5

@show calculate_fx(psi, x)
@show calculate_fx(psi1, x) + calculate_fx(psi2, x)
@show cosh(x)