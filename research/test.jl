include("commonnetworks.jl")
include("commonfunctions.jl")
include("laughlinfunctions.jl")
include("utils.jl")

using ITensorNetworks: maxlinkdim
using NamedGraphs.GraphsExtensions: add_edges, nv, eccentricity, disjoint_union, degree
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_grid, named_binary_tree
using Random: Random, rand
using ITensors

using NPZ
using MKL

using ITensorNumericalAnalysis

Lx = 18
g12 = named_comb_tree((2,Lx))
delt = 2.0^(-Lx)
s12 = continuous_siteinds(g12, [[(j,i) for i in 1:Lx] for j in 1:2])
f = x -> 1 / (x[1] + x[2] + 1)^2
χ = 10
tn12 = itensornetwork(interpolate(f, s12; initial_state = rand_itn(s12; link_space = 2), nsweeps=10, maxdim=χ, cutoff=1e-17, outputlevel=1))

g1234 = named_comb_tree((4,Lx))
s1234 = continuous_siteinds(g1234, [[(j,i) for i in 1:Lx] for j in 1:4])
tn1234 = const_itn(s1234)
for v in vertices(g12)
    tn1234[v] = replaceinds(tn12[v], s12[v], s1234[v])
end

ngrid_points = 50
xs, ys = [delt * Random.rand(1:(2^Lx-1)) for i in 1:ngrid_points], [delt * Random.rand(1:(2^Lx-1)) for i in 1:ngrid_points]
exact_vals = Float64[]
tn_vals = Float64[]
for x in xs
  for y in ys
    push!(tn_vals, real(evaluate(tn1234, [x, y, 0.5, 0.5])))
    push!(exact_vals, real(f([x, y, 0.5, 0.5])))
  end
end

err = calc_error(exact_vals, tn_vals)
@show maximum(abs.(exact_vals - tn_vals))
@show err