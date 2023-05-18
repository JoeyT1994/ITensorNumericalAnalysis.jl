# QTTITensorNetworks

Package for constructing and manipulating low-rank Tensor Network representations of arbitrary functions.
For example to build a bond-dimension 2 cubic tensor-network representation of the function $\cosh(kx + a)$ over the domain $x \in [0,1]$ we can do

```
using ITensors, NamedGraphs, ITensorNetworks, EllipsisNotation
using NamedGraphs: add_edges
using ITensorNetworks: delta_network

include("src/QTT_utils.jl")
L = 2;
g = named_grid((L, L, L));
a, k = 1.0, 0.5;
s = siteinds("S=1/2", g);
vertex_map = Dict(vertices(g) .=> [i for i in 1:length(vertices(g))]);
ψ = cosh_itn(s, vertex_map; a, k)


