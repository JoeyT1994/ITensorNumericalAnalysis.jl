# TensorNetworkFunctionals

Package for constructing and manipulating low-rank Tensor Network representations of arbitrary functions.
For example to build a bond-dimension 2 cubic tensor-network representation of the function $\cosh(kx + a)$ over the domain $x \in [0,1]$ we can do

```
julia> using ITensors, NamedGraphs, ITensorNetworks, EllipsisNotation

julia> using NamedGraphs: add_edges

julia> using ITensorNetworks: delta_network

julia> include("src/QTT_utils.jl")
calculate_xis (generic function with 1 method)

julia> L = 2;

julia> g = named_grid((L, L, L));

julia> a, k = 1.0, 0.5;

julia> s = siteinds("S=1/2", g);

julia> vertex_map = Dict(vertices(g) .=> [i for i in 1:length(vertices(g))]);

julia> ψ = cosh_itn(s, vertex_map; a, k)
ITensorNetwork{Tuple{Int64, Int64, Int64}} with 8 vertices:
8-element Vector{Tuple{Int64, Int64, Int64}}:
 (1, 1, 1)
 (2, 1, 1)
 (1, 2, 1)
 (2, 2, 1)
 (1, 1, 2)
 (2, 1, 2)
 (1, 2, 2)
 (2, 2, 2)

and 12 edge(s):
(1, 1, 1) => (2, 1, 1)
(1, 1, 1) => (1, 2, 1)
(1, 1, 1) => (1, 1, 2)
(2, 1, 1) => (2, 2, 1)
(2, 1, 1) => (2, 1, 2)
(1, 2, 1) => (2, 2, 1)
(1, 2, 1) => (1, 2, 2)
(2, 2, 1) => (2, 2, 2)
(1, 1, 2) => (2, 1, 2)
(1, 1, 2) => (1, 2, 2)
(2, 1, 2) => (2, 2, 2)
(1, 2, 2) => (2, 2, 2)

with vertex data:
8-element Dictionaries.Dictionary{Tuple{Int64, Int64, Int64}, Any}
 (1, 1, 1) │ ((dim=2|id=165|"1×1×1,S=1/2,Site"), (dim=2|id=95|"1×1×1↔2×1×1"), (dim=2|id=729|"1×1×1↔1×2×1"), (dim=2|id=15|"1×1×1↔1…
 (2, 1, 1) │ ((dim=2|id=942|"2×1×1,S=1/2,Site"), (dim=2|id=95|"1×1×1↔2×1×1"), (dim=2|id=74|"2×1×1↔2×2×1"), (dim=2|id=711|"2×1×1↔2…
 (1, 2, 1) │ ((dim=2|id=353|"1×2×1,S=1/2,Site"), (dim=2|id=729|"1×1×1↔1×2×1"), (dim=2|id=352|"1×2×1↔2×2×1"), (dim=2|id=105|"1×2×1…
 (2, 2, 1) │ ((dim=2|id=16|"2×2×1,S=1/2,Site"), (dim=2|id=74|"2×1×1↔2×2×1"), (dim=2|id=352|"1×2×1↔2×2×1"), (dim=2|id=457|"2×2×1↔2…
 (1, 1, 2) │ ((dim=2|id=44|"1×1×2,S=1/2,Site"), (dim=2|id=15|"1×1×1↔1×1×2"), (dim=2|id=235|"1×1×2↔2×1×2"), (dim=2|id=986|"1×1×2↔1…
 (2, 1, 2) │ ((dim=2|id=712|"2×1×2,S=1/2,Site"), (dim=2|id=711|"2×1×1↔2×1×2"), (dim=2|id=235|"1×1×2↔2×1×2"), (dim=2|id=577|"2×1×2…
 (1, 2, 2) │ ((dim=2|id=480|"1×2×2,S=1/2,Site"), (dim=2|id=105|"1×2×1↔1×2×2"), (dim=2|id=986|"1×1×2↔1×2×2"), (dim=2|id=176|"1×2×2…
 (2, 2, 2) │ ((dim=2|id=233|"2×2×2,S=1/2,Site"), (dim=2|id=457|"2×2×1↔2×2×2"), (dim=2|id=577|"2×1×2↔2×2×2"), (dim=2|id=176|"1×2×2…

julia> 


