using Test
using ITensorNumericalAnalysis
using TensorNetworkQuantumSimulator

using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: NamedGraph, vertices, rename_vertices
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree
using Dictionaries: Dictionary
using Random: Random

Random.seed!(1234)

@testset "test real itensorfunctions" begin
    @testset "test constructor from ITensorNetwork" begin
        L = 10

        g = named_grid((L, 3))
        s = continuous_siteinds(g)

        nterms = 5
        ks = [randn() for _ in 1:nterms]
        ψ = reduce(+, [exp_tnf(g, s; k = k) for k in ks])

        ψ_mod = truncate(ψ; maxdim = 4, alg = "boundarymps", mps_bond_dimension = 24)

        @show evaluate(ψ, 0.5)
        @show evaluate(ψ_mod, 0.5)

    end
end
