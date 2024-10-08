using Test
using ITensorNumericalAnalysis

using NamedGraphs: vertices
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs.GraphsExtensions: is_tree
using ITensors: siteinds, inds
using Dictionaries: Dictionary
using Random
using ITensorNetworks: union_all_inds, subgraph
using ITensorNumericalAnalysis: reduced_indsnetworkmap

Random.seed!(1234)

@testset "indexmap tests" begin
  @testset "test real single dimensional index map" begin
    L = 4

    g = named_grid((L, L))

    s = continuous_siteinds(g)

    @test dimension(s) == 1
    @test isa(indexmap(s), RealIndexMap)
    @test indexmaptype(s) <: AbstractIndexMap

    x = 0.625
    ind_to_ind_value_map = calculate_ind_values(s, x)
    @test Set(keys(ind_to_ind_value_map)) == Set(inds(s))
    @test only(calculate_p(s, ind_to_ind_value_map)) == x
  end

  @testset "test complex single dimensional index map" begin
    L = 4

    g = named_grid((L, L))

    s = complex_continuous_siteinds(g)

    @test dimension(s) == 1
    @test isa(indexmap(s), ComplexIndexMap)
    @test indexmaptype(s) <: AbstractIndexMap

    z = 0.625 + 0.5 * im
    ind_to_ind_value_map = calculate_ind_values(s, z)
    @test Set(keys(ind_to_ind_value_map)) == Set(inds(s))
    @test only(calculate_p(s, ind_to_ind_value_map)) == z
  end

  @testset "test complex multi dimensional index map" begin
    L = 10
    g = named_grid((L, L))
    dimension_vertices = [[(i, j) for i in 1:L] for j in 1:L]
    s = complex_continuous_siteinds(g, dimension_vertices, dimension_vertices)

    s4 = reduced_indsnetworkmap(s, 4)
    @test is_tree(s4)
    @test issetequal(inds(s4), dimension_inds(s, 4))

    z1, z2 = 0.5 + 0.125 * im, 0.75 + 0.625 * im
    ind_to_ind_value_map = calculate_ind_values(s, [z1, z2], [1, 2])
    xyz = calculate_p(s, ind_to_ind_value_map, [1, 2])
    @test first(xyz) == z1
    @test last(xyz) == z2

    xyzvals = [rand() + im * rand() for i in 1:L]
    ind_to_ind_value_map = calculate_ind_values(s, xyzvals)

    xyzvals_approx = calculate_p(s, ind_to_ind_value_map, [i for i in 1:dimension(s)])
    xyzvals ≈ xyzvals_approx
  end
end
