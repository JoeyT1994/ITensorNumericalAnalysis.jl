using Test
using ITensorNumericalAnalysis

using NamedGraphs: vertices
using NamedGraphs.NamedGraphGenerators: named_grid
using ITensors: siteinds, inds
using Dictionaries: Dictionary

@testset "indexmap tests" begin
  @testset "test single dimensional index map" begin
    L = 4

    g = named_grid((L, L))

    s = continuous_siteinds(g)

    @test dimension(s) == 1

    x = 0.625
    ind_to_ind_value_map = calculate_ind_values(s, x)
    @test Set(keys(ind_to_ind_value_map)) == Set(inds(s))
    @test only(calculate_p(s, ind_to_ind_value_map)) == x
  end

  @testset "test multi dimensional index map" begin
    L = 10
    g = named_grid((L, L))
    s = continuous_siteinds(g, [[(i, j) for i in 1:L] for j in 1:L])

    x, y = 0.5, 0.75
    ind_to_ind_value_map = calculate_ind_values(s, [x, y], [1, 2])
    xyz = calculate_p(s, ind_to_ind_value_map, [1, 2])
    @test first(xyz) == x
    @test last(xyz) == y

    xyzvals = [rand() for i in 1:L]
    ind_to_ind_value_map = calculate_ind_values(s, xyzvals)

    xyzvals_approx = calculate_p(s, ind_to_ind_value_map, [i for i in 1:dimension(s)])
    xyzvals ≈ xyzvals_approx
  end
end
