using Test
using ITensorNumericalAnalysis

using NamedGraphs: vertices
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree
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

@testset "grid_points tests" begin
  @testset "test grid_points irregular span" begin
    L = 16
    base = 2
    g = named_comb_tree((2, L ÷ 2))
    s = continuous_siteinds(g; map_dimension=2)

    #first set -- irregular span
    N = 125
    a = 0.12
    b = 0.95

    # test that giving an irregular span still works and it obeys the constraints
    test_gridpoints = grid_points(s, N, 1; span=[a, b])
    @test length(test_gridpoints) == N
    @test test_gridpoints[1] >= a
    @test test_gridpoints[end] < b
  end

  @testset "test grid_points spacing" begin
    L = 16
    g = named_comb_tree((2, L ÷ 2))
    s = continuous_siteinds(g; map_dimension=2)

    #second set
    N = 32
    a = 0.25
    b = 0.5
    test2 = grid_points(s, N, 1; span=[a, b])
    @test length(test2) == N
    @test test2[1] >= a
    @test test2[end] < b

    # test that the spacing between all points is the same
    spacing_equal = true
    gap = (test2[2] - test2[1])
    for i in 2:(N - 1)
      if !(test2[i + 1] - test2[i] ≈ gap)
        spacing_equal = false
      end
    end
    @test spacing_equal
  end

  @testset "test grid_points exact_grid" begin
    L = 16
    g = named_comb_tree((2, L ÷ 2))
    s = continuous_siteinds(g; map_dimension=2)
    a = 0.25
    b = 0.6

    test3 = grid_points(s, 1; span=[a, b], exact_grid=true)

    # tests that the number of points generated matches the number of exact gridpoints that fall in the range [a,b).
    @test length(test3) == round(b * 2^(L ÷ 2) - 0.5) - round(a * 2^(L ÷ 2)) + 1
    @test test3[1] >= a
    @test test3[end] < b
  end

  @testset "test grid_points high precision" begin
    #fourth set -- very large L
    L = 140
    base = 2
    g = named_comb_tree((2, L ÷ 2))
    s = continuous_siteinds(g; map_dimension=2)
    n_grid = 16
    @test length(grid_points(s, n_grid, 1)) == n_grid
  end
end

@testset "test rand_p" begin
  #test the rand_p() function to see if it succeeds on large L
  L = 140
  g = named_comb_tree((2, L ÷ 2))
  s = continuous_siteinds(g; map_dimension=2)
  ψ = cos_itn(s; dim=1) * cos_itn(s; dim=2)

  #test the use of seedeed randomness
  rng = Random.Xoshiro(42)
  rand_gridpoint = rand_p(rng, s)
  x1 = real(ITensorNumericalAnalysis.evaluate(ψ, rand_gridpoint))
  @test x1 >= -1 && x1 <= 1 # check to make sure ψ can be evaluated at these points

  rand_gridpoint1 = rand_p(rng, s, 1)
  rand_gridpoint2 = rand_p(rng, s, 2)
  x2 = real(ITensorNumericalAnalysis.evaluate(ψ, [rand_gridpoint1, rand_gridpoint2]))
  @test x2 >= -1 && x2 <= 1 # check to make sure ψ can be evaluated at these points

  #test the use of default rng
  default_rng_gridpoint1 = rand_p(s)
  y = real(ITensorNumericalAnalysis.evaluate(ψ, default_rng_gridpoint1))
  @test y >= -1 && y <= 1 # check to make sure ψ can be evaluated at these points

  # same things but with smaller L as well
  L = 12
  g = named_comb_tree((2, L ÷ 2))
  s = continuous_siteinds(g; map_dimension=2)
  ψ = cos_itn(s; dim=1) * cos_itn(s; dim=2)

  #test the use of seedeed randomness
  rng = Random.Xoshiro(42)
  rand_gridpoint = rand_p(rng, s)
  x1 = real(ITensorNumericalAnalysis.evaluate(ψ, rand_gridpoint))
  @test x1 >= -1 && x1 <= 1 # check to make sure ψ can be evaluated at these points

  rand_gridpoint1 = rand_p(rng, s, 1)
  rand_gridpoint2 = rand_p(rng, s, 2)
  x2 = real(ITensorNumericalAnalysis.evaluate(ψ, [rand_gridpoint1, rand_gridpoint2]))
  @test x2 >= -1 && x2 <= 1 # check to make sure ψ can be evaluated at these points

  #test the use of default rng
  default_rng_gridpoint1 = rand_p(s)
  y = real(ITensorNumericalAnalysis.evaluate(ψ, default_rng_gridpoint1))
  @test y >= -1 && y <= 1 # check to make sure ψ can be evaluated at these points
end
