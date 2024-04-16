using Test
using ITensorNumericalAnalysis

using NamedGraphs: named_grid, vertices
using ITensors: siteinds, inds
using Dictionaries: Dictionary

@testset "test single dimensional index map" begin
  L = 4

  g = named_grid((L, L))

  s = continuous_siteinds(g)
  index_map = IndexMap(s)

  @test dimension(index_map) == 1
  @test Set(inds(index_map)) == Set(inds(s))

  x = 0.625
  ind_to_ind_value_map = calculate_ind_values(index_map, x)
  @test Set(keys(ind_to_ind_value_map)) == Set(inds(s))
  @test calculate_x(index_map, ind_to_ind_value_map) == x
end

@testset "test multi dimensional index map" begin
  L = 10
  g = named_grid((L, L))
  s = continuous_siteinds(g)
  index_map = IndexMap(s, [[(i, j) for i in 1:L] for j in 1:L])

  x, y = 0.5, 0.75
  ind_to_ind_value_map = calculate_ind_values(index_map, [x, y], [1, 2])
  xyz = calculate_xyz(index_map, ind_to_ind_value_map, [1, 2])
  @test first(xyz) == x
  @test last(xyz) == y

  xyzvals = [rand() for i in 1:L]
  ind_to_ind_value_map = calculate_ind_values(index_map, xyzvals)

  xyzvals_approx = calculate_xyz(index_map, ind_to_ind_value_map)
  xyzvals ≈ xyzvals_approx
end
