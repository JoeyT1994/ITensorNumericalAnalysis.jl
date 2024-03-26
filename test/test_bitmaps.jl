using Test
using TensorNetworkFunctionals
using ITensors
using ITensorNetworks
using SplitApplyCombine
using Dictionaries

#include("../src/itensornetworksutils.jl")

@testset "test single dimensional bit map" begin
  L = 4

  g = named_grid((L, L))
  s = siteinds("S=1/2", g)
  bit_map = BitMap(g)

  @test dimension(bit_map) == 1
  @test Set(vertices(bit_map)) == Set(vertices(g))

  x = 0.625
  vertex_to_bit_value_map = calculate_bit_values(bit_map, x)
  @test Set(keys(vertex_to_bit_value_map)) == Set(vertices(g))
  @test calculate_x(bit_map, vertex_to_bit_value_map) == x
end

@testset "test multi dimensional bit map" begin
  L = 50
  g = named_grid((L, L))
  vertex_to_dimension_map = Dictionary(vertices(g), [v[1] for v in vertices(g)])
  vertex_to_bit_map = Dictionary(vertices(g), [v[2] for v in vertices(g)])
  bit_map = BitMap(vertex_to_bit_map, vertex_to_dimension_map)

  x, y = 0.5, 0.75
  vertex_to_bit_value_map = calculate_bit_values(bit_map, [x, y], [1, 2])
  xyz = calculate_xyz(bit_map, vertex_to_bit_value_map, [1, 2])
  @test first(xyz) == x
  @test last(xyz) == y

  xyzvals = [rand() for i in 1:L]
  vertex_to_bit_value_map = calculate_bit_values(bit_map, xyzvals)

  xyzvals_approx = calculate_xyz(bit_map, vertex_to_bit_value_map)
  xyzvals ≈ xyzvals_approx
end
