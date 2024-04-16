using Test
using ITensorNumericalAnalysis

using ITensors: siteinds
using ITensorNetworks: maxlinkdim
using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: named_grid, named_comb_tree, NamedGraph, nv, vertices
using ITensorNumericalAnalysis: itensornetwork
using Dictionaries: Dictionary

@testset "test differentiation in 1D on MPS" begin
  g = named_grid((9, 1))
  L = nv(g)
  delta = (2.0)^(-Float64(L))
  s = continuous_siteinds(g)
  index_map = IndexMap(s)
  left_boundary, right_boundary = "Periodic", "Periodic"

  f1 = first_derivative_operator(s, index_map; cutoff=1e-12, left_boundary, right_boundary)
  f2 = second_derivative_operator(s, index_map; cutoff=1e-12, left_boundary, right_boundary)
  f3 = third_derivative_operator(s, index_map; cutoff=1e-12, left_boundary, right_boundary)
  f4 = fourth_derivative_operator(s, index_map; cutoff=1e-12, left_boundary, right_boundary)

  ψ_fx = sin_itn(s, index_map; k=2.0 * Float64(pi))

  ψ_f1x = operate(f1, ψ_fx; cutoff=1e-8)
  ψ_f2x = operate(f2, ψ_fx; cutoff=1e-8)
  ψ_f3x = operate(f3, ψ_fx; cutoff=1e-8)
  ψ_f4x = operate(f4, ψ_fx; cutoff=1e-8)

  xs = [0.0, 0.25, 0.625, 0.875, 1.0 - delta]
  for x in xs
    @test 1.0 + calculate_fx(ψ_fx, x) ≈ 1.0 + sin(2.0 * pi * x) rtol = 1e-3
    @test 1.0 + calculate_fx(ψ_f1x, x) ≈ 1.0 + 2.0 * pi * cos(2.0 * pi * x) rtol = 1e-3
    @test 1.0 + calculate_fx(ψ_f2x, x) ≈ 1.0 + -1.0 * (2.0 * pi)^2 * sin(2.0 * pi * x) rtol =
      1e-3
    @test 1.0 + calculate_fx(ψ_f3x, x) ≈ 1.0 + -1.0 * (2.0 * pi)^3 * cos(2.0 * pi * x) rtol =
      1e-3
    @test 1.0 + calculate_fx(ψ_f4x, x) ≈ 1.0 + 1.0 * (2.0 * pi)^4 * sin(2.0 * pi * x) rtol =
      1e-3
  end
end

@testset "test derivative in 1D on tree" begin
  g = named_comb_tree((4, 3))
  L = nv(g)
  delta = 2.0^(-Float64(L))
  s = continuous_siteinds(g)
  index_map = IndexMap(s)

  ∂_∂x = first_derivative_operator(s, index_map; cutoff=1e-10)

  ψ_fx = sin_itn(s, index_map; k=Float64(pi))
  ∂x_ψ_fx = operate(∂_∂x, ψ_fx; cutoff=1e-12)

  xs = [delta, 0.125, 0.25, 0.625, 0.875]
  for x in xs
    ∂x_ψ_fx_x = real(calculate_fx(∂x_ψ_fx, x))
    @test ∂x_ψ_fx_x ≈ pi * cos(pi * x) atol = 1e-3
  end
end

@testset "test multiplication_operator_in_1D" begin
  g = named_comb_tree((4, 3))
  L = nv(g)
  s = continuous_siteinds(g)
  index_map = IndexMap(s)

  ψ_gx = sin_itn(s, index_map; k=0.5 * Float64(pi))
  ψ_fx = cos_itn(s, index_map; k=0.25 * Float64(pi))

  ψ_fxgx = ψ_gx * ψ_fx
  xs = [0.025, 0.1, 0.25, 0.625, 0.875]
  for x in xs
    ψ_fxgx_x = real(calculate_fx(ψ_fxgx, x))
    @test ψ_fxgx_x ≈ sin(0.5 * pi * x) * cos(0.25 * pi * x) atol = 1e-3
  end
end

@testset "test multiplication_operator_in_2D" begin
  L = 8
  g = NamedGraph(SimpleGraph(uniform_tree(L)))

  s = continuous_siteinds(g)
  index_map = IndexMap(s; map_dimension=2)

  ψ_fx = cos_itn(s, index_map; k=0.25 * Float64(pi), dimension=1)
  ψ_gy = sin_itn(s, index_map; k=0.5 * Float64(pi), dimension=2)
  @assert dimension(ψ_fx) == dimension(ψ_gy) == 2

  ψ_fxgy = ψ_fx * ψ_gy

  xs = [0.125, 0.25, 0.625, 0.875]
  ys = [0.125, 0.25, 0.625, 0.875]
  for x in xs
    for y in ys
      ψ_fx_x = real(calculate_fxyz(ψ_fx, [x, y]))
      ψ_gy_y = real(calculate_fxyz(ψ_gy, [x, y]))
      @test ψ_fx_x ≈ cos(0.25 * pi * x)
      @test ψ_gy_y ≈ sin(0.5 * pi * y)
      ψ_fxgy_xy = real(calculate_fxyz(ψ_fxgy, [x, y]))
      @test ψ_fxgy_xy ≈ cos(0.25 * pi * x) * sin(0.5 * pi * y) atol = 1e-3
    end
  end
end

@testset "test differentiation_operator_on_3D_function" begin
  L = 45
  g = named_grid((L, 1))

  s = continuous_siteinds(g)
  index_map = IndexMap(s; map_dimension=3)

  ψ_fx = poly_itn(s, index_map, [0.0, -1.0, 1.0]; dimension=1)
  ψ_gy = sin_itn(s, index_map; k=Float64(pi), dimension=2)
  ψ_hz = sin_itn(s, index_map; k=Float64(pi), dimension=3)
  @assert dimension(ψ_fx) == dimension(ψ_gy) == dimension(ψ_hz) == 3

  ψ_fxgyhz = ψ_fx * ψ_gy * ψ_hz

  ∂_∂y = first_derivative_operator(s, index_map; dimension=2, cutoff=1e-10)

  ∂_∂y_ψ_fxgyhz = operate([∂_∂y], ψ_fxgyhz; cutoff=1e-10)

  xs = [0.125, 0.25, 0.675]
  ys = [0.125, 0.25, 0.675]
  zs = [0.125, 0.25, 0.675]
  for x in xs
    for y in ys
      for z in zs
        ψ_fxgyhz_xyz = real(calculate_fxyz(ψ_fxgyhz, [x, y, z]))
        @test ψ_fxgyhz_xyz ≈ (x^2 - x) * sin(pi * y) * sin(pi * z) atol = 1e-3

        ∂_∂y_ψ_fxgyhz_xyz = real(calculate_fxyz(∂_∂y_ψ_fxgyhz, [x, y, z]))
        @test ∂_∂y_ψ_fxgyhz_xyz ≈ pi * (x^2 - x) * cos(pi * y) * sin(pi * z) atol = 1e-3
      end
    end
  end
end
