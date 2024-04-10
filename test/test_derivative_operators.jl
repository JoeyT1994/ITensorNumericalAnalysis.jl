using Test
using ITensorNumericalAnalysis

using ITensors: siteinds
using ITensorNetworks: maxlinkdim
using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: named_grid, named_comb_tree, NamedGraph, nv, vertices
using ITensorNumericalAnalysis: itensornetwork
using Dictionaries: Dictionary

@testset "test laplacian in 1D on MPS" begin
  g = named_grid((12, 1))
  L = nv(g)
  delta = (2.0)^(-Float64(L))
  bit_map = BitMap(g)
  s = siteinds(g, bit_map)

  ∇sq = laplacian_operator(s, bit_map; cutoff=1e-12)
  @test maxlinkdim(∇sq) == 3

  ψ_fx = sin_itn(s, bit_map; k=Float64(pi))
  ∂2x_ψ_fx = operate(∇sq, ψ_fx; truncate_kwargs=(; cutoff=1e-12))

  xs = [delta, 0.25, 0.625, 0.875]
  for x in xs
    ∂2x_ψ_fx_x = real(calculate_fx(∂2x_ψ_fx, x))
    @test ∂2x_ψ_fx_x ≈ -pi * pi * sin(pi * x) atol = 1e-3
  end

  ∇sq = laplacian_operator(
    s, bit_map; cutoff=1e-12, left_boundary="Periodic", right_boundary="Periodic"
  )

  ψ_fx = sin_itn(s, bit_map; k=2.0 * Float64(pi))
  ∂2x_ψ_fx = operate(∇sq, ψ_fx; truncate_kwargs=(; cutoff=1e-12))

  #Can include 0.0 now because the boundary conditions fix this
  xs = [0.0, delta, 0.25, 0.625, 0.875]
  for x in xs
    ∂2x_ψ_fx_x = real(calculate_fx(∂2x_ψ_fx, x))
    @test ∂2x_ψ_fx_x ≈ -4.0 * pi * pi * sin(2.0 * pi * x) atol = 1e-3
  end
end

@testset "test derivative in 1D on tree" begin
  g = named_comb_tree((4, 3))
  L = nv(g)
  delta = 2.0^(-Float64(L))
  bit_map = BitMap(g)
  s = siteinds(g, bit_map)

  ∂_∂x = derivative_operator(s, bit_map; cutoff=1e-10)

  ψ_fx = sin_itn(s, bit_map; k=Float64(pi))
  ∂x_ψ_fx = operate(∂_∂x, ψ_fx; truncate_kwargs=(; cutoff=1e-12))

  xs = [delta, 0.125, 0.25, 0.625, 0.875]
  for x in xs
    ∂x_ψ_fx_x = real(calculate_fx(∂x_ψ_fx, x))
    @test ∂x_ψ_fx_x ≈ pi * cos(pi * x) atol = 1e-3
  end
end

@testset "test multiplication_operator_in_1D" begin
  g = named_comb_tree((4, 3))
  L = nv(g)
  bit_map = BitMap(g)
  s = siteinds(g, bit_map)

  ψ_gx = sin_itn(s, bit_map; k=0.5 * Float64(pi))
  ψ_fx = cos_itn(s, bit_map; k=0.25 * Float64(pi))

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

  bit_map = BitMap(g; map_dimension=2)
  s = siteinds(g, bit_map)

  ψ_fx = cos_itn(s, bit_map; k=0.25 * Float64(pi), dimension=1)
  ψ_gy = sin_itn(s, bit_map; k=0.5 * Float64(pi), dimension=2)
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

  bit_map = BitMap(g; map_dimension=3)
  s = siteinds(g, bit_map)

  ψ_fx = poly_itn(s, bit_map, [0.0, -1.0, 1.0]; dimension=1)
  ψ_gy = sin_itn(s, bit_map; k=Float64(pi), dimension=2)
  ψ_hz = sin_itn(s, bit_map; k=Float64(pi), dimension=3)
  @assert dimension(ψ_fx) == dimension(ψ_gy) == dimension(ψ_hz) == 3

  ψ_fxgyhz = ψ_fx * ψ_gy * ψ_hz

  ∂_∂y = derivative_operator(s, bit_map; dimension=2, cutoff=1e-10)

  ∂_∂y_ψ_fxgyhz = operate([∂_∂y], ψ_fxgyhz; truncate_kwargs=(; cutoff=1e-10))

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
