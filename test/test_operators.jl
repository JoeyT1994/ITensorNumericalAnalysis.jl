using Test
using ITensorNumericalAnalysis
using TensorOperations: TensorOperations

using ITensors: siteinds
using ITensorNetworks: ITensorNetwork, maxlinkdim, ttn, inner
using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: NamedGraph, nv, vertices
using NamedGraphs.GraphsExtensions: rename_vertices
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree
using ITensorNumericalAnalysis:
  itensornetwork, forward_shift_op, backward_shift_op, delta_kernel
using Dictionaries: Dictionary

@testset "test operators" begin
  @testset "test differentiation in 1D on MPS" begin
    g = named_grid((9, 1))
    L = nv(g)
    delta = (2.0)^(-Number(L))
    s = continuous_siteinds(g)
    left_boundary, right_boundary = "Periodic", "Periodic"

    f1 = first_derivative_operator(s; cutoff=1e-12, left_boundary, right_boundary)
    f2 = second_derivative_operator(s; cutoff=1e-12, left_boundary, right_boundary)
    f3 = third_derivative_operator(s; cutoff=1e-12, left_boundary, right_boundary)
    f4 = fourth_derivative_operator(s; cutoff=1e-12, left_boundary, right_boundary)

    ψ_fx = sin_itn(s; k=2.0 * Number(pi))

    ψ_f1x = operate(f1, ψ_fx; cutoff=1e-8)
    ψ_f2x = operate(f2, ψ_fx; cutoff=1e-8)
    ψ_f3x = operate(f3, ψ_fx; cutoff=1e-8)
    ψ_f4x = operate(f4, ψ_fx; cutoff=1e-8)

    xs = [0.0, 0.25, 0.625, 0.875, 1.0 - delta]
    for x in xs
      @test 1.0 + evaluate(ψ_fx, x; alg="exact") ≈ 1.0 + sin(2.0 * pi * x) rtol = 1e-3
      @test 1.0 + evaluate(ψ_f1x, x) ≈ 1.0 + 2.0 * pi * cos(2.0 * pi * x) rtol = 1e-3
      @test 1.0 + evaluate(ψ_f2x, x) ≈ 1.0 + -1.0 * (2.0 * pi)^2 * sin(2.0 * pi * x) rtol =
        1e-3
      @test 1.0 + evaluate(ψ_f3x, x) ≈ 1.0 + -1.0 * (2.0 * pi)^3 * cos(2.0 * pi * x) rtol =
        1e-3
      @test 1.0 + evaluate(ψ_f4x, x) ≈ 1.0 + 1.0 * (2.0 * pi)^4 * sin(2.0 * pi * x) rtol =
        1e-3
    end
  end

  @testset "test differentiation in 1D on tree" begin
    g = named_comb_tree((4, 3))
    L = nv(g)
    delta = 2.0^(-Number(L))
    s = continuous_siteinds(g)

    ∂_∂x = first_derivative_operator(s; cutoff=1e-10)

    ψ_fx = sin_itn(s; k=Number(pi))
    ∂x_ψ_fx = operate(∂_∂x, ψ_fx; cutoff=1e-12)

    xs = [delta, 0.125, 0.25, 0.625, 0.875]
    for x in xs
      ∂x_ψ_fx_x = real(evaluate(∂x_ψ_fx, x))
      @test ∂x_ψ_fx_x ≈ pi * cos(pi * x) atol = 1e-3
    end
  end

  @testset "test differentiation_operator_on_3D_function" begin
    L = 45
    g = named_grid((L, 1))

    s = continuous_siteinds(g; map_dimension=3)

    ψ_fx = poly_itn(s, [0.0, -1.0, 1.0]; dim=1)
    ψ_gy = sin_itn(s; k=Number(pi), dim=2)
    ψ_hz = sin_itn(s; k=Number(pi), dim=3)
    @assert dimension(ψ_fx) == dimension(ψ_gy) == dimension(ψ_hz) == 3

    ψ_fxgyhz = ψ_fx * ψ_gy * ψ_hz

    ∂_∂y = first_derivative_operator(s; dim=2, cutoff=1e-10)

    ∂_∂y_ψ_fxgyhz = operate([∂_∂y], ψ_fxgyhz; cutoff=1e-10)

    xs = [0.125, 0.25, 0.675]
    ys = [0.125, 0.25, 0.675]
    zs = [0.125, 0.25, 0.675]
    for x in xs
      for y in ys
        for z in zs
          ψ_fxgyhz_xyz = real(evaluate(ψ_fxgyhz, [x, y, z]))
          @test ψ_fxgyhz_xyz ≈ (x^2 - x) * sin(pi * y) * sin(pi * z) atol = 1e-3

          ∂_∂y_ψ_fxgyhz_xyz = real(evaluate(∂_∂y_ψ_fxgyhz, [x, y, z]))
          @test ∂_∂y_ψ_fxgyhz_xyz ≈ pi * (x^2 - x) * cos(pi * y) * sin(pi * z) atol = 1e-3
        end
      end
    end
  end

  @testset "test multiplication_operator_in_1D" begin
    g = named_comb_tree((4, 3))
    L = nv(g)
    s = continuous_siteinds(g)

    ψ_gx = sin_itn(s; k=0.5 * Number(pi))
    ψ_fx = cos_itn(s; k=0.25 * Number(pi))

    ψ_fxgx = ψ_gx * ψ_fx
    ψ_sq = ψ_fx * ψ_fx
    xs = [0.025, 0.1, 0.25, 0.625, 0.875]
    for x in xs
      ψ_fxgx_x = real(evaluate(ψ_fxgx, x))
      @test ψ_fxgx_x ≈ sin(0.5 * pi * x) * cos(0.25 * pi * x) atol = 1e-3
      ψ_sq_x = real(evaluate(ψ_sq, x))
      @test ψ_sq_x ≈ cos(0.25 * pi * x) * cos(0.25 * pi * x) atol = 1e-3
    end
  end

  @testset "test multiplication_operator_in_2D" begin
    L = 8
    g = NamedGraph(SimpleGraph(uniform_tree(L)))
    g = rename_vertices(v -> (v, 1), g)

    s = continuous_siteinds(g; map_dimension=2)

    ψ_fx = cos_itn(s; k=0.25 * Number(pi), dim=1)
    ψ_gy = sin_itn(s; k=0.5 * Number(pi), dim=2)
    @assert dimension(ψ_fx) == dimension(ψ_gy) == 2

    ψ_fxgy = ψ_fx * ψ_gy
    ψ_sq = ψ_gy * ψ_gy

    xs = [0.125, 0.25, 0.625, 0.875]
    ys = [0.125, 0.25, 0.625, 0.875]
    for x in xs
      for y in ys
        ψ_fx_x = real(evaluate(ψ_fx, [x, y]))
        ψ_gy_y = real(evaluate(ψ_gy, [x, y]))
        @test ψ_fx_x ≈ cos(0.25 * pi * x)
        @test ψ_gy_y ≈ sin(0.5 * pi * y)
        ψ_fxgy_xy = real(evaluate(ψ_fxgy, [x, y]))
        @test ψ_fxgy_xy ≈ cos(0.25 * pi * x) * sin(0.5 * pi * y) atol = 1e-3
        ψ_sq_xy = real(evaluate(ψ_sq, [x, y]))
        @test ψ_sq_xy ≈ (sin(0.5 * pi * y))^2 atol = 1e-3
      end
    end
  end

  @testset "test operator_proj in 1D" begin
    g = named_comb_tree((4, 3))
    L = nv(g)
    s = continuous_siteinds(g)

    ψ_gx = sin_itn(s; k=0.5 * Number(pi))
    ψ_fx = cos_itn(s; k=0.25 * Number(pi))
    O = operator_proj(ψ_fx)

    ψ_fxgx = operate(O, ψ_gx; cutoff=1e-14)
    ψ_sq = operate(O, ψ_fx; cutoff=1e-14)
    xs = [0.025, 0.1, 0.25, 0.625, 0.875]
    for x in xs
      ψ_fxgx_x = real(evaluate(ψ_fxgx, x))
      @test ψ_fxgx_x ≈ sin(0.5 * pi * x) * cos(0.25 * pi * x) atol = 1e-3
      ψ_sq_x = real(evaluate(ψ_sq, x))
      @test ψ_sq_x ≈ cos(0.25 * pi * x) * cos(0.25 * pi * x) atol = 1e-3
    end
  end

  @testset "test shift operators in 1D on Tree" begin
    g = named_comb_tree((2, 3))
    L = nv(g)
    delta = 2.0^(-1.0 * L)
    s = continuous_siteinds(g)
    xs = [0.0, delta, 0.25, 0.5, 0.625, 0.875, 1.0 - delta]
    ψ_fx = poly_itn(s, [1.0, 0.5, 0.25])

    forward_shift_dirichlet = forward_shift_op(
      s; boundary="Dirichlet", truncate_kwargs=(; cutoff=1e-10)
    )
    backward_shift_dirichlet = backward_shift_op(
      s; boundary="Dirichlet", truncate_kwargs=(; cutoff=1e-10)
    )
    forward_shift_pbc = forward_shift_op(
      s; boundary="Periodic", truncate_kwargs=(; cutoff=1e-10)
    )
    backward_shift_pbc = backward_shift_op(
      s; boundary="Periodic", truncate_kwargs=(; cutoff=1e-10)
    )
    forward_shift_neumann = forward_shift_op(
      s; boundary="Neumann", truncate_kwargs=(; cutoff=1e-10)
    )
    backward_shift_neumann = backward_shift_op(
      s; boundary="Neumann", truncate_kwargs=(; cutoff=1e-10)
    )

    ψ_fx_pshift_dirichlet = operate(forward_shift_dirichlet, ψ_fx; cutoff=1e-12)
    ψ_fx_mshift_dirichlet = operate(backward_shift_dirichlet, ψ_fx; cutoff=1e-12)
    ψ_fx_pshift_pbc = operate(forward_shift_pbc, ψ_fx; cutoff=1e-12)
    ψ_fx_mshift_pbc = operate(backward_shift_pbc, ψ_fx; cutoff=1e-12)
    ψ_fx_pshift_neumann = operate(forward_shift_neumann, ψ_fx; cutoff=1e-12)
    ψ_fx_mshift_neumann = operate(backward_shift_neumann, ψ_fx; cutoff=1e-12)

    for x in xs
      if x + delta < 1
        fx_xplus = evaluate(ψ_fx, x + delta)
        @test fx_xplus ≈ evaluate(ψ_fx_pshift_dirichlet, x) atol = 1e-8
        @test fx_xplus ≈ evaluate(ψ_fx_pshift_pbc, x) atol = 1e-8
        @test fx_xplus ≈ evaluate(ψ_fx_pshift_neumann, x) atol = 1e-8
      elseif x == 1.0 - delta
        @test evaluate(ψ_fx_pshift_dirichlet, x) ≈ 0.0 atol = 1e-8
        @test evaluate(ψ_fx_pshift_pbc, x) ≈ evaluate(ψ_fx, 0.0) atol = 1e-8
        @test evaluate(ψ_fx_pshift_neumann, x) ≈ evaluate(ψ_fx, 1.0 - delta) atol = 1e-8
      end

      if x - delta >= 0.0
        fx_xminus = evaluate(ψ_fx, x - delta)
        @test fx_xminus ≈ evaluate(ψ_fx_mshift_dirichlet, x) atol = 1e-8
        @test fx_xminus ≈ evaluate(ψ_fx_mshift_pbc, x) atol = 1e-8
        @test fx_xminus ≈ evaluate(ψ_fx_mshift_neumann, x) atol = 1e-8
      elseif x == 0.0
        @test evaluate(ψ_fx_mshift_dirichlet, x) ≈ 0.0 atol = 1e-8
        @test evaluate(ψ_fx_mshift_pbc, x) ≈ evaluate(ψ_fx, 1.0 - delta) atol = 1e-8
        @test evaluate(ψ_fx_mshift_neumann, x) ≈ evaluate(ψ_fx, 0.0) atol = 1e-8
      end
    end
  end

  @testset "test double shift operators in 1D on Tree" begin
    g = named_comb_tree((2, 3))
    L = nv(g)
    delta = 2.0^(-1.0 * L)
    s = continuous_siteinds(g)
    xs = [0.0, delta, 0.25, 0.5, 0.625, 0.875, 1.0 - delta]
    ψ_fx = poly_itn(s, [1.0, 0.5, 0.25])
    n = 1

    forward_shift_dirichlet = forward_shift_op(
      s; n, boundary="Dirichlet", truncate_kwargs=(; cutoff=1e-10)
    )
    backward_shift_dirichlet = backward_shift_op(
      s; n, boundary="Dirichlet", truncate_kwargs=(; cutoff=1e-10)
    )
    forward_shift_pbc = forward_shift_op(
      s; n, boundary="Periodic", truncate_kwargs=(; cutoff=1e-10)
    )
    backward_shift_pbc = backward_shift_op(
      s; n, boundary="Periodic", truncate_kwargs=(; cutoff=1e-10)
    )
    forward_shift_neumann = forward_shift_op(
      s; n, boundary="Neumann", truncate_kwargs=(; cutoff=1e-10)
    )
    backward_shift_neumann = backward_shift_op(
      s; n, boundary="Neumann", truncate_kwargs=(; cutoff=1e-10)
    )

    ψ_fx_pshift_dirichlet = operate(forward_shift_dirichlet, ψ_fx; cutoff=1e-12)
    ψ_fx_mshift_dirichlet = operate(backward_shift_dirichlet, ψ_fx; cutoff=1e-12)
    ψ_fx_pshift_pbc = operate(forward_shift_pbc, ψ_fx; cutoff=1e-12)
    ψ_fx_mshift_pbc = operate(backward_shift_pbc, ψ_fx; cutoff=1e-12)
    ψ_fx_pshift_neumann = operate(forward_shift_neumann, ψ_fx; cutoff=1e-12)
    ψ_fx_mshift_neumann = operate(backward_shift_neumann, ψ_fx; cutoff=1e-12)

    for x in xs
      if x + 2.0 * delta < 1
        fx_xplus = evaluate(ψ_fx, x + 2.0 * delta)
        @test fx_xplus ≈ evaluate(ψ_fx_pshift_dirichlet, x) atol = 1e-8
        @test fx_xplus ≈ evaluate(ψ_fx_pshift_pbc, x) atol = 1e-8
        @test fx_xplus ≈ evaluate(ψ_fx_pshift_neumann, x) atol = 1e-8
      elseif x == 1.0 - 2.0 * delta || x == 1.0 - delta
        @test evaluate(ψ_fx_pshift_dirichlet, x; alg="exact") ≈ 0.0 atol = 1e-8
        @test evaluate(ψ_fx_pshift_pbc, x) ≈ evaluate(ψ_fx, x + 2.0 * delta - 1.0) atol =
          1e-8
        @test evaluate(ψ_fx_pshift_neumann, x) ≈ evaluate(ψ_fx, x) atol = 1e-8
      end

      if x - 2.0 * delta >= 0.0
        fx_xminus = evaluate(ψ_fx, x - 2.0 * delta)
        @test fx_xminus ≈ evaluate(ψ_fx_mshift_dirichlet, x) atol = 1e-8
        @test fx_xminus ≈ evaluate(ψ_fx_mshift_pbc, x) atol = 1e-8
        @test fx_xminus ≈ evaluate(ψ_fx_mshift_neumann, x) atol = 1e-8
      else
        @test evaluate(ψ_fx_mshift_dirichlet, x) ≈ 0.0 atol = 1e-8
        @test evaluate(ψ_fx_mshift_pbc, x) ≈ evaluate(ψ_fx, x - 2.0 * delta + 1.0) atol =
          1e-8
        @test evaluate(ψ_fx_mshift_neumann, x) ≈ evaluate(ψ_fx, x) atol = 1e-8
      end
    end
  end

  @testset "test shift operators in 2D on Tree" begin
    g = named_comb_tree((3, 3))
    s = continuous_siteinds(g; map_dimension=2)
    L = length(dimension_inds(s, 2))
    delta = 2.0^(-1.0 * L)
    x = 0.5
    ys = [0.0, delta, 0.25, 0.5, 0.625, 0.875, 1.0 - delta]
    ψ_fx = poly_itn(s, [1.0, 0.5, 0.25]; dim=1)
    ψ_fy = cos_itn(s; dim=2)
    ψ_fxy = ψ_fx + ψ_fx

    forward_shift_dirichlet = forward_shift_op(
      s; boundary="Dirichlet", dim=2, truncate_kwargs=(; cutoff=1e-10)
    )
    backward_shift_dirichlet = backward_shift_op(
      s; boundary="Dirichlet", dim=2, truncate_kwargs=(; cutoff=1e-10)
    )
    forward_shift_pbc = forward_shift_op(
      s; boundary="Periodic", dim=2, truncate_kwargs=(; cutoff=1e-10)
    )
    backward_shift_pbc = backward_shift_op(
      s; boundary="Periodic", dim=2, truncate_kwargs=(; cutoff=1e-10)
    )
    forward_shift_neumann = forward_shift_op(
      s; boundary="Neumann", dim=2, truncate_kwargs=(; cutoff=1e-10)
    )
    backward_shift_neumann = backward_shift_op(
      s; boundary="Neumann", dim=2, truncate_kwargs=(; cutoff=1e-10)
    )

    ψ_fxy_pshift_dirichlet = operate(forward_shift_dirichlet, ψ_fxy; cutoff=1e-12)
    ψ_fxy_mshift_dirichlet = operate(backward_shift_dirichlet, ψ_fxy; cutoff=1e-12)
    ψ_fxy_pshift_pbc = operate(forward_shift_pbc, ψ_fxy; cutoff=1e-12)
    ψ_fxy_mshift_pbc = operate(backward_shift_pbc, ψ_fxy; cutoff=1e-12)
    ψ_fxy_pshift_neumann = operate(forward_shift_neumann, ψ_fxy; cutoff=1e-12)
    ψ_fxy_mshift_neumann = operate(backward_shift_neumann, ψ_fxy; cutoff=1e-12)

    for y in ys
      if y + delta < 1
        fxy_xyplus = evaluate(ψ_fxy, [x, y + delta])
        @test fxy_xyplus ≈ evaluate(ψ_fxy_pshift_dirichlet, [x, y]) atol = 1e-8
        @test fxy_xyplus ≈ evaluate(ψ_fxy_pshift_pbc, [x, y]) atol = 1e-8
        @test fxy_xyplus ≈ evaluate(ψ_fxy_pshift_neumann, [x, y]) atol = 1e-8
      elseif y == 1.0 - delta
        @test evaluate(ψ_fxy_pshift_dirichlet, [x, y]) ≈ 0.0 atol = 1e-8
        @test evaluate(ψ_fxy_pshift_pbc, [x, y]) ≈ evaluate(ψ_fxy, [x, 0.0]) atol = 1e-8
        @test evaluate(ψ_fxy_pshift_neumann, [x, y]) ≈ evaluate(ψ_fxy, [x, 1.0 - delta]) atol =
          1e-8
      end

      if y - delta >= 0.0
        fxy_xyminus = evaluate(ψ_fxy, [x, y - delta])
        @test fxy_xyminus ≈ evaluate(ψ_fxy_mshift_dirichlet, [x, y]) atol = 1e-8
        @test fxy_xyminus ≈ evaluate(ψ_fxy_mshift_pbc, [x, y]) atol = 1e-8
        @test fxy_xyminus ≈ evaluate(ψ_fxy_mshift_neumann, [x, y]) atol = 1e-8
      elseif y == 0.0
        @test evaluate(ψ_fxy_mshift_dirichlet, [x, y]) ≈ 0.0 atol = 1e-8
        @test evaluate(ψ_fxy_mshift_pbc, [x, y]) ≈ evaluate(ψ_fxy, [x, 1.0 - delta]) atol =
          1e-8
        @test evaluate(ψ_fxy_mshift_neumann, [x, y]) ≈ evaluate(ψ_fxy, [x, 0.0]) atol = 1e-8
      end
    end
  end

  @testset "test boundary operator in 1D on Tree" begin
    g = named_comb_tree((2, 3))
    L = nv(g)
    delta = 2.0^(-1.0 * L)
    lastDigit = 1 - delta
    s = continuous_siteinds(g)

    xs = [0.0, delta, 0.25, 0.5, 0.625, 0.875, lastDigit]
    ψ_fx = poly_itn(s, [1.0, 0.5, 0.25])

    Zo = map_to_zero_operator(s, [0, lastDigit])

    @testset "corner boundary test" begin
      for p1 in [0, lastDigit]
        p = (itensornetwork(delta_p(s, p1)))
        @test inner(p, Zo, p) ≈ 0.0
      end
    end
    @testset "boundary apply" begin
      maxdim, cutoff = 10, 1e-16
      ϕ_fx = map_to_zeros(ψ_fx, [0, lastDigit]; cutoff, maxdim)
      for x in xs
        val = real(evaluate(ϕ_fx, x))
        @test (x ∈ [0, lastDigit]) ? isapprox(val, 0.0; atol=1e-8) : !(val ≈ 0.0)
      end
    end
  end

  @testset "test boundary operator in 2D on Tree" begin
    g = named_comb_tree((3, 3))
    s = continuous_siteinds(g; map_dimension=2)
    L = length(dimension_inds(s, 2))
    delta = 2.0^(-1.0 * L)
    lastDigit = 1 - delta

    ys = [0.0, delta, 0.25, 0.5, 0.625, 0.875, lastDigit]
    ψ_fx = poly_itn(s, [1.0, 0.5, 0.25]; dim=1)
    ψ_fy = cos_itn(s; dim=2)
    ψ_fxy = ψ_fx + ψ_fy

    Zo = map_to_zero_operator(s, [0, lastDigit, 0, lastDigit], [1, 1, 2, 2])
    @testset "corner boundary test" begin
      for p1 in [0, lastDigit]
        for p2 in [0, lastDigit]
          p = itensornetwork(delta_p(s, [p1, p2]))
          @test inner(p, Zo, p) ≈ 0.0
        end
      end
    end

    @testset "boundary apply" begin
      maxdim, cutoff = 10, 0e-16
      ϕ_fxy = operate([Zo], ψ_fxy; cutoff, maxdim, normalize=false)
      for x in [0, lastDigit]
        vals = zeros(length(ys))
        for (i, y) in enumerate(ys)
          vals[i] = real(evaluate(ϕ_fxy, [x, y]))
        end
        @test all(isapprox.(vals, 0.0, atol=1e-8))
      end
    end
  end
  @testset "test delta-kernel " begin
    @testset "test delta-kernel in 1D" begin
      g = named_comb_tree((2, 3))
      L = nv(g)
      delta = 2.0^(-1.0 * L)
      lastDigit = 1 - delta
      s = continuous_siteinds(g)

      xs = [0.0, delta, 0.25, 0.625, 0.875, lastDigit]
      ψ_fx = delta_kernel(s, [[0.5]]; coeff=-1, include_identity=true)
      @test evaluate(ψ_fx, [0.5]) ≈ 0
      for x in xs
        @test evaluate(ψ_fx, [x]) ≈ 1
      end
    end
    @testset "test delta-kernel in 2D" begin
      g = named_comb_tree((2, 3))
      L = nv(g)
      delta = 2.0^(-1.0 * (L ÷ 2))
      lastDigit = 1 - delta
      s = continuous_siteinds(g; map_dimension=2)

      xs = [0.0, delta, 0.25, 0.625, 0.875, lastDigit]
      @testset "insersecting lines" begin
        ψ_f = delta_kernel(s, [[0.5], [0.5]], [[1], [2]]; coeff=-1, include_identity=true)
        @test evaluate(ψ_f, [0.5, 0.5]) ≈ 0
        for x in xs
          @test evaluate(ψ_f, [x, 0.5]) ≈ 0
          @test evaluate(ψ_f, [0.5, x], [1, 2]) ≈ 0

          @test evaluate(ψ_f, [x, delta], [1, 2]) ≈ 1
          @test evaluate(ψ_f, [delta, x], [1, 2]) ≈ 1
        end
      end
      @testset "line and point" begin
        ψ_f = delta_kernel(
          s, [[0.5], [0.5, 0.1]], [[1], [1, 2]]; coeff=-1, include_identity=true
        )
        @test evaluate(ψ_f, [0.5, 0.5]) ≈ 0
        @test evaluate(ψ_f, [0.5, 0.1]) ≈ 0
        for x in xs
          @test evaluate(ψ_f, [0.5, x], [1, 2]) ≈ 0

          @test evaluate(ψ_f, [x, delta], [1, 2]) ≈ 1
          @test evaluate(ψ_f, [delta, x], [1, 2]) ≈ 1
        end
      end
    end

    @testset "test delta-kernel in 3D" begin
      g = named_comb_tree((3, 2))
      L = nv(g)
      delta = 2.0^(-1.0 * (L ÷ 3))
      lastDigit = 1 - delta
      s = continuous_siteinds(g; map_dimension=3)

      xs = [0.0, delta, lastDigit]
      zs = [0, delta, 0.5, lastDigit]
      @testset "insersecting planes" begin
        ψ_f = delta_kernel(s, [[0.5], [0.5]], [[1], [2]]; coeff=-1, include_identity=true)
        for z in zs
          @test evaluate(ψ_f, [0.5, 0.5, z]) ≈ 0
          for x in xs
            @test evaluate(ψ_f, [x, 0.5, z]) ≈ 0
            @test evaluate(ψ_f, [0.5, x, z], [1, 2, 3]) ≈ 0

            @test evaluate(ψ_f, [x, delta, z], [1, 2, 3]) ≈ 1
            @test evaluate(ψ_f, [delta, x, z], [1, 2, 3]) ≈ 1
          end
        end
      end
      @testset "plane and line" begin
        ψ_f = delta_kernel(
          s, [[0.5], [0.5, 0]], [[1], [1, 2]]; coeff=-1, include_identity=true
        )
        for z in zs
          @test evaluate(ψ_f, [0.5, 0.5, z]) ≈ 0
          @test evaluate(ψ_f, [0.5, 0, z]) ≈ 0
          for x in xs
            @test evaluate(ψ_f, [0.5, x, z], [1, 2, 3]) ≈ 0

            @test evaluate(ψ_f, [x, delta, z], [1, 2, 3]) ≈ 1
            @test evaluate(ψ_f, [delta, x, z], [1, 2, 3]) ≈ 1
          end
        end
      end
      @testset "two lines (w/ point overlap at endpoint)" begin
        ψ_f = delta_kernel(
          s, [[0.5, 0], [0, 0.5]], [[2, 3], [1, 2]]; coeff=-1, include_identity=true
        )
        @test evaluate(ψ_f, [0.0, 0.5, 0.5]) ≈ 0
        for z in [0]
          @test evaluate(ψ_f, [0.0, 0.5, z]) ≈ 0
          for x in xs
            @test evaluate(ψ_f, [x, 0.5, z], [1, 2, 3]) ≈ 0

            @test evaluate(ψ_f, [x, delta, z], [1, 2, 3]) ≈ 1
            @test evaluate(ψ_f, [delta, x, z], [1, 2, 3]) ≈ 1
          end
        end
      end
    end
  end
end
