using Test
using ITensorNumericalAnalysis

using ITensors: siteinds
using ITensorNetworks: maxlinkdim
using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: named_grid, named_comb_tree, NamedGraph, nv, vertices
using ITensorNumericalAnalysis: itensornetwork
using Dictionaries: Dictionary

using ITensorNumericalAnalysis: backward_shift_op, forward_shift_op

@testset "test shift operators in 1D on Tree" begin
  g = named_comb_tree((2, 3))
  L = nv(g)
  delta = 2.0^(-1.0 * L)
  bit_map = BitMap(g)
  s = siteinds(g, bit_map)
  xs = [0.0, delta, 0.25, 0.5, 0.625, 0.875, 1.0 - delta]
  ψ_fx = poly_itn(s, bit_map, [1.0, 0.5, 0.25])

  forward_shift_dirichlet = forward_shift_op(
    s, bit_map; boundary="Dirichlet", truncate_kwargs=(; cutoff=1e-10)
  )
  backward_shift_dirichlet = backward_shift_op(
    s, bit_map; boundary="Dirichlet", truncate_kwargs=(; cutoff=1e-10)
  )
  forward_shift_pbc = forward_shift_op(
    s, bit_map; boundary="Periodic", truncate_kwargs=(; cutoff=1e-10)
  )
  backward_shift_pbc = backward_shift_op(
    s, bit_map; boundary="Periodic", truncate_kwargs=(; cutoff=1e-10)
  )
  forward_shift_neumann = forward_shift_op(
    s, bit_map; boundary="Neumann", truncate_kwargs=(; cutoff=1e-10)
  )
  backward_shift_neumann = backward_shift_op(
    s, bit_map; boundary="Neumann", truncate_kwargs=(; cutoff=1e-10)
  )

  ψ_fx_pshift_dirichlet = operate(forward_shift_dirichlet, ψ_fx; cutoff=1e-12)
  ψ_fx_mshift_dirichlet = operate(backward_shift_dirichlet, ψ_fx; cutoff=1e-12)
  ψ_fx_pshift_pbc = operate(forward_shift_pbc, ψ_fx; cutoff=1e-12)
  ψ_fx_mshift_pbc = operate(backward_shift_pbc, ψ_fx; cutoff=1e-12)
  ψ_fx_pshift_neumann = operate(forward_shift_neumann, ψ_fx; cutoff=1e-12)
  ψ_fx_mshift_neumann = operate(backward_shift_neumann, ψ_fx; cutoff=1e-12)

  for x in xs
    if x + delta < 1
      fx_xplus = calculate_fx(ψ_fx, x + delta)
      @test fx_xplus ≈ calculate_fx(ψ_fx_pshift_dirichlet, x) atol = 1e-8
      @test fx_xplus ≈ calculate_fx(ψ_fx_pshift_pbc, x) atol = 1e-8
      @test fx_xplus ≈ calculate_fx(ψ_fx_pshift_neumann, x) atol = 1e-8
    elseif x == 1.0 - delta
      @test calculate_fx(ψ_fx_pshift_dirichlet, x) ≈ 0.0 atol = 1e-8
      @test calculate_fx(ψ_fx_pshift_pbc, x) ≈ calculate_fx(ψ_fx, 0.0) atol = 1e-8
      @test calculate_fx(ψ_fx_pshift_neumann, x) ≈ calculate_fx(ψ_fx, 1.0 - delta) atol =
        1e-8
    end

    if x - delta >= 0.0
      fx_xminus = calculate_fx(ψ_fx, x - delta)
      @test fx_xminus ≈ calculate_fx(ψ_fx_mshift_dirichlet, x) atol = 1e-8
      @test fx_xminus ≈ calculate_fx(ψ_fx_mshift_pbc, x) atol = 1e-8
      @test fx_xminus ≈ calculate_fx(ψ_fx_mshift_neumann, x) atol = 1e-8
    elseif x == 0.0
      @test calculate_fx(ψ_fx_mshift_dirichlet, x) ≈ 0.0 atol = 1e-8
      @test calculate_fx(ψ_fx_mshift_pbc, x) ≈ calculate_fx(ψ_fx, 1.0 - delta) atol = 1e-8
      @test calculate_fx(ψ_fx_mshift_neumann, x) ≈ calculate_fx(ψ_fx, 0.0) atol = 1e-8
    end
  end
end

@testset "test double shift operators in 1D on Tree" begin
  g = named_comb_tree((2, 3))
  L = nv(g)
  delta = 2.0^(-1.0 * L)
  bit_map = BitMap(g)
  s = siteinds(g, bit_map)
  xs = [0.0, delta, 0.25, 0.5, 0.625, 0.875, 1.0 - delta]
  ψ_fx = poly_itn(s, bit_map, [1.0, 0.5, 0.25])
  n = 1

  forward_shift_dirichlet = forward_shift_op(
    s, bit_map; n, boundary="Dirichlet", truncate_kwargs=(; cutoff=1e-10)
  )
  backward_shift_dirichlet = backward_shift_op(
    s, bit_map; n, boundary="Dirichlet", truncate_kwargs=(; cutoff=1e-10)
  )
  forward_shift_pbc = forward_shift_op(
    s, bit_map; n, boundary="Periodic", truncate_kwargs=(; cutoff=1e-10)
  )
  backward_shift_pbc = backward_shift_op(
    s, bit_map; n, boundary="Periodic", truncate_kwargs=(; cutoff=1e-10)
  )
  forward_shift_neumann = forward_shift_op(
    s, bit_map; n, boundary="Neumann", truncate_kwargs=(; cutoff=1e-10)
  )
  backward_shift_neumann = backward_shift_op(
    s, bit_map; n, boundary="Neumann", truncate_kwargs=(; cutoff=1e-10)
  )

  ψ_fx_pshift_dirichlet = operate(forward_shift_dirichlet, ψ_fx; cutoff=1e-12)
  ψ_fx_mshift_dirichlet = operate(backward_shift_dirichlet, ψ_fx; cutoff=1e-12)
  ψ_fx_pshift_pbc = operate(forward_shift_pbc, ψ_fx; cutoff=1e-12)
  ψ_fx_mshift_pbc = operate(backward_shift_pbc, ψ_fx; cutoff=1e-12)
  ψ_fx_pshift_neumann = operate(forward_shift_neumann, ψ_fx; cutoff=1e-12)
  ψ_fx_mshift_neumann = operate(backward_shift_neumann, ψ_fx; cutoff=1e-12)

  for x in xs
    if x + 2.0 * delta < 1
      fx_xplus = calculate_fx(ψ_fx, x + 2.0 * delta)
      @test fx_xplus ≈ calculate_fx(ψ_fx_pshift_dirichlet, x) atol = 1e-8
      @test fx_xplus ≈ calculate_fx(ψ_fx_pshift_pbc, x) atol = 1e-8
      @test fx_xplus ≈ calculate_fx(ψ_fx_pshift_neumann, x) atol = 1e-8
    elseif x == 1.0 - 2.0 * delta || x == 1.0 - delta
      @test calculate_fx(ψ_fx_pshift_dirichlet, x) ≈ 0.0 atol = 1e-8
      @test calculate_fx(ψ_fx_pshift_pbc, x) ≈ calculate_fx(ψ_fx, x + 2.0 * delta - 1.0) atol =
        1e-8
      @test calculate_fx(ψ_fx_pshift_neumann, x) ≈ calculate_fx(ψ_fx, x) atol = 1e-8
    end

    if x - 2.0 * delta >= 0.0
      fx_xminus = calculate_fx(ψ_fx, x - 2.0 * delta)
      @test fx_xminus ≈ calculate_fx(ψ_fx_mshift_dirichlet, x) atol = 1e-8
      @test fx_xminus ≈ calculate_fx(ψ_fx_mshift_pbc, x) atol = 1e-8
      @test fx_xminus ≈ calculate_fx(ψ_fx_mshift_neumann, x) atol = 1e-8
    else
      @test calculate_fx(ψ_fx_mshift_dirichlet, x) ≈ 0.0 atol = 1e-8
      @test calculate_fx(ψ_fx_mshift_pbc, x) ≈ calculate_fx(ψ_fx, x - 2.0 * delta + 1.0) atol =
        1e-8
      @test calculate_fx(ψ_fx_mshift_neumann, x) ≈ calculate_fx(ψ_fx, x) atol = 1e-8
    end
  end
end

@testset "test shift operators in 2D on Tree" begin
  g = named_comb_tree((3, 3))
  bit_map = BitMap(g; map_dimension=2)
  L = length(vertices(bit_map, 2))
  delta = 2.0^(-1.0 * L)
  s = siteinds(g, bit_map)
  x = 0.5
  ys = [0.0, delta, 0.25, 0.5, 0.625, 0.875, 1.0 - delta]
  ψ_fx = poly_itn(s, bit_map, [1.0, 0.5, 0.25]; dimension=1)
  ψ_fy = cos_itn(s, bit_map; dimension=2)
  ψ_fxy = ψ_fx + ψ_fx

  forward_shift_dirichlet = forward_shift_op(
    s, bit_map; boundary="Dirichlet", dimension=2, truncate_kwargs=(; cutoff=1e-10)
  )
  backward_shift_dirichlet = backward_shift_op(
    s, bit_map; boundary="Dirichlet", dimension=2, truncate_kwargs=(; cutoff=1e-10)
  )
  forward_shift_pbc = forward_shift_op(
    s, bit_map; boundary="Periodic", dimension=2, truncate_kwargs=(; cutoff=1e-10)
  )
  backward_shift_pbc = backward_shift_op(
    s, bit_map; boundary="Periodic", dimension=2, truncate_kwargs=(; cutoff=1e-10)
  )
  forward_shift_neumann = forward_shift_op(
    s, bit_map; boundary="Neumann", dimension=2, truncate_kwargs=(; cutoff=1e-10)
  )
  backward_shift_neumann = backward_shift_op(
    s, bit_map; boundary="Neumann", dimension=2, truncate_kwargs=(; cutoff=1e-10)
  )

  ψ_fxy_pshift_dirichlet = operate(forward_shift_dirichlet, ψ_fxy; cutoff=1e-12)
  ψ_fxy_mshift_dirichlet = operate(backward_shift_dirichlet, ψ_fxy; cutoff=1e-12)
  ψ_fxy_pshift_pbc = operate(forward_shift_pbc, ψ_fxy; cutoff=1e-12)
  ψ_fxy_mshift_pbc = operate(backward_shift_pbc, ψ_fxy; cutoff=1e-12)
  ψ_fxy_pshift_neumann = operate(forward_shift_neumann, ψ_fxy; cutoff=1e-12)
  ψ_fxy_mshift_neumann = operate(backward_shift_neumann, ψ_fxy; cutoff=1e-12)

  for y in ys
    if y + delta < 1
      fxy_xyplus = calculate_fxyz(ψ_fxy, [x, y + delta])
      @test fxy_xyplus ≈ calculate_fxyz(ψ_fxy_pshift_dirichlet, [x, y]) atol = 1e-8
      @test fxy_xyplus ≈ calculate_fxyz(ψ_fxy_pshift_pbc, [x, y]) atol = 1e-8
      @test fxy_xyplus ≈ calculate_fxyz(ψ_fxy_pshift_neumann, [x, y]) atol = 1e-8
    elseif y == 1.0 - delta
      @test calculate_fxyz(ψ_fxy_pshift_dirichlet, [x, y]) ≈ 0.0 atol = 1e-8
      @test calculate_fxyz(ψ_fxy_pshift_pbc, [x, y]) ≈ calculate_fxyz(ψ_fxy, [x, 0.0]) atol =
        1e-8
      @test calculate_fxyz(ψ_fxy_pshift_neumann, [x, y]) ≈
        calculate_fxyz(ψ_fxy, [x, 1.0 - delta]) atol = 1e-8
    end

    if y - delta >= 0.0
      fxy_xyminus = calculate_fxyz(ψ_fxy, [x, y - delta])
      @test fxy_xyminus ≈ calculate_fxyz(ψ_fxy_mshift_dirichlet, [x, y]) atol = 1e-8
      @test fxy_xyminus ≈ calculate_fxyz(ψ_fxy_mshift_pbc, [x, y]) atol = 1e-8
      @test fxy_xyminus ≈ calculate_fxyz(ψ_fxy_mshift_neumann, [x, y]) atol = 1e-8
    elseif y == 0.0
      @test calculate_fxyz(ψ_fxy_mshift_dirichlet, [x, y]) ≈ 0.0 atol = 1e-8
      @test calculate_fxyz(ψ_fxy_mshift_pbc, [x, y]) ≈
        calculate_fxyz(ψ_fxy, [x, 1.0 - delta]) atol = 1e-8
      @test calculate_fxyz(ψ_fxy_mshift_neumann, [x, y]) ≈ calculate_fxyz(ψ_fxy, [x, 0.0]) atol =
        1e-8
    end
  end
end
