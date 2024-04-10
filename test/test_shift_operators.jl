using Test
using ITensorNumericalAnalysis

using ITensors: siteinds
using ITensorNetworks: maxlinkdim
using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: named_grid, named_comb_tree, NamedGraph, nv, vertices
using ITensorNumericalAnalysis: itensornetwork
using Dictionaries: Dictionary

using ITensorNumericalAnalysis: plus_shift_ttn, minus_shift_ttn

@testset "test shift operators in 1D on MPS" begin

  g = named_grid((8, 1))
  L = nv(g)
  delta = 2.0^(-1.0*L)
  bit_map = BitMap(g)
  s = siteinds(g, bit_map)
  xs = [0.0, delta, 0.25, 0.5, 0.625, 0.875, 1.0 - delta]
  ψ_fx = poly_itn(s, bit_map, [1.0, 0.5, 0.25])

  plus_shift_dirichlet = plus_shift_ttn(s, bit_map; boundary="Dirichlet")
  minus_shift_dirichlet = minus_shift_ttn(s, bit_map; boundary="Dirichlet")
  plus_shift_pbc = plus_shift_ttn(s, bit_map; boundary="Periodic")
  minus_shift_pbc = minus_shift_ttn(s, bit_map; boundary="Periodic")
  plus_shift_neumann = plus_shift_ttn(s, bit_map; boundary="Neumann")
  minus_shift_neumann = minus_shift_ttn(s, bit_map; boundary="Neumann")

  ψ_fx_pshift_dirichlet = operate(plus_shift_dirichlet, ψ_fx; truncate_kwargs=(; cutoff=1e-12))
  ψ_fx_mshift_dirichlet = operate(minus_shift_dirichlet, ψ_fx; truncate_kwargs=(; cutoff=1e-12))
  ψ_fx_pshift_pbc = operate(plus_shift_pbc, ψ_fx; truncate_kwargs=(; cutoff=1e-12))
  ψ_fx_mshift_pbc = operate(minus_shift_pbc, ψ_fx; truncate_kwargs=(; cutoff=1e-12))
  ψ_fx_pshift_neumann = operate(plus_shift_neumann, ψ_fx; truncate_kwargs=(; cutoff=1e-12))
  ψ_fx_mshift_neumann = operate(minus_shift_neumann, ψ_fx; truncate_kwargs=(; cutoff=1e-12))

  for x in xs
    if x + delta < 1
      fx_xplus = calculate_fx(ψ_fx, x + delta)
      @test fx_xplus ≈ calculate_fx(ψ_fx_pshift_dirichlet, x) atol = 1e-8
      @test fx_xplus ≈ calculate_fx(ψ_fx_pshift_pbc, x) atol = 1e-8
      @test fx_xplus ≈ calculate_fx(ψ_fx_pshift_neumann, x) atol = 1e-8
    elseif x == 1.0 - delta
      @test calculate_fx(ψ_fx_pshift_dirichlet, x) ≈ 0.0 atol = 1e-8
      @test calculate_fx(ψ_fx_pshift_pbc, x) ≈ calculate_fx(ψ_fx, 0.0) atol = 1e-8
      @test calculate_fx(ψ_fx_pshift_neumann, x) ≈ calculate_fx(ψ_fx, 1.0 - delta) atol = 1e-8
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


@testset "test shift operators in 1D on Tree" begin

    g = named_comb_tree((3,3))
    L = nv(g)
    delta = 2.0^(-1.0*L)
    bit_map = BitMap(g)
    s = siteinds(g, bit_map)
    xs = [0.0, delta, 0.25, 0.5, 0.625, 0.875, 1.0 - delta]
    ψ_fx = poly_itn(s, bit_map, [1.0, 0.5, 0.25])
  
    plus_shift_dirichlet = plus_shift_ttn(s, bit_map; boundary="Dirichlet")
    minus_shift_dirichlet = minus_shift_ttn(s, bit_map; boundary="Dirichlet")
    plus_shift_pbc = plus_shift_ttn(s, bit_map; boundary="Periodic")
    minus_shift_pbc = minus_shift_ttn(s, bit_map; boundary="Periodic")
    plus_shift_neumann = plus_shift_ttn(s, bit_map; boundary="Neumann")
    minus_shift_neumann = minus_shift_ttn(s, bit_map; boundary="Neumann")
  
    ψ_fx_pshift_dirichlet = operate(plus_shift_dirichlet, ψ_fx; truncate_kwargs=(; cutoff=1e-12))
    ψ_fx_mshift_dirichlet = operate(minus_shift_dirichlet, ψ_fx; truncate_kwargs=(; cutoff=1e-12))
    ψ_fx_pshift_pbc = operate(plus_shift_pbc, ψ_fx; truncate_kwargs=(; cutoff=1e-12))
    ψ_fx_mshift_pbc = operate(minus_shift_pbc, ψ_fx; truncate_kwargs=(; cutoff=1e-12))
    ψ_fx_pshift_neumann = operate(plus_shift_neumann, ψ_fx; truncate_kwargs=(; cutoff=1e-12))
    ψ_fx_mshift_neumann = operate(minus_shift_neumann, ψ_fx; truncate_kwargs=(; cutoff=1e-12))
  
    for x in xs
      if x + delta < 1
        fx_xplus = calculate_fx(ψ_fx, x + delta)
        @test fx_xplus ≈ calculate_fx(ψ_fx_pshift_dirichlet, x) atol = 1e-8
        @test fx_xplus ≈ calculate_fx(ψ_fx_pshift_pbc, x) atol = 1e-8
        @test fx_xplus ≈ calculate_fx(ψ_fx_pshift_neumann, x) atol = 1e-8
      elseif x == 1.0 - delta
        @test calculate_fx(ψ_fx_pshift_dirichlet, x) ≈ 0.0 atol = 1e-8
        @test calculate_fx(ψ_fx_pshift_pbc, x) ≈ calculate_fx(ψ_fx, 0.0) atol = 1e-8
        @test calculate_fx(ψ_fx_pshift_neumann, x) ≈ calculate_fx(ψ_fx, 1.0 - delta) atol = 1e-8
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

@testset "test shift operators in 2D on Tree" begin

    g = named_comb_tree((3,3))
    bit_map = BitMap(g; map_dimension = 2)
    L = length(vertices(bit_map, 2))
    delta = 2.0^(-1.0*L)
    s = siteinds(g, bit_map)
    x = 0.5
    ys = [0.0, delta, 0.25, 0.5, 0.625, 0.875, 1.0 - delta]
    ψ_fx = poly_itn(s, bit_map, [1.0, 0.5, 0.25]; dimension = 1) 
    ψ_fy = cos_itn(s, bit_map; dimension = 2)
    ψ_fxy = ψ_fx + ψ_fx
  
    plus_shift_dirichlet = plus_shift_ttn(s, bit_map; dimension = 2, boundary="Dirichlet")
    minus_shift_dirichlet = minus_shift_ttn(s, bit_map; dimension = 2, boundary="Dirichlet")
    plus_shift_pbc = plus_shift_ttn(s, bit_map; dimension = 2, boundary="Periodic")
    minus_shift_pbc = minus_shift_ttn(s, bit_map; dimension = 2, boundary="Periodic")
    plus_shift_neumann = plus_shift_ttn(s, bit_map; dimension = 2, boundary="Neumann")
    minus_shift_neumann = minus_shift_ttn(s, bit_map; dimension = 2, boundary="Neumann")
  
    ψ_fxy_pshift_dirichlet = operate(plus_shift_dirichlet, ψ_fxy; truncate_kwargs=(; cutoff=1e-12))
    ψ_fxy_mshift_dirichlet = operate(minus_shift_dirichlet, ψ_fxy; truncate_kwargs=(; cutoff=1e-12))
    ψ_fxy_pshift_pbc = operate(plus_shift_pbc, ψ_fxy; truncate_kwargs=(; cutoff=1e-12))
    ψ_fxy_mshift_pbc = operate(minus_shift_pbc, ψ_fxy; truncate_kwargs=(; cutoff=1e-12))
    ψ_fxy_pshift_neumann = operate(plus_shift_neumann, ψ_fxy; truncate_kwargs=(; cutoff=1e-12))
    ψ_fxy_mshift_neumann = operate(minus_shift_neumann, ψ_fxy; truncate_kwargs=(; cutoff=1e-12))
  
    for y in ys
      if y + delta < 1
        fxy_xyplus = calculate_fxy(ψ_fxy, [x, y + delta])
        @test fxy_xyplus ≈ calculate_fxy(ψ_fxy_pshift_dirichlet, [x,y]) atol = 1e-8
        @test fxy_xyplus ≈ calculate_fxy(ψ_fxy_pshift_pbc, [x,y]) atol = 1e-8
        @test fxy_xyplus ≈ calculate_fxy(ψ_fxy_pshift_neumann, [x,y]) atol = 1e-8
      elseif y == 1.0 - delta
        @test calculate_fxy(ψ_fxy_pshift_dirichlet, [x,y]) ≈ 0.0 atol = 1e-8
        @test calculate_fx(ψ_fxy_pshift_pbc, [x,y]) ≈ calculate_fxy(ψ_fxy, [x, 0.0]) atol = 1e-8
        @test calculate_fx(ψ_fxy_pshift_neumann, [x,y]) ≈ calculate_fxy(ψ_fxy, [x, 1.0 - delta]) atol = 1e-8
      end
  
    #   if x - delta >= 0.0
    #       fx_xminus = calculate_fx(ψ_fx, x - delta)
    #       @test fx_xminus ≈ calculate_fx(ψ_fx_mshift_dirichlet, x) atol = 1e-8
    #       @test fx_xminus ≈ calculate_fx(ψ_fx_mshift_pbc, x) atol = 1e-8
    #       @test fx_xminus ≈ calculate_fx(ψ_fx_mshift_neumann, x) atol = 1e-8
    #   elseif x == 0.0
    #       @test calculate_fx(ψ_fx_mshift_dirichlet, x) ≈ 0.0 atol = 1e-8
    #       @test calculate_fx(ψ_fx_mshift_pbc, x) ≈ calculate_fx(ψ_fx, 1.0 - delta) atol = 1e-8
    #       @test calculate_fx(ψ_fx_mshift_neumann, x) ≈ calculate_fx(ψ_fx, 0.0) atol = 1e-8
    #   end
    end
end