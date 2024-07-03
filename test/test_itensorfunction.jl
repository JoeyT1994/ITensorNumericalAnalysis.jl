using Test
using ITensorNumericalAnalysis

using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: NamedGraph, vertices, rename_vertices
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree
using ITensors: siteinds
using Dictionaries: Dictionary
using Random: Random

Random.seed!(1234)

@testset "test itensorfunctions" begin
  @testset "test constructor from ITensorNetwork" begin
    L = 10

    g = named_grid((L, 1))
    s = continuous_siteinds(g)

    ψ = random_itensornetwork(s; link_space=2)

    fψ = ITensorNetworkFunction(ψ)

    @test dimension_vertices(fψ, 1) == vertices(fψ)
    @test dimension(fψ) == 1

    dim_vertices = [
      collect(filter(v -> first(v) < Int(0.5 * L), vertices(ψ))),
      collect(filter(v -> first(v) >= Int(0.5 * L), vertices(ψ))),
    ]
    fψ = ITensorNetworkFunction(ψ, dim_vertices)
    @test union(Set(dimension_vertices(fψ, 1)), Set(dimension_vertices(fψ, 2))) ==
      Set(vertices(fψ))
    @test dimension(fψ) == 2
  end

  @testset "test 1D elementary function construction" begin
    @testset "test const" begin
      L = 3
      g = named_grid((L, L))
      s = continuous_siteinds(g)
      c = 1.5

      ψ_fx = const_itn(s; c)

      x = 0.5
      fx_x = evaluate(ψ_fx, x; alg="exact")
      @test fx_x ≈ c

      # link dims section
      ψ_fx = const_itn(s; c, linkdim=4)

      x = 0.5
      fx_x = evaluate(ψ_fx, x; alg="exact")
      @test fx_x ≈ c
    end
    funcs = [
      ("cosh", cosh_itn, cosh),
      ("sinh", sinh_itn, sinh),
      ("exp", exp_itn, exp),
      ("cos", cos_itn, cos),
      ("sin", sin_itn, sin),
    ]
    for (name, net_func, func) in funcs
      @testset "test $name in binary" begin
        Lx, Ly = 2, 3
        g = named_comb_tree((2, 3))
        a = rand()
        k = rand()
        c = rand()

        s = continuous_siteinds(g)

        x = 0.625
        ψ_fx = net_func(s; k, a, c)
        fx_x = evaluate(ψ_fx, x)
        @test c * func(k * x + a) ≈ fx_x

        fx_x_sq = evaluate(ψ_fx * ψ_fx, x)
        @test c * c * func(k * x + a) * func(k * x + a) ≈ fx_x_sq
      end
    end

    funcs = [
      ("cosh", cosh_itn, cosh),
      ("sinh", sinh_itn, sinh),
      ("exp", exp_itn, exp),
      ("cos", cos_itn, cos),
      ("sin", sin_itn, sin),
    ]
    for (name, net_func, func) in funcs
      @testset "test $name in trinary" begin
        Lx, Ly = 2, 3
        g = named_comb_tree((2, 3))
        a = rand()
        k = rand()
        c = rand()

        s = continuous_siteinds(g; base=3)

        x = (5.0 / 9.0)
        ψ_fx = net_func(s; k, a, c)
        fx_x = evaluate(ψ_fx, x)
        @test c * func(k * x + a) ≈ fx_x
      end
    end

    @testset "test tanh" begin
      L = 10
      g = named_grid((L, 1))
      a = rand()
      k = rand()
      c = rand()
      nterms = 50

      s = continuous_siteinds(g)

      x = 0.625
      ψ_fx = tanh_itn(s; k, a, c, nterms)
      fx_x = evaluate(ψ_fx, x)

      @test c * tanh(k * x + a) ≈ fx_x
    end

    @testset "test poly" begin
      L = 6
      degrees = [i + 1 for i in 0:10]

      ###Generate a series of random polynomials on random graphs. Evaluate them at random x values"""
      for deg in degrees
        g = NamedGraph(SimpleGraph(uniform_tree(L)))
        g = rename_vertices(v -> (v, 1), g)
        s = continuous_siteinds(g)
        k = rand()
        c = rand()

        coeffs = [rand() for i in 1:(deg + 1)]

        x = 0.875
        ψ_fx = poly_itn(s, coeffs; k, c)
        fx_x = evaluate(ψ_fx, x)

        fx_exact = c * sum([coeffs[i] * ((k * x)^(i - 1)) for i in 1:(deg + 1)])
        @test fx_x ≈ fx_exact atol = 1e-4
      end
    end
  end

  @testset "test multi-dimensional elementary function construction" begin
    #Constant function but represented in three dimension
    @testset "test const" begin
      g = named_grid((3, 3))
      s = continuous_siteinds(g; map_dimension=3)

      c = 1.5

      ψ_fxyz = const_itn(s; c)

      x, y, z = 0.5, 0.25, 0.0

      fx_xyz = evaluate(ψ_fxyz, [x, y, z], [1, 2, 3])
      @test fx_xyz ≈ c
    end

    #Two dimensional functions as sum of two 1D functions
    funcs = [
      ("cosh", cosh_itn, cosh),
      ("sinh", sinh_itn, sinh),
      ("exp", exp_itn, exp),
      ("cos", cos_itn, cos),
      ("sin", sin_itn, sin),
    ]
    L = 10
    g = named_grid((L, 1))
    s = continuous_siteinds(g; map_dimension=2)
    x, y = 0.625, 0.25

    for (name, net_func, func) in funcs
      @testset "test $name" begin
        a = rand()
        k = rand()
        c = rand()

        ψ_fx = net_func(s; k, a, c, dim=1)
        ψ_fy = net_func(s; k, a, c, dim=2)

        ψ_fxy = ψ_fx + ψ_fy
        fxy_xy = evaluate(ψ_fxy, [x, y], [1, 2])
        @test c * func(k * x + a) + c * func(k * y + a) ≈ fxy_xy
      end
    end

    #Sum of two tanhs in two different directions
    @testset "test tanh" begin
      L = 3
      g = named_grid((L, 2))
      a = rand()
      k = rand()
      c = rand()
      nterms = 20
      s = continuous_siteinds(g; map_dimension=2)

      x, y = 0.625, 0.875
      ψ_fx = tanh_itn(s; k, a, c, nterms, dim=1)
      ψ_fy = tanh_itn(s; k, a, c, nterms, dim=2)

      ψ_fxy = ψ_fx + ψ_fy
      fxy_xy = evaluate(ψ_fxy, [x, y], [1, 2]; alg="exact")
      @test c * tanh(k * x + a) + c * tanh(k * y + a) ≈ fxy_xy
    end
  end
  @testset "test delta_p" begin
    L = 10
    g = named_grid((L, 1))
    s = continuous_siteinds(g; map_dimension=2)
    x0, y0 = 0.625, 0.25
    delta = 2.0^(-1.0 * L)
    lastDigit = 1 - delta
    xs = [0.0, delta, 0.25, 0.5, 0.625, 0.875, lastDigit]
    @testset "test single point" begin
      ψ = delta_p(s, [x0, y0])
      @test evaluate(ψ, [x0, y0], [1, 2]) ≈ 1
      # test another point
      @test evaluate(ψ, [y0, x0], [1, 2]) ≈ 0
    end
    @testset "test plane" begin
      ψ = delta_p(s, [y0], [2])

      # should be 1 in the plane
      for x in xs
        @test evaluate(ψ, [x, y0], [1, 2]) ≈ 1
      end
      # test random points
      for x in xs
        @test evaluate(ψ, [x, 0.5], [1, 2]) ≈ 0
      end
    end
    @testset "test sums of points" begin
      points = [[x0, y0], [y0, x0]]
      ψ = delta_p(s, points)
      @test evaluate(ψ, [x0, y0], [1, 2]) ≈ 1
      @test evaluate(ψ, [y0, x0], [1, 2]) ≈ 1
      # test other points
      @test evaluate(ψ, [0, 0], [1, 2]) ≈ 0
      @test evaluate(ψ, [0, y0], [1, 2]) ≈ 0
    end

    @testset "test sums of points and plane" begin
      p0 = 0.5
      points = [[x0, y0], [p0]]
      dims = [[1, 2], [2]]
      ψ = delta_p(s, points, dims)
      @test evaluate(ψ, [x0, y0], [1, 2]) ≈ 1
      for x in xs
        @test evaluate(ψ, [x, p0], [1, 2]) ≈ 1
      end
      ## test other points
      @test evaluate(ψ, [0, 0], [1, 2]) ≈ 0
      @test evaluate(ψ, [0, y0], [1, 2]) ≈ 0
    end
  end
end
