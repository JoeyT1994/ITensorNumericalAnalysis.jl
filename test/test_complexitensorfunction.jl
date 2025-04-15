using Test
using ITensorNumericalAnalysis
using TensorOperations: TensorOperations

using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: NamedGraph, vertices, rename_vertices
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree
using ITensors: siteinds
using Dictionaries: Dictionary
using Random: Random

Random.seed!(1234)

@testset "test complex itensorfunctions" begin
  @testset "test constructor from ITensorNetwork" begin
    L = 10

    g = named_grid((L, 1))
    s = complex_continuous_siteinds(g)

    ψ = random_itensornetwork(s; link_space=2)

    fψ = ITensorNetworkFunction(ψ)

    @test dimension_vertices(fψ, 1) == vertices(fψ)
    @test dimension(fψ) == 1

    dim_vertices = [
      collect(filter(v -> first(v) < Int(0.5 * L), vertices(ψ))),
      collect(filter(v -> first(v) >= Int(0.5 * L), vertices(ψ))),
    ]
    fψ = ITensorNetworkFunction(ψ, dim_vertices, dim_vertices)
    @test union(Set(dimension_vertices(fψ, 1)), Set(dimension_vertices(fψ, 2))) ==
      Set(vertices(fψ))
    @test dimension(fψ) == 2
  end

  @testset "test 1D elementary function construction" begin
    @testset "test const" begin
      L = 3
      g = named_grid((L, L))
      s = complex_continuous_siteinds(g)
      c = 1.5

      ψ_fz = const_itn(s; c)

      z = 0.5 + 0.625 * im
      fz_z = evaluate(ψ_fz, z) # exact
      @test fz_z ≈ c

      # link dims section
      ψ_fz = const_itn(s; c, linkdim=4)

      fz_z = evaluate(ψ_fz, z; alg="exact")
      @test fz_z ≈ c
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
        a = rand() + im * rand()
        k = rand() + im * rand()
        c = rand() + im * rand()

        #Put the imaginary and real indices on different vertices
        real_dimension_vertices = [[(1, 1), (1, 2), (1, 3)]]
        imag_dimension_vertices = [[(2, 1), (2, 2), (2, 3)]]
        s = complex_continuous_siteinds(g, real_dimension_vertices, imag_dimension_vertices)

        z = 0.625 + 0.25 * im
        ψ_fz = net_func(s; k, a, c)
        fz_z = evaluate(ψ_fz, z)
        @test c * func(k * z + a) ≈ fz_z
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
        a = rand() + im * rand()
        k = rand() + im * rand()
        c = rand() + im * rand()

        #Put the imaginary and real indices on different vertices
        real_dimension_vertices = [[(1, 1), (1, 2), (1, 3)]]
        imag_dimension_vertices = [[(2, 1), (2, 2), (2, 3)]]
        s = complex_continuous_siteinds(
          g, real_dimension_vertices, imag_dimension_vertices; base=3
        )

        z = (5.0 / 9.0) + (4.0 / 9.0) * im
        ψ_fz = net_func(s; k, a, c)
        fz_z = evaluate(ψ_fz, z)
        @test c * func(k * z + a) ≈ fz_z
      end
    end

    @testset "test tanh" begin
      L = 10
      g = named_grid((L, 1))
      a = rand() + im * rand()
      k = rand() + im * rand()
      c = rand() + im * rand()
      nterms = 50

      real_dimension_vertices = [[(i, 1) for i in 1:2:L]]
      imag_dimension_vertices = [[(i, 1) for i in 2:2:L]]
      s = complex_continuous_siteinds(g, real_dimension_vertices, imag_dimension_vertices)

      z = 0.625 + 0.125 * im
      ψ_fz = tanh_itn(s; k, a, c, nterms)
      fz_z = evaluate(ψ_fz, z)

      @test c * tanh(k * z + a) ≈ fz_z
    end

    @testset "test poly" begin
      L = 6
      degrees = [i + 1 for i in 0:10]

      ###Generate a series of random polynomials on random graphs. Evaluate them at random x values"""
      for deg in degrees
        g = NamedGraph(SimpleGraph(uniform_tree(L)))
        g = rename_vertices(v -> (v, 1), g)

        real_dimension_vertices = [[(i, 1) for i in 1:2:L]]
        imag_dimension_vertices = [[(i, 1) for i in 2:2:L]]
        s = complex_continuous_siteinds(g, real_dimension_vertices, imag_dimension_vertices)
        k = rand() + im * rand()
        c = rand() + im * rand()

        coeffs = [rand() + im * rand() for i in 1:(deg + 1)]

        z = 0.875 + 0.25 * im
        ψ_fz = poly_itn(s, coeffs; k, c)
        fz_z = evaluate(ψ_fz, z)

        fx_exact = c * sum([coeffs[i] * ((k * z)^(i - 1)) for i in 1:(deg + 1)])
        @test fz_z ≈ fx_exact atol = 1e-4
      end
    end
  end

  @testset "test multi-dimensional elementary function construction" begin
    #Constant function but represented in three dimension
    @testset "test const" begin
      g = named_grid((3, 3))
      s = complex_continuous_siteinds(g; map_dimension=3)

      c = 1.5

      ψ_fzyz = const_itn(s; c)

      z1, z2, z3 = 0.5 + 0.125 * im, 0.25 + 0.875 * im, 0.0

      fz_zyz = evaluate(ψ_fzyz, [z1, z2, z3], [1, 2, 3])
      @test fz_zyz ≈ c
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
    real_dimension_vertices = [
      [(i, 1) for i in 1:Int(L / 2)], [(i, 1) for i in ((Int(L / 2)) + 1):L]
    ]
    imag_dimension_vertices = [
      [(i, 1) for i in ((Int(L / 2)) + 1):L], [(i, 1) for i in 1:Int(L / 2)]
    ]
    s = complex_continuous_siteinds(g, real_dimension_vertices, imag_dimension_vertices)
    z1, z2 = 0.625 + 0.875 * im, 0.25 + 0.125 * im

    for (name, net_func, func) in funcs
      @testset "test $name" begin
        a = rand() + im * rand()
        k = rand() + im * rand()
        c = rand() + im * rand()

        ψ_fz1 = net_func(s; k, a, c, dim=1)
        ψ_fz2 = net_func(s; k, a, c, dim=2)

        ψ_fz = ψ_fz1 + ψ_fz2
        fz_z = evaluate(ψ_fz, [z1, z2], [1, 2])
        @test c * func(k * z1 + a) + c * func(k * z2 + a) ≈ fz_z
      end
    end

    #Sum of two tanhs in two different directions
    @testset "test tanh" begin
      L = 3
      g = named_grid((L, 2))
      a = rand() + im * rand()
      k = rand() + im * rand()
      c = rand() + im * rand()
      nterms = 50
      s = complex_continuous_siteinds(g; map_dimension=2)

      z1, z2 = 0.5 + 0.125 * im, 0.625 + 0.25 * im
      ψ_fz1 = tanh_itn(s; k, a, c, nterms, dim=1)
      ψ_fz2 = tanh_itn(s; k, a, c, nterms, dim=2)

      ψ_fz = ψ_fz1 + ψ_fz2
      fz_z = evaluate(ψ_fz, [z1, z2], [1, 2]; alg="exact")
      @test c * tanh(k * z1 + a) + c * tanh(k * z2 + a) ≈ fz_z
    end
  end

  @testset "test delta_p" begin
    L = 10
    g = named_grid((L, 1))
    s = complex_continuous_siteinds(g; map_dimension=2)
    z10, z20 = 0.625 + 0.25 * im, 0.25 * im
    delta = 2.0^(-1.0 * L)
    lastDigit = 1 - delta
    zs = [0.0, delta, 0.25 + 0.5 * im, 0.5, 0.625 * im, 0.875, lastDigit + lastDigit * im]
    @testset "test single point" begin
      ψ = delta_p(s, [z10, z20])
      @test evaluate(ψ, [z10, z20], [1, 2]) ≈ 1
      # test another point
      @test evaluate(ψ, [z20, z10], [1, 2]) ≈ 0
    end
    @testset "test plane" begin
      ψ = delta_p(s, [z20], [2])

      # should be 1 in the plane
      for z in zs
        @test evaluate(ψ, [z, z20], [1, 2]) ≈ 1
      end
      # test random points
      for z in zs
        @test evaluate(ψ, [z, 0.5], [1, 2]) ≈ 0
      end
    end
    @testset "test sums of points" begin
      points = [[z10, z20], [z20, z10]]
      ψ = delta_p(s, points)
      @test evaluate(ψ, [z10, z20], [1, 2]) ≈ 1
      @test evaluate(ψ, [z20, z10], [1, 2]) ≈ 1
      # test other points
      @test evaluate(ψ, [0, 0], [1, 2]) ≈ 0
      @test evaluate(ψ, [0, z20], [1, 2]) ≈ 0
    end

    @testset "test sums of points and plane" begin
      p0 = 0.5 + 0.5 * im
      points = [[z10, z20], [p0]]
      dims = [[1, 2], [2]]
      ψ = delta_p(s, points, dims)
      @test evaluate(ψ, [z10, z20], [1, 2]) ≈ 1
      for z in zs
        @test evaluate(ψ, [z, p0], [1, 2]) ≈ 1
      end
      ## test other points
      @test evaluate(ψ, [0, 0], [1, 2]) ≈ 0
      @test evaluate(ψ, [0, z20], [1, 2]) ≈ 0
    end
  end
end
