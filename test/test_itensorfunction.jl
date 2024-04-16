using Test
using ITensorNumericalAnalysis

using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: NamedGraph, named_grid, vertices, named_comb_tree, rename_vertices
using ITensors: siteinds
using Dictionaries: Dictionary
using SplitApplyCombine: group
using Random: seed!
using Distributions: Uniform

@testset "test constructor from ITensorNetwork" begin
  L = 10

  g = named_grid((L, 1))
  s = continuous_siteinds(g)

  ψ = random_itensornetwork(s; link_space=2)

  fψ = ITensorNetworkFunction(ψ)

  @test dimension_vertices(fψ, 1) == vertices(fψ)
  @test dimension(fψ) == 1

  dim_vertices = collect(values(group(v -> first(v) < Int64(0.5 * L), vertices(ψ))))
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
    ind_to_ind_value_map = calculate_ind_values(ψ_fx, x)

    fx_x = calculate_fx(ψ_fx, x)
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
      a = 1.2
      k = 0.125

      s = continuous_siteinds(g)

      x = 0.625
      ψ_fx = net_func(s; k, a)
      fx_x = calculate_fx(ψ_fx, x)
      @test func(k * x + a) ≈ fx_x
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
      a = 1.2
      k = 0.125
      b = 3

      s = continuous_siteinds(g; base=3)

      x = (5.0 / 9.0)
      ψ_fx = net_func(s; k, a)
      fx_x = calculate_fx(ψ_fx, x)
      @test func(k * x + a) ≈ fx_x
    end
  end

  @testset "test tanh" begin
    L = 10
    g = named_grid((L, 1))
    a = 1.3
    k = 0.15
    nterms = 50

    s = continuous_siteinds(g)

    x = 0.625
    ψ_fx = tanh_itn(s; k, a, nterms)
    fx_x = calculate_fx(ψ_fx, x)

    @test tanh(k * x + a) ≈ fx_x
  end

  @testset "test poly" begin
    L = 6
    degrees = [i + 1 for i in 0:10]

    ###Generate a series of random polynomials on random graphs. Evaluate them at random x values"""
    for deg in degrees
      seed!(1234 * deg)
      g = NamedGraph(SimpleGraph(uniform_tree(L)))
      g = rename_vertices(g, Dict(zip(vertices(g), [(v, 1) for v in vertices(g)])))
      s = continuous_siteinds(g)

      coeffs = [rand(Uniform(-2, 2)) for i in 1:(deg + 1)]

      x = 0.875
      ψ_fx = poly_itn(s, coeffs)
      fx_x = calculate_fx(ψ_fx, x)

      fx_exact = sum([coeffs[i] * (x^(i - 1)) for i in 1:(deg + 1)])
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

    fx_xyz = calculate_fxyz(ψ_fxyz, [x, y, z], [1, 2, 3])
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
      a = 1.2
      k = 0.125

      ψ_fx = net_func(s; k, a, dimension=1)
      ψ_fy = net_func(s; k, a, dimension=2)

      ψ_fxy = ψ_fx + ψ_fy
      fxy_xy = calculate_fxyz(ψ_fxy, [x, y], [1, 2])
      @test func(k * x + a) + func(k * y + a) ≈ fxy_xy
    end
  end

  #Sum of two tanhs in two different directions
  @testset "test tanh" begin
    L = 3
    g = named_grid((L, 2))
    a = 1.3
    k = 0.15
    nterms = 10
    s = continuous_siteinds(g; map_dimension=2)

    x, y = 0.625, 0.875
    ψ_fx = tanh_itn(s; k, a, nterms, dimension=1)
    ψ_fy = tanh_itn(s; k, a, nterms, dimension=2)

    ψ_fxy = ψ_fx + ψ_fy
    fxy_xy = calculate_fxyz(ψ_fxy, [x, y], [1, 2])
    @test tanh(k * x + a) + tanh(k * y + a) ≈ fxy_xy
  end
end
