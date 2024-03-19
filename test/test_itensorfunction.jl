using Test

include("../src/itensornetworksutils.jl")

@testset "test constructor from ITensorNetwork" begin
  L = 10

  g = named_grid((L, 1))
  s = siteinds("S=1/2", g)

  ψ = randomITensorNetwork(s; link_space=2)

  fψ = ITensorNetworkFunction(ψ)

  @test vertices(fψ, 1) == vertices(fψ)
  @test dimension(fψ) == 1

  dimension_vertices = collect(values(group(v -> first(v) < Int64(0.5 * L), vertices(ψ))))
  fψ = ITensorNetworkFunction(ψ, dimension_vertices)
  @test union(Set(vertices(fψ, 1)), Set(vertices(fψ, 2))) == Set(vertices(fψ))
  @test dimension(fψ) == 2
end

@testset "test 1D elementary function construction" begin
  @testset "test const" begin
    L = 3
    g = named_grid((L, L))
    s = siteinds("S=1/2", g)
    c = 1.5

    bit_map = BitMap(g)
    ψ_fx = const_itn(s, bit_map; c)

    x = 0.5
    vertex_to_bit_value_map = calculate_bit_values(ψ_fx, x)

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
    @testset "test $name" begin
      Lx, Ly = 2, 3
      g = named_comb_tree((2, 3))
      a = 1.2
      k = 0.125
      s = siteinds("S=1/2", g)

      bit_map = BitMap(g)

      x = 0.625
      ψ_fx = net_func(s, bit_map; k, a)
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
    s = siteinds("S=1/2", g)

    bit_map = BitMap(g)

    x = 0.625
    ψ_fx = tanh_itn(s, bit_map; k, a, nterms)
    fx_x = calculate_fx(ψ_fx, x)

    @test tanh(k * x + a) ≈ fx_x
  end

  @testset "test poly" begin
    L = 6
    degrees = [i + 1 for i in 0:10]

    ###Generate a series of random polynomials on random graphs. Evaluate them at random x values"""
    for deg in degrees
      Random.seed!(1234 * deg)
      g = NamedGraph(Graphs.SimpleGraph(uniform_tree(L)))
      g = rename_vertices(g, Dict(zip(vertices(g), [(v, 1) for v in vertices(g)])))
      s = siteinds("S=1/2", g)

      coeffs = [rand(Uniform(-2, 2)) for i in 1:(deg + 1)]

      bit_map = BitMap(g)
      x = 0.875
      ψ_fx = poly_itn(s, bit_map, coeffs)
      fx_x = calculate_fx(ψ_fx, x)

      fx_exact = sum([coeffs[i] * (x^(deg + 1 - i)) for i in 1:(deg + 1)])
      @test fx_x ≈ fx_exact atol = 1e-4
    end
  end
end