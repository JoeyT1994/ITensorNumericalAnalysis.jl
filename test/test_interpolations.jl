using Test
using ITensorNumericalAnalysis
using NamedGraphs.NamedGraphGenerators: named_comb_tree

@testset "test interpolations" begin
  @testset "interpolate 1D function" begin
    L = 10
    g = named_comb_tree((1, L))
    s = continuous_siteinds(g)
    f1(x) = 10x^4 - 5x^3 - 20x^2 + 3x + 10
    ψ_c = function_itn(s, f1; mode="chebyshev", chop_level=1e-5)
    ψ_c = truncate(ψ_c; cutoff=1e-16)
    x = 0.375
    a = f1(x)
    b = real(ITensorNumericalAnalysis.evaluate(ψ_c, [x]))
    ε = 0.0001 #numerical precision limit
    @test abs(a - b) <= ε #test that they evaluate to close to the same point
  end

  @testset "interpolate 2D function" begin
    L = 12
    #comb tree
    g = named_comb_tree((2, L ÷ 2))
    s = continuous_siteinds(g; map_dimension=2)
    f2(x, y) = exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.05)
    ψ_c = function_itn(s, f2; mode="chebyshev", chop_level=1e-5)
    ψ_c = truncate(ψ_c; cutoff=1e-16)
    x, y = 0.375, 0.125
    a = f2(x, y)
    b = real(ITensorNumericalAnalysis.evaluate(ψ_c, [x, y]))
    ε = 0.0001 #numerical precision limit
    @test abs(a - b) <= ε #test that they evaluate to close to the same point
  end

  @testset "interpolate 1D data" begin
    L = 10
    g = named_comb_tree((1, L))
    s = continuous_siteinds(g)
    x_vals = collect(range(0.0, 1.0; length=200))
    f3(x) = 10 * x^4 - 5 * x^3 - 20 * x^2 + 3 * x + 10
    data = [f3(x) for x in x_vals]
    ψ_c = data_itn(s, data, x_vals; mode="chebyshev", chop_level=1e-5)
    ψ_c = truncate(ψ_c; cutoff=1e-16)
    x = 0.375
    a = f3(x)
    b = real(ITensorNumericalAnalysis.evaluate(ψ_c, [x]))
    ε = 0.0001 #numerical precision limit
    @test abs(a - b) <= ε #test that they evaluate to close to the same point
  end

  @testset "interpolate 2D data" begin
    L = 12
    #comb tree
    g = named_comb_tree((2, L ÷ 2))
    s = continuous_siteinds(g; map_dimension=2)
    f4(x, y) = exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.05)
    x_vals, y_vals = collect(range(0.0, 1.0; length=200)),
    collect(range(0.0, 1.0; length=200))
    data = [f4(x, y) for x in x_vals, y in y_vals]
    ψ_c = data_itn(s, data, (x_vals, y_vals); mode="chebyshev", chop_level=1e-5)
    ψ_c = truncate(ψ_c; cutoff=1e-16)
    x, y = 0.375, 0.125
    a = f4(x, y)
    b = real(ITensorNumericalAnalysis.evaluate(ψ_c, [x, y]))
    ε = 0.0001 #numerical precision limit
    @test abs(a - b) <= ε #test that they evaluate to close to the same point
  end
end
