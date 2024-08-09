using Test
using ITensorNumericalAnalysis

using ITensors: siteinds
using ITensorNetworks: maxlinkdim
using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: NamedGraph, nv, vertices
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree
using ITensorNumericalAnalysis: reduced_indsnetworkmap
using Dictionaries: Dictionary
using Random: seed!
seed!(42)

@testset "test integration" begin
  @testset "integration of operator*function in 1D" begin
    L = 20
    g = named_comb_tree((2, L ÷ 2))
    s = continuous_siteinds(g; map_dimension=1)

    ψ_fx = exp_itn(s; dim=1)
    O = operator_proj(ψ_fx)
    correct = 1 / 2 * (-1 + exp(1)^2)
    ans = ITensorNumericalAnalysis.integrate(O, ψ_fx)
    #The integral ∫₀¹ exp(x)*exp(x) dx 
    @test ans ≈ correct atol = 1e-4
  end

  @testset "simple integration 2D" begin
    L = 30
    g = named_comb_tree((3, L ÷ 3))
    s = continuous_siteinds(g; map_dimension=2)
    ψ_fxy = exp_itn(s; dim=1) * exp_itn(s; dim=2)

    ans = ITensorNumericalAnalysis.integrate(ψ_fxy)
    correct = (exp(1) - 1)^2
    # The integral ∫₀¹ exp(x+y) dxdy
    @test ans ≈ correct atol = 1e-4
  end

  @testset "partial integration 3D" begin
    L = 90
    g = named_comb_tree((3, L ÷ 3))
    s = continuous_siteinds(g, [[(i, j) for j in 1:(L ÷ 3)] for i in 1:3])
    s1, s2, s3 = reduced_indsnetworkmap(s, 1),
    reduced_indsnetworkmap(s, 2),
    reduced_indsnetworkmap(s, 3)
    ψ_fxyz = exp_itn(s1; dim=1) * cos_itn(s2; dim=2) * exp_itn(s3; dim=3)

    ψ_fx = ITensorNumericalAnalysis.partial_integrate(ψ_fxyz, [2, 3])
    f_correct = x -> (exp(1) - 1) * sin(1) * exp(x)
    @test only(dimensions(ψ_fx)) == 1
    x = 0.875
    @test abs(evaluate(ψ_fx, x) - f_correct(x)) <= 1e-8
  end
end
