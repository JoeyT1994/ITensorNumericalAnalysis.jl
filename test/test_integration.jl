using Test
using ITensorNumericalAnalysis

using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: NamedGraph, nv, vertices
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree
using Dictionaries: Dictionary
using Random: seed!
seed!(42)

@testset "test integration" begin

    @testset "simple integration 1D" begin
        L = 30
        g = named_comb_tree((3, L ÷ 3))
        s = continuous_siteinds(g; map_dimension = 1)
        ψ_fx = exp_tnf(g, s)

        ans = integrate(ψ_fx)
        correct = (exp(1) - 1)
        # The integral ∫₀¹ exp(x+y) dxdy
        @test ans ≈ correct atol = 1.0e-4
    end

    @testset "integration of operator*function in 1D" begin
        L = 20
        g = named_comb_tree((2, L ÷ 2))
        s = continuous_siteinds(g; map_dimension = 1)

        ψ_fx = exp_tnf(g, s; dim = 1)
        O = operator_proj(ψ_fx)
        correct = 1 / 2 * (-1 + exp(1)^2)
        ans = integrate(O, ψ_fx)
        #The integral ∫₀¹ exp(x)*exp(x) dx
        @test ans ≈ correct atol = 1.0e-4
    end

    @testset "simple integration 2D" begin
        L = 30
        g = named_comb_tree((3, L ÷ 3))
        s = continuous_siteinds(g; map_dimension = 2)
        ψ_fxy = exp_tnf(g, s; dim = 1) * exp_tnf(g, s; dim = 2)

        ans = integrate(ψ_fxy)
        correct = (exp(1) - 1)^2
        # The integral ∫₀¹ exp(x+y) dxdy
        @test ans ≈ correct atol = 1.0e-4
    end

    @testset "partial integration 3D" begin
        L = 90
        g = named_comb_tree((3, L ÷ 3))
        s = continuous_siteinds(g, [[(i, j) for j in 1:(L ÷ 3)] for i in 1:3])
        ψ_fxyz = exp_tnf(g, s; dim = 1) + cos_tnf(g, s; dim = 2) + exp_tnf(g, s; dim = 3)

        ψ_fx = partial_integrate(ψ_fxyz, [2, 3])
        f_correct = x -> (exp(x) - 1) + sin(1) + exp(1)
        @test only(dimensions(ψ_fx)) == 1
        x = 0.875
        @test abs(evaluate(ψ_fx, x) - f_correct(x)) <= 1.0e-8
    end
end
