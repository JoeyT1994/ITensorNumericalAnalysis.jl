using Test
using ITensorNumericalAnalysis

using ITensors: siteinds
using ITensorNetworks: maxlinkdim
using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: NamedGraph, nv, vertices
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree
using ITensorNumericalAnalysis: itensornetwork, forward_shift_op, backward_shift_op
using Dictionaries: Dictionary
using Random: seed!
seed!(42)

@testset "test integration" begin
  @testset "integration of operator*function in 1D" begin
    L = 20
    g = named_comb_tree((2, L ÷ 2))
    s = continuous_siteinds(g; map_dimension=1)

    ψ_fx = exp_itn(s; dimension=1)
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
    ψ_fxy = exp_itn(s; dimension=1) * exp_itn(s; dimension=2)

    ans = ITensorNumericalAnalysis.integrate(ψ_fxy)
    correct = (exp(1) - 1)^2
    # The integral ∫₀¹ exp(x+y) dxdy
    @test ans ≈ correct atol = 1e-4
  end
end
