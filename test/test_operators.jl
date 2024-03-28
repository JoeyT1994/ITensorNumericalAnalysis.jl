using Test
using TensorNetworkFunctionals
using ITensors
using ITensorNetworks
using Random
using Distributions
using Graphs

using TensorNetworkFunctionals: itensornetwork

@testset "test laplacian on MPS" begin
  g = named_grid((12, 1))
  L = nv(g)
  s = siteinds("S=1/2", g)
  bit_map = BitMap(g)

  ∇sq = Laplacian_operator(s, bit_map; cutoff=1e-10)
  @test maxlinkdim(∇sq) == 3

  ψ_fx = sin_itn(s, bit_map; k=Float64(pi))
  ∂2x_ψ_fx = operate(∇sq, ψ_fx; truncate_kwargs=(; cutoff=1e-12))

  xs = [0.00025, 0.09765625, 0.25, 0.625, 0.875]
  for x in xs
    ∂2x_ψ_fx_x = real(calculate_fx(∂2x_ψ_fx, x))
    @test ∂2x_ψ_fx_x ≈ -pi * pi * sin(pi * x) atol = 1e-3
  end
end

@testset "test derivative on tree" begin
  g = named_comb_tree((3, 4))
  L = nv(g)
  s = siteinds("S=1/2", g)
  bit_map = BitMap(g)

  ∂_∂x = derivative_operator(s, bit_map; cutoff=1e-10)

  ψ_fx = sin_itn(s, bit_map; k=Float64(pi))
  ∂x_ψ_fx = operate(∂_∂x, ψ_fx; truncate_kwargs=(; cutoff=1e-12))

  xs = [0.025, 0.1, 0.25, 0.625, 0.875]
  for x in xs
    ∂x_ψ_fx_x = real(calculate_fx(∂x_ψ_fx, x))
    @test ∂x_ψ_fx_x ≈ pi * cos(pi * x) atol = 1e-3
  end
end
