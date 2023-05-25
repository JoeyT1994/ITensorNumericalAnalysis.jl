using ITensors
using ITensorNetworks
using NamedGraphs
using EllipsisNotation
using Graphs
using LinearAlgebra

using ITensorNetworks: delta_network
using NamedGraphs: add_edges

include("../src/QTT_utils.jl")

function coeff(n, r, c)
  @assert r ≥ c
  # make this more efficient later
  multi = factorial(n) / (factorial(r - c) * factorial(c - 1) * factorial(n + 1 - r))

  return multi * 1 / sqrt(binomial(n, n + 1 - r)) * 1 / sqrt(binomial(n, c - 1))
end

function main()
  L = 10
  g = named_grid((L, 1))
  s = siteinds("S=1/2", g)

  # polynomial degree, i.e. f(x) = ∑ᵢⁿ cᵢ xⁱ 
  coeffs = [-0.021, 0.31, -1.1, 1]
  # coeffs will build f(x) = 1 -x + 0.5x^2
  n = length(coeffs) - 1

  #Define a map which determines a canonical ordering of the vertices of the network
  vertex_map = Dict(vertices(g) .=> [i for i in 1:L])

  ψ = delta_network(s; link_space=n + 1)
  for v in vertices(ψ)
    ψ[v] = ITensor(Float64, inds(ψ[v]))
    xi = 1.0 / 2^vertex_map[v]
    if (vertex_map[v] in [1]) # left edge
      # left boundary has onehot(left_link => 1)
      # so identity on 1, 1/2^i on 2
      vec = zeros((2, n + 1))
      vec[1, 1] = 1
      for i in 1:(n + 1)
        vec[2, i] = xi^(i - 1)
      end
      ψ[v][:, :] = vec
    elseif vertex_map[v] in [L] # right edge
      # right boundary multiplies in the coefficients
      fake = Index(n + 1, "fake")
      temp_right = ITensor(Float64, inds(ψ[v]), fake)
      mat = Matrix{Float64}(I, n + 1, n + 1)
      temp_right[1, :, :] = mat
      for β in 2:(n + 1)
        for α in 1:(β - 1)
          c = binomial(β - 1, α - 1)
          mat[α, β] = c * xi^(β - α)
        end
      end
      temp_right[2, :, :] = mat
      ψ[v] = temp_right * ITensor(coeffs, fake)
    else # bulk
      mat = Matrix{Float64}(I, n + 1, n + 1)
      ψ[v][1, :, :] = mat
      for β in 2:(n + 1)
        for α in 1:(β - 1)
          c = binomial(β - 1, α - 1)
          mat[α, β] = c * xi^(β - α)
        end
      end
      ψ[v][2, :, :] = mat
    end
  end

  valid_points = [i / 2^L for i in 0:(2^L - 1)]
  values = []
  for point in valid_points
    xis = calculate_xis(point, vertex_map; print_x=false)
    eval_point = calculate_x(xis, vertex_map)
    @assert eval_point ≈ point
    ψ12proj = get_bitstring_network(ψ, s, xis)
    push!(values, ITensors.contract(ψ12proj)[])
  end

  ans = 0 * valid_points
  for (i, c) in enumerate(coeffs)
    ans .+= c * (valid_points) .^ (i - 1)
  end
  @show ans ≈ values
end

main()
