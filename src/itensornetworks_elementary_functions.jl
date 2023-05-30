include("itensornetworksutils.jl")

"""Construct the product state representation of the function f(x) = const."""
function const_itensornetwork(s::IndsNetwork; c::Union{Float64,ComplexF64}=1.0)
  ψ = delta_network(s; link_space=1)
  for v in vertices(ψ)
    ψ[v] = ITensor([1.0, 1.0], inds(ψ[v]))
  end
  ψ[first(vertices(ψ))] *= c

  ψ = is_tree(underlying_graph(s)) ? TTN(ψ) : ψ

  return ψ
end

"""Construct the representation of the function f(x) = kx"""
function x_itensornetwork(
  s::IndsNetwork, vertex_map::Dict; k::Union{Float64,ComplexF64}=1.0, cutoff=1e-16
)
  ψ = const_itensornetwork(s; c=0.0)
  g = underlying_graph(s)

  for v in vertices(s)
    ψxi = const_itensornetwork(s)
    ψxi[v] = ITensor([0.0, 1.0 / 2^vertex_map[v]], inds(ψxi[v]))
    ψ = ψ + ψxi

    if isa(ψ, TreeTensorNetwork)
      ψ = truncate(ψ; cutoff)
    end
  end
  ψ[first(vertices(ψ))] *= k

  return ψ
end

"""Construct the representation of the function f(x) = kx*x"""
function xsq_itensornetwork(
  s::IndsNetwork, vertex_map::Dict; k::Union{Float64,ComplexF64}=1.0, cutoff=1e-16
)
  ψ = const_itensornetwork(s; c=0.0)

  for v in vertices(s)
    for vp in vertices(s)
      ψxi = const_itensornetwork(s)
      if v != vp
        ψxi[v] = ITensor([0.0, 1.0 / 2^vertex_map[v]], inds(ψxi[v]))
        ψxi[vp] = ITensor([0.0, 1.0 / 2^vertex_map[vp]], inds(ψxi[vp]))
      else
        ψxi[v] = ITensor([0.0, 1.0 / (2^(2 * vertex_map[v]))], inds(ψxi[v]))
      end

      ψ = ψ + ψxi

      if isa(ψ, TreeTensorNetwork)
        ψ = truncate(ψ; cutoff)
      end
    end
  end
  ψ[first(vertices(ψ))] *= k

  return ψ
end

"""Construct the representation of the polynomial f(x) = ∑ᵢⁿ cᵢ xⁿ """
function polynomial_itensornetwork(s::IndsNetwork, vertex_map::Dict, coeffs::Vector)
  # currently only working for MPS like tree
  tree = is_tree(underlying_graph(s))
  max_degree = Δ(underlying_graph(s))
  if !(tree) || (tree && max_degree != 2)
    error("Graph passed to polynomial_itensornetwork is not a MPS: $s")
  end
  n = length(coeffs) - 1
  L = length(vertex_map)
  ψ = delta_network(s; link_space=n + 1)
  for v in vertices(ψ)
    ψ[v] = ITensor(0.0, inds(ψ[v]))
    xi = 1.0 / 2^vertex_map[v]
    if (vertex_map[v] in [1]) # left edge
      # left boundary has onehot(left_link => 1)
      # so identity on 1, 1/2^i on 2
      ψ[v][1, 1] = 1
      ψ[v][2, :] = [xi^(i - 1) for i in 1:(n + 1)]
    elseif vertex_map[v] in [L] # right edge
      # right boundary multiplies in the coefficients
      dummy = Index(n + 1, "dummy")
      temp_right = ITensor(Float64, inds(ψ[v]), dummy)
      temp_right[1, :, :] = Matrix{Float64}(I, n + 1, n + 1)
      temp_right[2, :, :] = [
        (α ≤ β) ? binomial(β - 1, α - 1) * xi^(β - α) : 0 for α in 1:(n + 1), β in 1:(n + 1)
      ]
      ψ[v] = temp_right * ITensor(coeffs, dummy)
    else # bulk
      ψ[v][1, :, :] = Matrix{Float64}(I, n + 1, n + 1)
      ψ[v][2, :, :] = [
        (α ≤ β) ? binomial(β - 1, α - 1) * xi^(β - α) : 0 for α in 1:(n + 1), β in 1:(n + 1)
      ]
    end
  end
  return TTN(ψ)
end

"""Construct the product state representation of the exp(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function exp_itensornetwork(
  s::IndsNetwork,
  vertex_map::Dict;
  k::Union{Float64,ComplexF64}=Float64(1.0),
  a::Union{Float64,ComplexF64}=Float64(1.0),
)
  ψ = delta_network(s; link_space=1)
  for v in vertices(ψ)
    ψ[v] = ITensor([1.0, exp(k / (2^vertex_map[v]))], inds(ψ[v]))
  end

  ψ = is_tree(underlying_graph(s)) ? TTN(ψ) : ψ
  ψ[first(vertices(ψ))] *= exp(a)

  return ψ
end

"""Construct the bond dim 2 representation of the cosh(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function cosh_itensornetwork(
  s::IndsNetwork,
  vertex_map::Dict;
  k::Union{Float64,ComplexF64}=Float64(1.0),
  a::Float64=1.0,
)
  ψ1 = exp_itensornetwork(s, vertex_map; a, k)
  ψ2 = exp_itensornetwork(s, vertex_map; a=-a, k=-k)

  ψ1[first(vertices(ψ1))] *= 0.5
  ψ2[first(vertices(ψ1))] *= 0.5

  return ψ1 + ψ2
end

"""Construct the bond dim 2 representation of the sinh(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function sinh_itensornetwork(
  s::IndsNetwork,
  vertex_map::Dict;
  k::Union{Float64,ComplexF64}=Float64(1.0),
  a::Float64=1.0,
)
  ψ1 = exp_itensornetwork(s, vertex_map; a, k)
  ψ2 = exp_itensornetwork(s, vertex_map; a=-a, k=-k)

  ψ1[first(vertices(ψ1))] *= 0.5
  ψ2[first(vertices(ψ1))] *= -0.5

  return ψ1 + ψ2
end

"""Construct the bond dim n representation of the tanh(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function tanh_itensornetwork(
  s::IndsNetwork,
  vertex_map::Dict,
  nterms::Int64;
  k::Union{Float64,ComplexF64}=Float64(1.0),
  a::Float64=1.0,
)
  ψ = const_itensornetwork(s)
  for n in 1:nterms
    ψt = exp_itensornetwork(s, vertex_map; a=-2 * n * a, k=-2 * k * n)
    ψt[first(vertices(ψt))] *= 2 * ((-1)^n)
    ψ = ψ + ψt
  end

  return ψ
end

"""Construct the bond dim 2 representation of the cos(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function cos_itensornetwork(
  s::IndsNetwork, vertex_map::Dict; k::Float64=1.0, a::Float64=1.0
)
  ψ1 = exp_itensornetwork(s, vertex_map; a=a * im, k=k * im)
  ψ2 = exp_itensornetwork(s, vertex_map; a=-a * im, k=-k * im)

  ψ1[first(vertices(ψ1))] *= 0.5
  ψ2[first(vertices(ψ1))] *= 0.5

  return ψ1 + ψ2
end

"""Construct the bond dim 2 representation of the sin(kx+a) function for x ∈ [0,1] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function sin_itensornetwork(
  s::IndsNetwork, vertex_map::Dict; k::Float64=1.0, a::Float64=1.0
)
  ψ1 = exp_itensornetwork(s, vertex_map; a=a * im, k=k * im)
  ψ2 = exp_itensornetwork(s, vertex_map; a=-a * im, k=-k * im)

  ψ1[first(vertices(ψ1))] *= -0.5 * im
  ψ2[first(vertices(ψ1))] *= 0.5 * im

  return ψ1 + ψ2
end

const const_itn = const_itensornetwork
const cosh_itn = cosh_itensornetwork
const sinh_itn = sinh_itensornetwork
const tanh_itn = tanh_itensornetwork
const exp_itn = exp_itensornetwork
const sin_itn = sin_itensornetwork
const cos_itn = cos_itensornetwork
const poly_itn = polynomial_itensornetwork
const x_itn = x_itensornetwork
