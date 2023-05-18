include("itensornetworksutils.jl")

"""Construct the product state representation of the function f(x) = const."""
function const_itensornetwork(s::IndsNetwork; c::Union{Float64,ComplexF64}=1.0)
  ψ = delta_network(s; link_space=1)
  for v in vertices(ψ)
    ψ[v] = ITensor([1.0, 1.0], inds(ψ[v]))
  end
  ψ[first(vertices(ψ))] *= c

  return ψ
end

"""Construct the representation of the function f(x) = kx"""
function x_itensornetwork(
  s::IndsNetwork, vertex_map::Dict; k::Union{Float64,ComplexF64}=1.0, cutoff=1e-15
)
  ψ = const_itensornetwork(s; c=0.0)
  for v in vertices(s)
    ψxi = const_itensornetwork(s)
    ψxi[v] = ITensor([0.0, 1.0 / 2^vertex_map[v]], inds(ψxi[v]))
    ψ = ψ + ψxi
  end
  ψ[first(vertices(ψ))] *= k

  return ψ
end

"""Construct the product state representation of the exp(kx) function for x ∈ [0,a] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function exp_itensornetwork(
  s::IndsNetwork,
  vertex_map::Dict;
  k::Union{Float64,ComplexF64}=Float64(1.0),
  a::Float64=1.0,
)
  ψ = delta_network(s; link_space=1)
  for v in vertices(ψ)
    ψ[v] = ITensor([1.0, exp(k * a / (2^vertex_map[v]))], inds(ψ[v]))
  end

  return ψ
end

"""Construct the bond dim 2 representation of the cosh(kx) function for x ∈ [0,a] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function cosh_itensornetwork(
  s::IndsNetwork,
  vertex_map::Dict;
  k::Union{Float64,ComplexF64}=Float64(1.0),
  a::Float64=1.0,
)
  ψ1 = exp_itensornetwork(s, vertex_map; a, k)
  ψ2 = exp_itensornetwork(s, vertex_map; a, k=-k)

  ψ1[first(vertices(ψ1))] *= 0.5
  ψ2[first(vertices(ψ1))] *= 0.5

  return add_itensornetworks(ψ1, ψ2)
end

"""Construct the bond dim 2 representation of the sinh(kx) function for x ∈ [0,a] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function sinh_itensornetwork(
  s::IndsNetwork,
  vertex_map::Dict;
  k::Union{Float64,ComplexF64}=Float64(1.0),
  a::Float64=1.0,
)
  ψ1 = exp_itensornetwork(s, vertex_map; a, k)
  ψ2 = exp_itensornetwork(s, vertex_map; a, k=-k)

  ψ1[first(vertices(ψ1))] *= 0.5
  ψ2[first(vertices(ψ1))] *= -0.5

  return add_itensornetworks(ψ1, ψ2)
end

"""Construct the bond dim n representation of the tanh(kx) function for x ∈ [0,a] as an ITensorNetwork, using an IndsNetwork which 
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
    ψt = exp_itensornetwork(s, vertex_map; a, k=-2 * k * n)
    ψt[first(vertices(ψt))] *= 2 * ((-1)^n)
    ψ = add_itensornetworks(ψ, ψt)
  end

  return ψ
end

"""Construct the bond dim 2 representation of the cos(kx) function for x ∈ [0,a] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function cos_itensornetwork(
  s::IndsNetwork, vertex_map::Dict; k::Float64=1.0, a::Float64=1.0
)
  ψ1 = exp_itensornetwork(s, vertex_map; a, k=k * im)
  ψ2 = exp_itensornetwork(s, vertex_map; a, k=-k * im)

  ψ1[first(vertices(ψ1))] *= 0.5
  ψ2[first(vertices(ψ1))] *= 0.5

  return add_itensornetworks(ψ1, ψ2)
end

"""Construct the bond dim 2 representation of the sin(kx) function for x ∈ [0,a] as an ITensorNetwork, using an IndsNetwork which 
defines the network geometry. Vertex map provides the ordering of the sites as bits"""
function sin_itensornetwork(
  s::IndsNetwork, vertex_map::Dict; k::Float64=1.0, a::Float64=1.0
)
  ψ1 = exp_itensornetwork(s, vertex_map; a, k=k * im)
  ψ2 = exp_itensornetwork(s, vertex_map; a, k=-k * im)

  ψ1[first(vertices(ψ1))] *= -0.5 * im
  ψ2[first(vertices(ψ1))] *= 0.5 * im

  return add_itensornetworks(ψ1, ψ2)
end

const const_itn = const_itensornetwork
const cosh_itn  = cosh_itensornetwork
const sinh_itn  = sinh_itensornetwork
const tanh_itn  = tanh_itensornetwork
const exp_itn   = exp_itensornetwork
const sin_itn   = sin_itensornetwork
const cos_itn   = cos_itensornetwork
const x_itn     = x_itensornetwork
