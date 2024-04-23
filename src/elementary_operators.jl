using Graphs: is_tree
using NamedGraphs: undirected_graph
using ITensors:
  OpSum,
  @OpName_str,
  @SiteType_str,
  SiteType,
  siteinds,
  noprime,
  op,
  truncate,
  replaceinds,
  delta,
  add!,
  prime,
  sim,
  noprime!,
  contract
using ITensorNetworks: IndsNetwork, ITensorNetwork, TreeTensorNetwork, combine_linkinds, ttn

default_boundary() = "Dirichlet"

function ITensors.op(::OpName"D+", ::SiteType"Digit", s::Index)
  d = dim(s)
  o = zeros(d, d)
  o[2, 1] = 1
  return ITensor(o, s, s')
end
function ITensors.op(::OpName"D-", ::SiteType"Digit", s::Index)
  d = dim(s)
  o = zeros(d, d)
  o[1, 2] = 1
  return ITensor(o, s, s')
end
function ITensors.op(::OpName"Ddn", ::SiteType"Digit", s::Index)
  d = dim(s)
  o = zeros(d, d)
  o[1, 1] = 1
  return ITensor(o, s, s')
end
function ITensors.op(::OpName"Dup", ::SiteType"Digit", s::Index)
  d = dim(s)
  o = zeros(d, d)
  o[2, 2] = 1
  return ITensor(o, s, s')
end

function forward_shift_opsum(
  s::IndsNetworkMap; dimension=default_dimension(), boundary=default_boundary(), n::Int=0
)
  @assert is_tree(s)
  @assert base(s) == 2
  ttn_op = OpSum()
  dim_vertices = dimension_vertices(s, dimension)
  L = length(dim_vertices)

  string_site = [("D+", vertex(s, dimension, L - n))]
  add!(ttn_op, 1.0, "D+", vertex(s, dimension, L - n))
  for i in (L - n):-1:2
    pop!(string_site)
    push!(string_site, ("D-", vertex(s, dimension, i)))
    push!(string_site, ("D+", vertex(s, dimension, i - 1)))
    add!(ttn_op, 1.0, (string_site...)...)
  end

  if boundary == "Neumann"
    string_site = [
      if j <= (L - n)
        ("Dup", vertex(s, dimension, j))
      else
        ("I", vertex(s, dimension, j))
      end for j in 1:L
    ]
    add!(ttn_op, 1.0, (string_site...)...)
  elseif boundary == "Periodic"
    string_site = [
      if j <= (L - n)
        ("D-", vertex(s, dimension, j))
      else
        ("I", vertex(s, dimension, j))
      end for j in 1:L
    ]
    add!(ttn_op, 1.0, (string_site...)...)
  end

  return ttn_op
end

function backward_shift_opsum(
  s::IndsNetworkMap; dimension=default_dimension(), boundary=default_boundary(), n::Int=0
)
  @assert is_tree(s)
  @assert base(s) == 2
  ttn_op = OpSum()
  dim_vertices = dimension_vertices(s, dimension)
  L = length(dim_vertices)

  string_site = [("D-", vertex(s, dimension, L - n))]
  add!(ttn_op, 1.0, "D-", vertex(s, dimension, L - n))
  for i in (L - n):-1:2
    pop!(string_site)
    push!(string_site, ("D+", vertex(s, dimension, i)))
    push!(string_site, ("D-", vertex(s, dimension, i - 1)))
    add!(ttn_op, 1.0, (string_site...)...)
  end

  if boundary == "Neumann"
    string_site = [
      if j <= (L - n)
        ("Ddn", vertex(s, dimension, j))
      else
        ("I", vertex(s, dimension, j))
      end for j in 1:L
    ]
    add!(ttn_op, 1.0, (string_site...)...)
  elseif boundary == "Periodic"
    string_site = [
      if j <= (L - n)
        ("D+", vertex(s, dimension, j))
      else
        ("I", vertex(s, dimension, j))
      end for j in 1:L
    ]
    add!(ttn_op, 1.0, (string_site...)...)
  end

  return ttn_op
end

function no_shift_opsum(s::IndsNetworkMap)
  ttn_op = OpSum()
  string_site_full = [("I", v) for v in vertices(s)]
  add!(ttn_op, 1.0, (string_site_full...)...)
  return ttn_op
end

function backward_shift_op(s::IndsNetworkMap; truncate_kwargs=(;), kwargs...)
  ttn_opsum = backward_shift_opsum(s; kwargs...)
  return ttn(ttn_opsum, indsnetwork(s); algorithm="svd", truncate_kwargs...)
end

function forward_shift_op(s::IndsNetworkMap; truncate_kwargs=(;), kwargs...)
  ttn_opsum = forward_shift_opsum(s; kwargs...)
  return ttn(ttn_opsum, indsnetwork(s); algorithm="svd", truncate_kwargs...)
end

function stencil(
  s::IndsNetworkMap,
  shifts::Vector,
  delta_power::Int;
  dimension=default_dimension(),
  left_boundary=default_boundary(),
  right_boundary=default_boundary(),
  scale=true,
  truncate_op=true,
  kwargs...,
)
  # shifts = [ x+2Δh, x+Δh, x, x-Δh, x-2Δh]
  @assert length(shifts) == 5
  b = base(s)
  stencil_opsum = shifts[3] * no_shift_opsum(s)
  for i in [1, 2]
    n = i == 1 ? 1 : 0
    if !iszero(shifts[i])
      stencil_opsum +=
        shifts[i] * forward_shift_opsum(s; dimension, boundary=right_boundary, n)
    end
  end

  for i in [4, 5]
    n = i == 5 ? 1 : 0
    if !iszero(shifts[i])
      stencil_opsum +=
        shifts[i] * backward_shift_opsum(s; dimension, boundary=left_boundary, n)
    end
  end

  stencil_op = ttn(stencil_opsum, indsnetwork(s); algorithm="svd", kwargs...)

  if scale
    for v in dimension_vertices(s, dimension)
      stencil_op[v] = (b^delta_power) * stencil_op[v]
    end
  end

  return stencil_op
end

function first_derivative_operator(s::IndsNetworkMap; kwargs...)
  return stencil(s, [0.0, 0.5, 0.0, -0.5, 0.0], 1; kwargs...)
end

function second_derivative_operator(s::IndsNetworkMap; kwargs...)
  return stencil(s, [0.0, 1.0, -2.0, 1.0, 0.0], 2; kwargs...)
end

function third_derivative_operator(s::IndsNetworkMap; kwargs...)
  return stencil(s, [0.5, -1.0, 0.0, 1.0, -0.5], 3; kwargs...)
end

function fourth_derivative_operator(s::IndsNetworkMap; kwargs...)
  return stencil(s, [1.0, -4.0, 6.0, -4.0, 1.0], 4; kwargs...)
end

function laplacian_operator(
  s::IndsNetworkMap; dimensions=[i for i in 1:dimension(s)], kwargs...
)
  remaining_dims = copy(dimensions)
  ∇ = second_derivative_operator(s; dimension=first(remaining_dims), kwargs...)
  popfirst!(remaining_dims)
  for rd in remaining_dims
    ∇ += second_derivative_operator(s; dimension=rd, kwargs...)
  end
  return ∇
end

function identity_operator(s::IndsNetworkMap; kwargs...)
  return stencil(s, [0.0, 0.0, 1.0, 0.0, 0.0], 0; kwargs...)
end

function operator(fx::ITensorNetworkFunction)
  fx = copy(fx)
  operator = itensornetwork(fx)
  s = siteinds(operator)
  for v in vertices(operator)
    sind = s[v]
    sindsim = sim(sind)
    operator[v] = replaceinds!(operator[v], sind, sindsim)
    operator[v] = operator[v] * delta(vcat(sind, sindsim, sind'))
  end
  return operator
end

function multiply(gx::ITensorNetworkFunction, fx::ITensorNetworkFunction)
  @assert vertices(gx) == vertices(fx)
  fx, fxgx = copy(fx), copy(gx)
  s = siteinds(fxgx)
  for v in vertices(fxgx)
    ssim = sim(s[v])
    temp_tensor = replaceinds(fx[v], s[v], ssim)
    fxgx[v] = noprime!(fxgx[v] * delta(s[v], s[v]', ssim) * temp_tensor)
  end

  return combine_linkinds(fxgx)
end

function multiply(
  gx::ITensorNetworkFunction,
  fx::ITensorNetworkFunction,
  hx::ITensorNetworkFunction,
  fs::ITensorNetworkFunction...,
)
  return multiply(multiply(gx, fx), hx, fs...)
end

Base.:*(fs::ITensorNetworkFunction...) = multiply(fs...)

function operate(operator::TreeTensorNetwork, ψ::ITensorNetworkFunction; kwargs...)
  ψ_tn = ttn(itensornetwork(ψ))
  ψO_tn = noprime(contract(operator, ψ_tn; init=prime(copy(ψ_tn)), kwargs...))

  return ITensorNetworkFunction(ITensorNetwork(ψO_tn), indsnetworkmap(ψ))
end

function operate(operator::ITensorNetwork, ψ::ITensorNetworkFunction; kwargs...)
  return operate(ttn(operator), ψ; kwargs...)
end

function operate(
  operators::Vector{TreeTensorNetwork{V}}, ψ::ITensorNetworkFunction; kwargs...
) where {V}
  ψ = copy(ψ)
  for op in operators
    ψ = operate(op, ψ; kwargs...)
  end
  return ψ
end

function operate(
  operators::Vector{ITensorNetwork{V}}, ψ::ITensorNetworkFunction; kwargs...
) where {V}
  return operate(ttn.(operators), ψ; kwargs...)
end
