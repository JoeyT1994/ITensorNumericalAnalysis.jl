using Graphs: is_tree
using NamedGraphs.GraphsExtensions: undirected_graph
using ITensors:
  OpSum,
  SiteType,
  noprime,
  op,
  Op,
  Ops,
  truncate,
  replaceinds,
  delta,
  prime,
  sim,
  noprime!,
  contract,
  replaceinds
using ITensorMPS: add!
using ITensorNetworks:
  IndsNetwork,
  ITensorNetwork,
  TreeTensorNetwork,
  combine_linkinds,
  ttn,
  union_all_inds,
  map_inds,
  siteinds
default_boundary() = "Dirichlet"

## TODO: turn this into a proper system ala sites which can be externally overloaded

function boundary_term(s::IndsNetworkMap, boundary::String, dim, isfwd::Bool, n::Int=0)
  ttn_op = OpSum()
  dim_vertices = dimension_vertices(s, dim)
  L = length(dim_vertices)

  if boundary == "Neumann"
    string_site = [
      if j <= (L - n)
        (isfwd ? "Dup" : "Ddn", vertex(s, dim, j))
      else
        ("I", vertex(s, dim, j))
      end for j in 1:L
    ]
    add!(ttn_op, 1.0, (string_site...)...)
  elseif boundary == "Periodic"
    string_site = [
      if j <= (L - n)
        (isfwd ? "D-" : "D+", vertex(s, dim, j))
      else
        ("I", vertex(s, dim, j))
      end for j in 1:L
    ]
    add!(ttn_op, 1.0, (string_site...)...)
  end
  return ttn_op
end

function forward_shift_opsum(
  s::IndsNetworkMap; dim=default_dim(), boundary=default_boundary(), n::Int=0
)
  @assert is_tree(s)
  @assert base(s) == 2
  ttn_op = OpSum()
  dim_vertices = dimension_vertices(s, dim)
  L = length(dim_vertices)

  string_site = [("D+", vertex(s, dim, L - n))]
  add!(ttn_op, 1.0, "D+", vertex(s, dim, L - n))
  for i in (L - n):-1:2
    pop!(string_site)
    push!(string_site, ("D-", vertex(s, dim, i)))
    push!(string_site, ("D+", vertex(s, dim, i - 1)))
    add!(ttn_op, 1.0, (string_site...)...)
  end

  ttn_op += boundary_term(s, boundary, dim, true, n)

  return ttn_op
end

function backward_shift_opsum(
  s::IndsNetworkMap; dim=default_dim(), boundary=default_boundary(), n::Int=0
)
  @assert is_tree(s)
  @assert base(s) == 2
  ttn_op = OpSum()
  dim_vertices = dimension_vertices(s, dim)
  L = length(dim_vertices)

  string_site = [("D-", vertex(s, dim, L - n))]
  add!(ttn_op, 1.0, "D-", vertex(s, dim, L - n))
  for i in (L - n):-1:2
    pop!(string_site)
    push!(string_site, ("D+", vertex(s, dim, i)))
    push!(string_site, ("D-", vertex(s, dim, i - 1)))
    add!(ttn_op, 1.0, (string_site...)...)
  end

  ttn_op += boundary_term(s, boundary, dim, false, n)

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
  return ttn(ttn_opsum, indsnetwork(s); truncate_kwargs...)
end

function forward_shift_op(s::IndsNetworkMap; truncate_kwargs=(;), kwargs...)
  ttn_opsum = forward_shift_opsum(s; kwargs...)
  return ttn(ttn_opsum, indsnetwork(s); truncate_kwargs...)
end

function stencil(
  s::IndsNetworkMap,
  shifts::Vector,
  delta_power::Int;
  dim=default_dim(),
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
      stencil_opsum += shifts[i] * forward_shift_opsum(s; dim, boundary=right_boundary, n)
    end
  end

  for i in [4, 5]
    n = i == 5 ? 1 : 0
    if !iszero(shifts[i])
      stencil_opsum += shifts[i] * backward_shift_opsum(s; dim, boundary=left_boundary, n)
    end
  end

  stencil_op = ttn(stencil_opsum, indsnetwork(s); kwargs...)

  if scale
    for v in dimension_vertices(s, dim)
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

function laplacian_operator(s::IndsNetworkMap; dims=[i for i in 1:dimension(s)], kwargs...)
  remaining_dims = copy(dims)
  ∇ = second_derivative_operator(s; dim=first(remaining_dims), kwargs...)
  popfirst!(remaining_dims)
  for rd in remaining_dims
    ∇ += second_derivative_operator(s; dim=rd, kwargs...)
  end
  return ∇
end
function laplacian_operator(s::IndsNetworkMap, boundary::String; kwargs...)
  return laplacian_operator(s; left_boundary=boundary, right_boundary=boundary, kwargs...)
end

function identity_operator(s::IndsNetworkMap; kwargs...)
  operator_inds = ITensorNetworks.union_all_inds(indsnetwork(s), prime(indsnetwork(s)))
  return ITensorNetwork(Op("I"), operator_inds; kwargs...)
end

" Create an operator bitstring corresponding to the number x"
function point_to_opsum(s::IndsNetworkMap, x::Number, dim::Int)
  ttn_op = OpSum()
  ind_to_ind_value_map = calculate_ind_values(s, x, dim)
  string_sites = [
    (ind_to_ind_value_map[only(s[v])] == 1) ? ("Dup", v) : ("Ddn", v) for
    v in dimension_vertices(s, dim)
  ]
  add!(ttn_op, 1.0, (string_sites...)...)
  return ttn_op
end

" Create an operator which maps a function to 0 at all points in xs"
function map_to_zero_operator(
  s::IndsNetworkMap, xs::Vector, dims::Vector=[1 for _ in xs]; truncate_kwargs...
)
  return operator_proj(
    delta_kernel(
      s,
      [[x] for x in xs],
      [[dim] for dim in dims];
      remove_overlap=true,
      coeff=-1,
      include_identity=true,
      truncate_kwargs...,
    ),
  )
end

function map_to_zero_operator(s::IndsNetworkMap, x::Number, dim::Int=1; truncate_kwargs...)
  return map_to_zero_operator(s, [x], [dim]; truncate_kwargs...)
end

" Map the points xs in dimension dims of the function f to 0"
function map_to_zeros(
  f::ITensorNetworkFunction,
  xs::Vector,
  dims::Vector=[1 for _ in xs];
  truncate_kwargs=(;), # for map_operator
  kwargs..., # for operate
)
  s = indsnetworkmap(f)
  zero_op = map_to_zero_operator(s, xs, dims; truncate_kwargs...)
  return operate(zero_op, f; kwargs...)
end

function map_to_zeros(f::ITensorNetworkFunction, x::Number, dim::Int=1; kwargs...)
  return map_to_zeros(f, [x], [dim]; kwargs...)
end

" Take |f> and create an operator |f><δ| "
function operator_proj(fx::ITensorNetworkFunction)
  fx = copy(fx)
  operator = itensornetwork(fx)
  s = siteinds(operator)
  for v in vertices(operator)
    sind = s[v]
    sindsim = sim(sind)
    operator[v] = replaceinds(operator[v], sind, sindsim)
    operator[v] = operator[v] * delta(vcat(sind, sindsim, sind'))
  end
  return operator
end

function multiply(g::ITensorNetworkFunction, f::ITensorNetworkFunction)
  imap = merge(indexmap(g), indexmap(f))
  g, f = sim(copy(itensornetwork(g)); sites=[]), copy(itensornetwork(f))
  verts = union(vertices(g), vertices(f))
  tensors = Pair{Any,ITensor}[]
  for v in verts
    if v ∉ vertices(g)
      push!(tensors, v => f[v])
    elseif v ∉ vertices(f)
      push!(tensors, v => g[v])
    else
      cinds = commoninds(g[v], f[v])
      fg_v = f[v] * replaceinds(g[v], cinds, cinds')
      fg_v *= prod([delta(cind, cind', cind'') for cind in cinds])
      push!(tensors, v => noprime(fg_v))
    end
  end
  fg = combine_linkinds(ITensorNetwork(tensors))
  s = siteinds(fg)
  inmap = IndsNetworkMap(s, imap)
  return ITensorNetworkFunction(fg, inmap)
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
