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
using TensorNetworkQuantumSimulator: map_virtualinds!, combine_virtualinds!
using ITensorNetworks: ITensorNetworks, underlying_graph
default_boundary() = "Dirichlet"

## TODO: turn this into a proper system ala sites which can be externally overloaded

function ITensorNetworks.IndsNetwork(g::NamedGraph, site_space::Dictionary)
  s = ITensorNetworks.IndsNetwork(g)
  for v in vertices(g)
    s[v] = site_space[v]
  end
  return s
end

function TensorNetworkQuantumSimulator.TensorNetwork(ttn::ITensorNetworks.TTN)
  g = underlying_graph(ttn)
  t_dict=  Dictionary(collect(vertices(g)), [ttn[v] for v in vertices(g)])
  return TensorNetwork(t_dict, g)
end

function boundary_term(g::NamedGraph,
  s::AbstractIndexMap, boundary::String, dim, isfwd::Bool, n::Int=0)
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
  g::NamedGraph,
  s::AbstractIndexMap; dim=default_dim(), boundary=default_boundary(), n::Int=0
)
  @assert is_tree(g)
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

  ttn_op += boundary_term(g, s, boundary, dim, true, n)

  return ttn_op
end

function backward_shift_opsum(
  g::NamedGraph,
  s::AbstractIndexMap; dim=default_dim(), boundary=default_boundary(), n::Int=0
)
  @assert is_tree(g)
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

  ttn_op += boundary_term(g, s, boundary, dim, false, n)

  return ttn_op
end

function no_shift_opsum(g::NamedGraph, s::AbstractIndexMap)
  ttn_op = OpSum()
  string_site_full = [("I", v) for v in vertices(g)]
  add!(ttn_op, 1.0, (string_site_full...)...)
  return ttn_op
end

function backward_shift_op(g::NamedGraph, s::AbstractIndexMap; truncate_kwargs=(;), kwargs...)
  ttn_opsum = backward_shift_opsum(g, s; kwargs...)
  sinds_network = ITensorNetworks.IndsNetwork(g, siteinds(s))
  t = ITensorNetworks.ttn(ttn_opsum, sinds_network; truncate_kwargs...)
  return TensorNetwork(t)
end

function forward_shift_op(g::NamedGraph, s::AbstractIndexMap; truncate_kwargs=(;), kwargs...)
  ttn_opsum = forward_shift_opsum(g, s; kwargs...)
  sinds_network = ITensorNetworks.IndsNetwork(g, siteinds(s))
  t = ITensorNetworks.ttn(ttn_opsum, sinds_network; truncate_kwargs...)
  return TensorNetwork(t)
end

function stencil(
  g::NamedGraph,
  s::AbstractIndexMap,
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
  stencil_opsum = shifts[3] * no_shift_opsum(g, s)
  for i in [1, 2]
    n = i == 1 ? 1 : 0
    if !iszero(shifts[i])
      stencil_opsum += shifts[i] * forward_shift_opsum(g, s; dim, boundary=right_boundary, n)
    end
  end

  for i in [4, 5]
    n = i == 5 ? 1 : 0
    if !iszero(shifts[i])
      stencil_opsum += shifts[i] * backward_shift_opsum(g, s; dim, boundary=left_boundary, n)
    end
  end
  sinds_network = ITensorNetworks.IndsNetwork(g, siteinds(s))

  stencil_op = ITensorNetworks.ttn(stencil_opsum, sinds_network; kwargs...)

  if scale
    for v in dimension_vertices(s, dim)
      stencil_op[v] = (b^delta_power) * stencil_op[v]
    end
  end

  return TensorNetwork(stencil_op)
end

function first_derivative_operator(g::NamedGraph, s::AbstractIndexMap; kwargs...)
  return stencil(g, s, [0.0, 0.5, 0.0, -0.5, 0.0], 1; kwargs...)
end

function second_derivative_operator(g::NamedGraph, s::AbstractIndexMap; kwargs...)
  return stencil(g, s, [0.0, 1.0, -2.0, 1.0, 0.0], 2; kwargs...)
end

function third_derivative_operator(g::NamedGraph, s::AbstractIndexMap; kwargs...)
  return stencil(g, s, [0.5, -1.0, 0.0, 1.0, -0.5], 3; kwargs...)
end

function fourth_derivative_operator(g::NamedGraph, s::AbstractIndexMap; kwargs...)
  return stencil(g, s, [1.0, -4.0, 6.0, -4.0, 1.0], 4; kwargs...)
end

function laplacian_operator(g::NamedGraph, s::AbstractIndexMap; dims=[i for i in 1:dimension(s)], kwargs...)
  remaining_dims = copy(dims)
  ∇ = second_derivative_operator(g, s; dim=first(remaining_dims), kwargs...)
  popfirst!(remaining_dims)
  for rd in remaining_dims
    ∇ += second_derivative_operator(g, s; dim=rd, kwargs...)
  end
  return ∇
end
function laplacian_operator(g::NamedGraph, s::AbstractIndexMap, boundary::String; kwargs...)
  return laplacian_operator(g, s; left_boundary=boundary, right_boundary=boundary, kwargs...)
end

function identity_operator(g::NamedGraph, s::AbstractIndexMap)
  sinds = siteinds(s)
  ts = Dictionary(collect(vertices(g)), [ITensors.op("I", only(sinds[v])) for v in vertices(g)])
  return TensorNetwork(ts, g)
end

" Create an operator which maps a function to 0 at all points in xs"
function map_to_zero_operator(g::NamedGraph,
  s::AbstractIndexMap, xs::Vector, dims::Vector=[1 for _ in xs]; truncate_kwargs...
)
  return operator_proj(
    delta_kernel(
      g, s,
      [[x] for x in xs],
      [[dim] for dim in dims];
      remove_overlap=true,
      coeff=-1,
      include_identity=true,
      truncate_kwargs...,
    ),
  )
end

function map_to_zero_operator(g::NamedGraph, s::AbstractIndexMap, x::Number, dim::Int=1; truncate_kwargs...)
  return map_to_zero_operator(g, s, [x], [dim]; truncate_kwargs...)
end

" Map the points xs in dimension dims of the function f to 0"
function map_to_zeros(
  tnf::TensorNetworkFunction,
  xs::Vector,
  dims::Vector=[1 for _ in xs];
  truncate_kwargs=(;), # for map_operator
  kwargs..., # for operate
)
  s = indexmap(tnf)
  g = graph(tnf)
  zero_op = map_to_zero_operator(g, s, xs, dims; truncate_kwargs...)
  return operate(zero_op, tnf; kwargs...)
end

function map_to_zeros(f::TensorNetworkFunction, x::Number, dim::Int=1; kwargs...)
  return map_to_zeros(f, [x], [dim]; kwargs...)
end

" Take |f> and create an operator |f><δ| "
function operator_proj(fx::TensorNetworkFunction)
  operator = copy(fx)
  map_virtualinds!(sim, operator)
  s = siteinds(operator)
  for v in vertices(operator)
    sind = s[v]
    sindsim = sim(sind)
    ov = replaceinds(operator[v], sind, sindsim)
    setindex_preserve!(operator, ov * delta(vcat(sind, sindsim, sind')), v)
  end
  return tensornetwork(operator)
end

function multiply(g::TensorNetworkFunction, f::TensorNetworkFunction)
  imap = merge(indexmap(g), indexmap(f))
  g, f = copy(tensornetwork(g)), copy(tensornetwork(f))
  g = map_virtualinds!(sim, g)
  verts = union(vertices(g), vertices(f))
  tensors = Dictionary{vertextype(g), ITensor}()
  for v in verts
    if v ∉ vertices(g)
      set!(tensors, v, f[v])
    elseif v ∉ vertices(f)
      set!(tensors, v, g[v])
    else
      cinds = commoninds(g[v], f[v])
      fg_v = f[v] * replaceinds(g[v], cinds, cinds')
      fg_v *= prod([delta(cind, cind', cind'') for cind in cinds])
      set!(tensors, v, noprime(fg_v))
    end
  end
  fg = TensorNetwork(tensors)
  fg = combine_virtualinds!(fg)
  return TensorNetworkFunction(fg, imap)
end

function multiply(
  gx::TensorNetworkFunction,
  fx::TensorNetworkFunction,
  hx::TensorNetworkFunction,
  fs::TensorNetworkFunction...,
)
  return multiply(multiply(gx, fx), hx, fs...)
end

Base.:*(fs::TensorNetworkFunction...) = multiply(fs...)

function operate(
  operators::Vector{<:AbstractTensorNetwork}, ψ::TensorNetworkFunction; kwargs...
)
  ψ = copy(ψ)
  for op in operators
    ψ = operate(op, ψ; kwargs...)
  end
  return ψ
end

function operate(operator::AbstractTensorNetwork, ψ::TensorNetworkFunction; maxdim = maxvirtualdim(operator)*maxvirtualdim(ψ), cutoff = nothing)
  ψ = copy(ψ)
  for v in vertices(ψ)
    ψv = noprime(ψ[v] * operator[v])
    setindex_preserve!(ψ, ψv, v)
  end
  combine_virtualinds!(ψ)
  return ψ
  #TODO: Enable truncation
  #return truncate(ψ; alg = "bp", maxdim, cutoff)
end
