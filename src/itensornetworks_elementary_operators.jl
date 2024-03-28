function plus_shift_TTN(s::IndsNetwork, bit_map; dimension=default_dimension())
  @assert is_tree(s)
  ttn_op = OpSum()
  dim_vertices = vertices(bit_map, dimension)
  L = length(dim_vertices)

  string_site = [("S+", vertex(bit_map, dimension, L))]
  add!(ttn_op, 1.0, "S+", vertex(bit_map, dimension, L))
  for i in L:-1:2
    pop!(string_site)
    push!(string_site, ("S-", vertex(bit_map, dimension, i)))
    push!(string_site, ("S+", vertex(bit_map, dimension, i - 1)))
    add!(ttn_op, 1.0, (string_site...)...)
  end

  return TTN(ttn_op, s; algorithm="svd")
end

function minus_shift_TTN(s::IndsNetwork, bit_map; dimension=default_dimension())
  @assert is_tree(s)
  ttn_op = OpSum()
  dim_vertices = vertices(bit_map, dimension)
  L = length(dim_vertices)

  string_site = [("S-", vertex(bit_map, dimension, L))]
  add!(ttn_op, 1.0, "S-", vertex(bit_map, dimension, L))
  for i in L:-1:2
    pop!(string_site)
    push!(string_site, ("S+", vertex(bit_map, dimension, i)))
    push!(string_site, ("S-", vertex(bit_map, dimension, i - 1)))
    add!(ttn_op, 1.0, (string_site...)...)
  end

  return TTN(ttn_op, s; algorithm="svd")
end

function no_shift_TTN(s::IndsNetwork)
  ttn_op = OpSum()
  string_site_full = [("I", v) for v in vertices(s)]
  add!(ttn_op, 1.0, (string_site_full...)...)
  return TTN(ttn_op, s; algorithm="svd")
end

function stencil(
  s::IndsNetwork,
  bit_map,
  shifts::Vector{Float64},
  delta_power::Int64;
  dimension=default_dimension(),
  truncate_kwargs...,
)
  @assert length(shifts) == 3
  plus_shift = first(shifts) * plus_shift_TTN(s, bit_map; dimension)
  minus_shift = last(shifts) * minus_shift_TTN(s, bit_map; dimension)
  no_shift = shifts[2] * no_shift_TTN(s)

  stencil_op = plus_shift + minus_shift + no_shift
  stencil_op = truncate(stencil_op; truncate_kwargs...)

  for v in vertices(stencil_op)
    stencil_op[v] = (2^delta_power) * stencil_op[v]
  end

  return truncate(stencil_op; truncate_kwargs...)
end

function Laplacian_operator(s::IndsNetwork, bit_map; kwargs...)
  return stencil(s, bit_map, [1.0, -2.0, 1.0], 2; kwargs...)
end

function derivative_operator(s::IndsNetwork, bit_map; kwargs...)
  return 0.5 * stencil(s, bit_map, [1.0, 0.0, -1.0], 1; kwargs...)
end
