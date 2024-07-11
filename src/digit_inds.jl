using Graphs: AbstractGraph
using ITensors: Index
using ITensorNetworks: IndsNetwork, vertex_data, vertex_tag

function default_dimension_vertices(g::AbstractGraph; map_dimension::Int64=1)
  vs = collect(vertices(g))
  L = length(vs)
  return [[v for v in vs[i:map_dimension:L]] for i in 1:map_dimension]
end

# reuse Qudit definitions for now
function ITensors.val(::ValName{N}, ::SiteType"Digit") where {N}
  return parse(Int, String(N)) + 1
end

function ITensors.state(::StateName{N}, ::SiteType"Digit", s::Index) where {N}
  n = parse(Int, String(N))
  st = zeros(dim(s))
  st[n + 1] = 1.0
  return ITensor(st, s)
end

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

function digit_tag(vertex, dim::Int, digit::Int)
  return "Digit, V$(vertex_tag(vertex)), Dim$(dim), Dig$(digit)"
end

function imag_digit_tag(vertex, dim::Int, digit::Int)
  return "Digit, Imag, V$(vertex_tag(vertex)), Dim$(dim), Dig$(digit)"
end

function real_digit_tag(vertex, dim::Int, digit::Int)
  return "Digit, Real, V$(vertex_tag(vertex)), Dim$(dim), Dig$(digit)"
end

function digit_siteinds(
  g::AbstractGraph,
  dimension_vertices::Vector{Vector{V}}=default_dimension_vertices(g);
  base=2,
) where {V}
  is = IndsNetwork(g; site_space=Dictionary(vertices(g), [Index[] for v in vertices(g)]))
  for (dim, verts) in enumerate(dimension_vertices)
    for (digit, v) in enumerate(verts)
      is[v] = vcat(is[v], Index(base, digit_tag(v, dim, digit)))
    end
  end

  return is
end

function complex_digit_siteinds(
  g::AbstractGraph,
  real_dimension_vertices::Vector{Vector{V}}=default_dimension_vertices(g),
  imag_dimension_vertices::Vector{Vector{V}}=default_dimension_vertices(g);
  base=2,
) where {V}
  is = IndsNetwork(g; site_space=Dictionary(vertices(g), [Index[] for v in vertices(g)]))
  for (dim, verts) in enumerate(real_dimension_vertices)
    for (digit, v) in enumerate(verts)
      is[v] = vcat(is[v], Index(base, real_digit_tag(v, dim, digit)))
    end
  end

  for (dim, verts) in enumerate(imag_dimension_vertices)
    for (digit, v) in enumerate(verts)
      is[v] = vcat(is[v], Index(base, imag_digit_tag(v, dim, digit)))
    end
  end

  return is
end
