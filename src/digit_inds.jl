using Graphs: AbstractGraph
using ITensors: Index
using ITensorNetworks: IndsNetwork, vertex_data, vertex_tag

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
