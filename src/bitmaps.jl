using Dictionaries: Dictionary, set!
using Graphs: Graphs

struct BitMap{VB,VD}
  vertex_digit::VB
  vertex_dimension::VD
  base::Int64
end

default_base() = 2

vertex_digit(bm::BitMap) = bm.vertex_digit
vertex_dimension(bm::BitMap) = bm.vertex_dimension
base(bm::BitMap) = bm.base

default_bit_map(vertices::Vector) = Dictionary(vertices, [i for i in 1:length(vertices)])
function default_dimension_map(vertices::Vector)
  return Dictionary(vertices, [1 for i in 1:length(vertices)])
end

function BitMap(g; base::Int64=default_base())
  return BitMap(default_bit_map(vertices(g)), default_dimension_map(vertices(g)), base)
end
function BitMap(vertex_digit, vertex_dimension; base::Int64=default_base())
  return BitMap(vertex_digit, vertex_dimension, base)
end
function BitMap(dimension_vertices::Vector{Vector{V}}; base::Int64=default_base()) where {V}
  vertex_digit = Dictionary()
  vertex_dimension = Dictionary()
  for (dimension, vertices) in enumerate(dimension_vertices)
    for (bit, v) in enumerate(vertices)
      set!(vertex_digit, v, bit)
      set!(vertex_dimension, v, dimension)
    end
  end
  return BitMap(vertex_digit, vertex_dimension, base)
end

function Base.copy(bm::BitMap)
  return BitMap(copy(vertex_digit(bm)), copy(vertex_dimension(bm)), copy(base(bm)))
end

dimension(bm::BitMap) = maximum(collect(values(vertex_dimension(bm))))
dimension(bm::BitMap, vertex) = vertex_dimension(bm)[vertex]
digit(bm::BitMap, vertex) = vertex_digit(bm)[vertex]
bit_value_to_scalar(bm::BitMap, vertex, value::Int64) = value / (base(bm)^digit(bm, vertex))

function Graphs.vertices(bm::BitMap)
  @assert keys(vertex_dimension(bm)) == keys(vertex_digit(bm))
  return collect(keys(vertex_dimension(bm)))
end
function Graphs.vertices(bm::BitMap, dimension::Int64)
  return collect(
    filter(v -> vertex_dimension(bm)[v] == dimension, keys(vertex_dimension(bm)))
  )
end
function vertex(bm::BitMap, dimension::Int64, bit::Int64)
  return only(
    filter(
      v -> vertex_dimension(bm)[v] == dimension && vertex_digit(bm)[v] == bit,
      keys(vertex_dimension(bm)),
    ),
  )
end

function calculate_xyz(bm::BitMap, vertex_to_bit_value_map, dimensions::Vector{Int64})
  out = Float64[]
  for dimension in dimensions
    vs = vertices(bm, dimension)
    push!(out, sum([bit_value_to_scalar(bm, v, vertex_to_bit_value_map[v]) for v in vs]))
  end
  return out
end

function calculate_xyz(bm::BitMap, vertex_to_bit_value_map)
  return calculate_xyz(bm, vertex_to_bit_value_map, [i for i in 1:dimension(bm)])
end
function calculate_x(bm::BitMap, vertex_to_bit_value_map, dimension::Int64)
  return only(calculate_xyz(bm, vertex_to_bit_value_map, [dimension]))
end
function calculate_x(bm::BitMap, vertex_to_bit_value_map)
  return calculate_x(bm, vertex_to_bit_value_map, 1)
end

function calculate_bit_values(
  bm::BitMap, xs::Vector{Float64}, dimensions::Vector{Int64}; print_x=false
)
  @assert length(xs) == length(dimensions)
  vertex_to_bit_value_map = Dictionary()
  for (i, x) in enumerate(xs)
    dimension = dimensions[i]
    x_rn = copy(x)
    vs = vertices(bm, dimension)
    sorted_vertices = sort(vs; by=vs -> digit(bm, vs))
    for v in sorted_vertices
      i = base(bm) - 1
      vertex_set = false
      while (!vertex_set)
        if x_rn >= bit_value_to_scalar(bm, v, i)
          set!(vertex_to_bit_value_map, v, i)
          x_rn -= bit_value_to_scalar(bm, v, i)
          vertex_set = true
        else
          i = i - 1
        end
      end
    end

    if print_x
      x_bitstring = calculate_x(bm, vertex_to_bit_value_map, dimension)
      println(
        "Dimension $dimension. Actual value of x is $x but bitstring rep. is $x_bitstring"
      )
    end
  end
  return vertex_to_bit_value_map
end

function calculate_bit_values(bm::BitMap, x::Float64, dimension::Int64; kwargs...)
  return calculate_bit_values(bm, [x], [dimension]; kwargs...)
end
function calculate_bit_values(bm::BitMap, xs::Vector{Float64}; kwargs...)
  return calculate_bit_values(bm, xs, [i for i in 1:length(xs)]; kwargs...)
end
function calculate_bit_values(bm::BitMap, x::Float64; kwargs...)
  return calculate_bit_values(bm, [x], [1]; kwargs...)
end
