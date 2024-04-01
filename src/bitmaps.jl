using Dictionaries: Dictionary, set!
using Graphs: Graphs

struct BitMap{VB,VD}
  vertex_bit::VB
  vertex_dimension::VD
end

vertex_bit(bm::BitMap) = bm.vertex_bit
vertex_dimension(bm::BitMap) = bm.vertex_dimension

default_bit_map(vertices::Vector) = Dictionary(vertices, [i for i in 1:length(vertices)])
function default_dimension_map(vertices::Vector)
  return Dictionary(vertices, [1 for i in 1:length(vertices)])
end

BitMap(g) = BitMap(default_bit_map(vertices(g)), default_dimension_map(vertices(g)))
function BitMap(dimension_vertices::Vector{Vector{V}}) where {V}
  vertex_bit = Dictionary()
  vertex_dimension = Dictionary()
  for (dimension, vertices) in enumerate(dimension_vertices)
    for (bit, v) in enumerate(vertices)
      set!(vertex_bit, v, bit)
      set!(vertex_dimension, v, dimension)
    end
  end
  return BitMap(vertex_bit, vertex_dimension)
end

Base.copy(bm::BitMap) = BitMap(copy(vertex_bit(bm)), copy(vertex_dimension(bm)))

dimension(bm::BitMap) = maximum(collect(values(vertex_dimension(bm))))
dimension(bm::BitMap, vertex) = vertex_dimension(bm)[vertex]
bit(bm::BitMap, vertex) = vertex_bit(bm)[vertex]

function Graphs.vertices(bm::BitMap)
  @assert keys(vertex_dimension(bm)) == keys(vertex_bit(bm))
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
      v -> vertex_dimension(bm)[v] == dimension && vertex_bit(bm)[v] == bit,
      keys(vertex_dimension(bm)),
    ),
  )
end

function calculate_xyz(bm::BitMap, vertex_to_bit_value_map, dimensions::Vector{Int64})
  out = Float64[]
  for dimension in dimensions
    vs = vertices(bm, dimension)
    push!(out, sum([vertex_to_bit_value_map[v] / (2^bit(bm, v)) for v in vs]))
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
    sorted_vertices = sort(vs; by=vs -> bit(bm, vs))
    for v in sorted_vertices
      if (x_rn >= 1.0 / (2^bit(bm, v)))
        set!(vertex_to_bit_value_map, v, 1)
        x_rn -= 1.0 / (2^bit(bm, v))
      else
        set!(vertex_to_bit_value_map, v, 0)
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
