using ITensorNumericalAnalysis
using NamedGraphs.GraphsExtensions: disjoint_union, add_edges, eccentricity
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs: NamedGraph, NamedEdge, add_vertex!, add_edge!, vertextype

function qtt_siteinds_canonical(L::Int64; map_dimension, is_complex = false)
  g = named_grid((L, 1))
  dimension_vertices = Vector{vertextype(g)}[]
  for d in 1:map_dimension
    vertices = [(i, 1) for i in d:map_dimension:L]
    push!(dimension_vertices, vertices)
  end
  s = is_complex ? continuous_siteinds(g, dimension_vertices, dimension_vertices; is_complex) : continuous_siteinds(g, dimension_vertices)
  return s
end

function qtt_siteinds_canonical_sequentialdims(L::Int64; map_dimension, is_complex = false)
  g = named_grid((L, 1))
  dimension_vertices = Vector{vertextype(g)}[]
  dim_length = Int64(L/ map_dimension)
  for d in 1:map_dimension
    vertices = [(i, 1) for i in (1+(d-1)*dim_length):((d)*dim_length)]
    push!(dimension_vertices, vertices)
  end
  s = is_complex ? continuous_siteinds(g, dimension_vertices, dimension_vertices; is_complex) : continuous_siteinds(g, dimension_vertices)
  return s
end

function star(no_points::Int64, length::Int64)
  g = NamedGraph([(1, 1)])
  x = 2
  for i in 1:length
    for j in 1:no_points
      add_vertex!(g, (x, 1))
      if i == 1
        add_edge!(g, NamedEdge((x, 1) => (1, 1)))
      else
        add_edge!(g, NamedEdge((x, 1) => (x - no_points, 1)))
      end
      x += 1
    end
  end
  return g
end

function order_star_vertices(mi_matrix, L, z)
  row_sums = [sum(mi_matrix[i, :]) for i in 1:L]
  verts = reverse(sort([i for i in 1:L]; by = i -> row_sums[i]))
  g = star(z, Int((L - 1)/z))
  dimension_vertices = [sort([(i,1) for i in 1:L]; by = x -> verts[first(x)])]
  return g, dimension_vertices
end

function continuous_siteinds_ordered(g; map_dimension = 1, is_complex = false)
  sorted_vertices = sort(collect(vertices(g)); by = v -> eccentricity(g, v))
  L = length(sorted_vertices)
  dimension_vertices = Vector{vertextype(g)}[]
  for d in 1:map_dimension
    push!(dimension_vertices, sorted_vertices[d:map_dimension:L])
  end

  s = is_complex ? continuous_siteinds(g, dimension_vertices, dimension_vertices; is_complex) : continuous_siteinds(g, dimension_vertices)
  return s
end

function continuous_siteinds_ordered_by_names(g; is_complex = false)
  vs = collect(vertices(g))
  no_dims = maximum(first.(vs))
  L = maximum(last.(vs))
  dimension_vertices = [[(i, j) for j in 1:L] for i in 1:no_dims]
  return continuous_siteinds(g, dimension_vertices)
end

function qtt_siteinds_multidimstar_ordered(L, npoints; map_dimension = 1, is_complex = false)
  L = Int64(L/map_dimension)
  pointlength = Int64((L-1) / npoints)
  g_singlestar = star(npoints, pointlength)
  g = disjoint_union(([i => g_singlestar for i in 1:map_dimension]...))
  for d in 1:(map_dimension-1)
    g = add_edges(g, [NamedEdge(((1, 1), d) => ((1, 1), d+1))])
  end
  sorted_vertices = sort(vertices(g_singlestar); by = v -> eccentricity(g_singlestar, v))
  dimension_vertices = Vector{vertextype(g)}[]
  for d in 1:map_dimension
    push!(dimension_vertices, [(v, d) for v in sorted_vertices])
  end
  s = is_complex ? continuous_siteinds(g, dimension_vertices, dimension_vertices; is_complex) : continuous_siteinds(g, dimension_vertices)
  return s
end