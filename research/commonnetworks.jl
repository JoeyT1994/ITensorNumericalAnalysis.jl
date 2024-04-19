using ITensorNumericalAnalysis
using NamedGraphs: named_grid
using NamedGraphs: NamedGraph, NamedEdge, add_vertex!, add_edge!

function qtt_siteinds(L::Int64; order="standard")
  g = named_grid((L, 1))
  dimension_vertices = [[(i, 1) for i in 1:L]]
  s = continuous_siteinds(g, dimension_vertices)
  return s
end

function star_siteinds(no_points::Int64, length::Int64)
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
  s = continuous_siteinds(g, [[(i, 1) for i in 1:(x - 1)]])
  return s
end
