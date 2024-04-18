using ITensorNumericalAnalysis
using NamedGraphs: named_grid

function qtt_siteinds(L::Int64; order = "standard")
    g = named_grid((L, 1))
    dimension_vertices = [[(i, 1) for i in 1:L]]
    s = continuous_siteinds(g, dimension_vertices)
    return s
end