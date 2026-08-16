using Graphs: vertices, edges, AbstractGraph, src, dst
using NamedGraphs: vertextype, edgetype
using NamedGraphs.GraphsExtensions: default_root_vertex

# taken from ITensorNetworks.jl: https://github.com/ITensor/ITensorNetworks.jl/tree/main/src/solvers/region_plans
function euler_sequence(
    graph::AbstractGraph; nsites::Int, root_vertex::V = default_root_vertex(graph)
) where {V}
    if nsites == 1
        return euler_tour_vertices(graph, root_vertex)
    elseif nsites == 2
        return euler_tour_edges(graph, root_vertex)
    else
        error("nsites must be 1 or 2")
    end
end
function compute_adjacencies(G::AbstractGraph)
    adj = Dict(v => Vector{vertextype(G)}() for v in vertices(G))
    for e in edges(G)
        push!(adj[src(e)], dst(e))
        push!(adj[dst(e)], src(e))
    end
    return adj
end
function euler_tour_edges(G::AbstractGraph, start_vertex::V) where {V}
    adj = compute_adjacencies(G)
    etype = edgetype(G)
    vtype = vertextype(G)
    visited = Set{Tuple{vtype, vtype}}()
    tour = Vector{etype}()
    stack = Vector{vtype}()
    push!(stack, start_vertex)
    while !isempty(stack)
        u = stack[end]
        pushed = false
        for v in adj[u]
            if (u, v) ∉ visited
                push!(visited, (u, v))
                push!(visited, (v, u))
                push!(tour, etype(u => v))
                push!(stack, v)
                pushed = true
                break  # handle one neighbor at a time
            end
        end
        if !pushed
            pop!(stack)
            if !isempty(stack)
                v = stack[end]
                push!(tour, etype(u => v))  # Backtracking step
            end
        end
    end
    return tour
end
function euler_tour_vertices(G::AbstractGraph, start_vertex::V) where {V}
    edges = euler_tour_edges(G, start_vertex)
    isempty(edges) && return eltype(vertices(G))[]
    return [src(edges[1]), dst.(edges)...]
end