using Dictionaries: Dictionary, set!
using Graphs: AbstractEdge, AbstractGraph, bfs_tree, dst, edges, edgetype,
    inneighbors, neighbors, src, topological_sort_by_dfs, vertices, is_tree
using ITensors: ITensor, Index, dag, dim, inds, onehot, space, uniqueinds
using NamedGraphs: vertextype
using TensorNetworkQuantumSimulator: TensorNetwork, maxvirtualdim,
    setindex_preserve!, siteinds

"""
    tci(
        f::Function, 
        g::AbstractGraph,
        imap::IndsNetworkMap; 
        initialpivot = random_initialpivot(imap),
        pivotsearch = FullPivot(),
        nsweeps = 1,
        maxdim = typemax(Int),
        mindim = 1,
        cutoff = eps(),
        kwargs...
    )

Compute the tensor cross interpolation of a function `f` on a tensor network with 
indices given by `imap` and a graph structure defined by `graph`.
"""
function tci(
    f::Function,
    g::AbstractGraph,
    imap::AbstractIndexMap;
    initialpivot = random_initialpivot(imap),
    edge_sequence = euler_sequence(g; nsites = 2),
    use_caching = true,
    kwargs...
)
    is_tree(g) || throw(ArgumentError("`tci` is currently not supported on tensor networks with loops."))
    fmemo = MemoizedFunction(
        input -> f(calculate_p(imap, input)), 
        initialpivot; 
        use_caching
    )
    tnf = init_tensornetworkfunction(
        g, 
        imap,
        src(edge_sequence[1]); 
        initialpivot,
        c = fmemo(initialpivot)
    )
    tnf = tci!(fmemo, tnf; edge_sequence, kwargs...)
    return tnf
end
"""
    tci!(
        fmemo::MemoizedFunction,
        ψ::TensorNetworkFunction;
        pivotsearch = FullPivot(),
        nsweeps = 1,
        maxdim = typemax(Int),
        mindim = 1,
        cutoff = eps(),
        outputlevel = 0,
        kwargs...
    )

Sweep tci pivots into the tensor network `ψ` in place.
"""
function tci!(
    fmemo::MemoizedFunction,
    ψ::TensorNetworkFunction;
    pivotsearch::AbstractPivotSearch = FullPivot(),
    nsweeps = 1,
    maxdim = typemax(Int),
    mindim = 1,
    cutoff = eps(),
    outputlevel = 0,
    kwargs...
)
    @inbounds for sweep in 1:nsweeps
        ψ = addpivots!(
            fmemo, 
            ψ; 
            pivotsearch,
            maxdim,
            mindim,
            cutoff,
            kwargs...
        )
        if outputlevel >= 1
            println("After sweep $sweep :")
            println(" maxlinkdim= $(maxvirtualdim(ψ))")
        end
    end
    
    return ψ
end
function addpivots!(
    fmemo::MemoizedFunction,
    ψ::TensorNetworkFunction;
    edge_sequence,
    pivotsearch,
    kwargs...
)
    @inbounds for edge in edge_sequence
        ψ = addpivot!(fmemo, ψ, edge; pivotsearch, kwargs...)
    end
    return ψ
end
"""
Add tci pivot to a "Pi tensor" in the tensor network `ψ`
with the center edge given by `edge`.
"""
function addpivot!(
    fmemo::MemoizedFunction,
    ψ::TensorNetworkFunction,
    edge::AbstractEdge;
    pivotsearch::AbstractPivotSearch,
    maxdim::Int,
    mindim::Int,
    cutoff::Real
)   
    v1, v2 = src(edge), dst(edge)
    imap = indexmap(ψ)
    s1, s2 = siteinds(imap, v1), siteinds(imap, v2)
    site_inds = vcat(s1, s2)
    outer_inds = vcat(uniqueinds(ψ[v1], ψ[v2]), uniqueinds(ψ[v2], ψ[v1]))
    link_inds = setdiff(outer_inds, site_inds)

    # two site update tensor
    Pi = ITensor(eltype(fmemo), site_inds..., link_inds...)
    site_ranges = [1:dim(s) for s in site_inds]
    link_ranges = [1:dim(l) for l in link_inds]
    @inbounds for link_vals in Iterators.product(link_ranges...)
        link_pivs = vcat(
            [space(l)[lval] for (l, lval) in zip(link_inds, link_vals)]...
        )
        @inbounds for site_vals in Iterators.product(site_ranges...)
            site_pivs = [ind => s for (ind, s) in zip(site_inds, site_vals)]
            arg = vcat(site_pivs, link_pivs)
            val = fmemo(arg)
            Pi[site_vals..., link_vals...] = val
        end
    end
    # col_inds are the unique indices of v1 (Left block), used to split the tensor
    col_inds = uniqueinds(ψ[v1], ψ[v2])
    ltags = "Link"
    C, Z, _ = interpolative(
        Pi, col_inds;
        tags = ltags, 
        pivotsearch, 
        maxdim, 
        mindim, 
        cutoff
    )
    # Z carries `col_inds` (v1), C carries `row_inds` (v2)
    setindex_preserve!(ψ, Z, v1)
    setindex_preserve!(ψ, C, v2)
    return ψ
end

function random_initialpivot(s::AbstractIndexMap)
    return [ind => rand(1:dim(ind)) for ind in inds(s)]
end
function init_tensornetworkfunction(
    g::AbstractGraph, 
    imap::AbstractIndexMap,
    root::V;
    initialpivot, 
    c = 1.0
) where {V}
    vs = collect(vertices(g))
    sites = siteinds(imap)
    piv_map = Dict(initialpivot)
    tree_dir = bfs_tree(g, root)
    # collect descendants for every node
    desc = Dictionary{vertextype(g), Vector{vertextype(g)}}()
    @inbounds for v in vertices(tree_dir)
        set!(desc, v, [v])
    end
    # orders nodes such that children always appear before parents
    sorted_nodes = reverse(topological_sort_by_dfs(tree_dir))
    @inbounds for v in sorted_nodes
        @inbounds for p in inneighbors(tree_dir, v)
            append!(desc[p], desc[v])
        end
    end
    ls = Dict{edgetype(g), Index}()
    @inbounds for e in edges(tree_dir)
        u, v = src(e), dst(e)
        # match direction to original graph edge
        eorig = edgetype(g)(u, v)
        ind_args = [[only(sites[vert]) => piv_map[only(sites[vert])] for vert in desc[v]]]
        ls[eorig] = Index(ind_args; tags = "Link")
    end
    ls_rev = Dict(reverse(e) => dag(ls[e]) for e in keys(ls))
    l = merge(ls, ls_rev)
    tensors = Dictionary{vertextype(g), ITensor}()
    @inbounds for v in vs
        sind = only(sites[v])
        bit = piv_map[sind]
        linds = [l[edgetype(g)(v, vn)] for vn in neighbors(g, v)]
        T = onehot(sind => bit, [link => 1 for link in linds]...)
        set!(tensors, v, T)
    end
    tensors[root] *= c
    tn = TensorNetwork(tensors, g)
    return TensorNetworkFunction(tn, imap)
end