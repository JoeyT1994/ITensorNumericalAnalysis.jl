using LinearAlgebra: eigvals, tr
using Random
using Graphs: merge_vertices
using NamedGraphs.GraphsExtensions: add_edge, a_star, dst, src, add_vertex, add_edge, neighbors, rem_edge
using NamedGraphs: NamedGraph, NamedEdge
using ITensors: Index, ITensor, combiner, array
using LinearAlgebra: Array

include("commonfunctions.jl")

function binary_to_position(binary_digits, L, no_dims)
    return [sum([binary_digits[d, i] / (2^i) for i in 1:L]) for d in 1:no_dims]
end

function random_binary(L, no_dims)
    binary_digits = zeros(Int, (no_dims, L))
    for d in 1:no_dims
        for j in 1:L
            binary_digits[d, j] = rand(0:1)
        end
    end
    return binary_digits
end

function entanglement(f, nsamples, dimension, digit, L, no_dims)
    rho = zeros((2,2))
    for i in 1:nsamples
        b1 = random_binary(L, no_dims)
        b2 = copy(b1)
        for val in 0:1
            for valp in 0:1
                b1[dimension, digit] = val
                b2[dimension, digit] = valp
                p1, p2 = binary_to_position(b1, L, no_dims), binary_to_position(b2, L, no_dims)
                fi, ficonj = f(p1), conj(f(p2))
                rho[val + 1, valp + 1] += fi*ficonj
                rho[valp + 1, val + 1] += fi*ficonj
            end
        end
    end
    rho /= tr(rho)
    eigs = eigvals(rho)
    s = 0
    for eig in eigs
        s -= eig > 1e-18 ? eig *log(eig) : 0
    end
    return s
end

function mutual_info(f, nsamples, dimension1, digit1, dimension2, digit2, L, no_dims)
    rho = zeros((2,2,2,2))
    for i in 1:nsamples
        b1 = random_binary(L, no_dims)
        b2 = copy(b1)
        for val1 in 0:1
            for val1_p in 0:1
                for val2 in 0:1
                    for val2_p in 0:1
                        b1[dimension1, digit1] = val1
                        b1[dimension2, digit2] = val2
                        b2[dimension1, digit1] = val1_p
                        b2[dimension2, digit2] = val2_p
                        p1, p2 = binary_to_position(b1, L, no_dims), binary_to_position(b2, L, no_dims)
                        fi, ficonj = f(p1), conj(f(p2))
                        rho[val1 + 1, val2 + 1, val1_p + 1, val2_p + 1] += fi*ficonj
                        rho[val1_p + 1, val2_p + 1, val1 + 1, val2 + 1] += fi*ficonj
                    end
                end
            end
        end
    end
    

    rhoi = zeros((2,2))
    for val1 in 1:2
        for val1_p in 1:2
            rhoi[val1, val1_p] = rho[val1, 1, val1_p, 1] + rho[val1, 2, val1_p, 2]
        end
    end
    rhoi /= tr(rhoi)

    rhoj = zeros((2,2))
    for val2 in 1:2
        for val2_p in 1:2
            rhoj[val2, val2_p] = rho[1, val2, 1, val2_p] + rho[2, val2, 2, val2_p]
        end
    end
    rhoj /= tr(rhoj)

    s1, s2 = Index(2, "s1"), Index(2, "s2")
    rho_itensor = ITensor(rho, s1, s2, s1', s2')
    rho_itensor = rho_itensor * combiner(s1, s2) * combiner(s1', s2')
    rho_itensor = array(rho_itensor)
    rho_itensor /= tr(rho_itensor)

    eigs, eigsi, eigsj = eigvals(rho_itensor), eigvals(rhoi), eigvals(rhoj)
    s, si, sj = 0, 0, 0
    for eig in eigs
        s -=  eig > 1e-18 ? eig *log(eig) : 0
    end
    for eigi in eigsi
        si -=  eigi > 1e-18 ? eigi *log(eigi) : 0
    end
    for eigj in eigsj
        sj -=  eigj > 1e-18 ? eigj *log(eigj) : 0
    end
    si + sj - s >= 0 && return si + sj - s
    return 0
end



function generate_mi_matrix(f, nsamples, L, no_dims)
    I = zeros((no_dims, L, no_dims, L))
    for di in 1:no_dims
        for bi in 1:L
            for dj in 1:no_dims
                for bj in 1:L
                    if (di != dj || bi != bj)
                        i = mutual_info(f, nsamples, di, bi, dj, bj, L, no_dims)
                        I[di, bi, dj, bj] += i
                        I[dj, bj, di, bi] += i
                    end
                end
            end
        end
    end

    return I
end

function generate_entanglements(f, nsamples, L, no_dims)
    e = zeros((no_dims, L))
    for d in 1:no_dims
        for b in 1:L
            e[d, b] = entanglement(f, nsamples, d,b, L, no_dims)
        end
    end
    return e
end

function cost_function(g, mi_matrix; alpha = 1, nn_only = false)
    c = 0
    verts = collect(vertices(g))
    for (i, v) in enumerate(verts)
        for vp in verts[(i+1):length(verts)]
            d = length(a_star(g, v, vp))
            if !nn_only
                c += (d^alpha) * mi_matrix[last(v), first(v), last(vp), first(vp)]
            else
                c += d == 1 ? mi_matrix[last(v), first(v), last(vp), first(vp)] : 0.0
            end
        end
    end
    return c
end

function cost_function_V2(g, mi_matrix; alpha)
    c = 0
    for v in vertices(g)
        c += eccentricity(g, v) * sum(mi_matrix[last(v), first(v), :, :])
    end
    return c
end

function cost_function_entanglement(g, entanglements; alpha)
    c = 0
    for v in vertices(g)
        c += eccentricity(g, v) * entanglements[last(v), first(v)]
    end
    return c
end

function minimize_me(mi_matrix; max_z, dim = 1, cost_function_kwargs...)
    verts = [(j,i) for i in 1:size(mi_matrix)[1] for j in 1:size(mi_matrix)[2]]
    verts_sort = reverse(sort(verts; by = v -> sum(mi_matrix[last(v), first(v), :, :])))

    v_init = first(verts_sort)
    g = NamedGraph([v_init])

    for v in verts_sort[2:length(verts_sort)]
        cs_gs = []
        g = add_vertex(g, v)
        for vn in filter(x -> degree(g, x) < max_z, setdiff(vertices(g), [v]))
            #Decide where to add v
            g_t = add_edge(g, NamedEdge(v => vn))
            push!(cs_gs, g_t => cost_function_V2(g_t, mi_matrix; cost_function_kwargs...))
        end
        cs_gs_sort = sort(cs_gs; by = c -> last(c))
        g = copy(first(first(cs_gs_sort)))
    end

    c_f = cost_function_V2(g, mi_matrix; cost_function_kwargs...)
    println("Final cost function is $c_f")
    return g
end

function minimize_entanglement(entanglements; max_z, dim = 1, cost_function_kwargs...)
    verts = [(j,i) for i in 1:size(entanglements)[1] for j in 1:size(entanglements)[2]]
    verts_sort = reverse(sort(verts; by = v -> entanglements[last(v), first(v)]))

    v_init = first(verts_sort)
    g = NamedGraph([v_init])

    for v in verts_sort[2:length(verts_sort)]
        cs_gs = []
        g = add_vertex(g, v)
        for vn in filter(x -> degree(g, x) < max_z, setdiff(vertices(g), [v]))
            #Decide where to add v
            g_t = add_edge(g, NamedEdge(v => vn))
            push!(cs_gs, g_t => cost_function_entanglement(g_t, entanglements; cost_function_kwargs...))
        end
        cs_gs_sort = sort(cs_gs; by = c -> last(c))
        g = copy(first(first(cs_gs_sort)))
    end

    c_f = cost_function_entanglement(g,entanglements; cost_function_kwargs...)
    println("Final cost function is $c_f")
    return g
end