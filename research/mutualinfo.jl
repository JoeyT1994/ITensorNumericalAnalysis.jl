
using LinearAlgebra: eigvals, tr
using Random
using NamedGraphs.GraphsExtensions: add_edge, a_star
using NamedGraphs: NamedGraph, NamedEdge

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


function mutual_info(f, nsamples, dimension1, digit1, dimension2, digit2, L, no_dims)
    rho = zeros((4,4))
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
                        rho[val1 + 1 + 2*val2, val1_p + 1+  2*val2_p] += fi*ficonj
                        rho[val1_p + 1 + 2*val2_p, val1 + 1 + 2*val2] += fi*ficonj
                    end
                end
            end
        end
    end
    rho /= nsamples
    rho /= tr(rho)

    rhoi = zeros((2,2))
    rhoi[1,1] = rho[1,1] + rho[3,3]
    rhoi[1,2] = rho[1,2] + rho[3,4]
    rhoi[2,1] = rho[2,1] + rho[4,3]
    rhoi[2,2] = rho[2,2] + rho[4,4]

    rhoj = zeros((2,2))
    rhoj[1,1] = rho[1,1] + rho[2,2]
    rhoj[1,2] = rho[1,3] + rho[2,4]
    rhoj[2,1] = rho[3,1] + rho[4,2]
    rhoj[2,2] = rho[3,3] + rho[4,4]

    eigs, eigsi, eigsj = eigvals(rho), eigvals(rhoi), eigvals(rhoj)
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

function generate_tree(mi_matrix, L, no_dims, max_z)
    g = NamedGraph([(i, j) for i in 1:no_dims for j in 1:L])
    mi_matrix_t = copy(mi_matrix)
    adj_mat = zeros((no_dims, L, no_dims, L))
    co_ords = zeros((no_dims, L))
    while !iszero(sum(mi_matrix_t)) && sum(co_ords) != max_z * L * no_dims 
        _, ind = findmax(mi_matrix_t)
        di, bi, dj, bj = ind[1], ind[2], ind[3], ind[4]
        zi, zj = co_ords[di, bi], co_ords[dj, bj]
        path = a_star(g, (di, bi), (dj, bj))
        if !iszero(mi_matrix_t[di, bi, dj, bj]) && iszero(adj_mat[di, bi, dj, bj]) && zi < max_z && zj < max_z && isempty(path)
            adj_mat[di, bi, dj, bj], adj_mat[dj, bj, di, bi] = 1, 1
            co_ords[di, bi] += 1
            co_ords[dj, bj] += 1
            g = add_edge(g, NamedEdge((di, bi) => (dj, bj)))
        end
        mi_matrix_t[di, bi, dj, bj] = 0
        mi_matrix_t[dj, bj, di, bi] = 0
    end

    println("Bought MI down from $(sum(mi_matrix)) to $(sum(mi_matrix - adj_mat .* mi_matrix))")
    return g
end