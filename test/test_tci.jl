using Test
using ITensorNumericalAnalysis

using ITensorNumericalAnalysis: tci, prrldu, interpolative,
    MemoizedFunction, cache, use_caching,
    AbstractPivotSearch, FullPivot, RookPivot, searchpivot, isexhaustive,
    pivotsearch_grid, haspivots, PivotIndex, random_initialpivot
using ITensors: ITensor, Index, dag, dim, hasqns, inds, norm, random_itensor, space
using LinearAlgebra: Diagonal, I, UnitLowerTriangular, UnitUpperTriangular
using Graphs: SimpleGraph, uniform_tree, eccentricity
using NamedGraphs: NamedGraph, nv, vertices
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree, named_binary_tree
using TensorNetworkQuantumSimulator: maxvirtualdim
using Dictionaries: Dictionary
using Random: seed!, MersenneTwister

seed!(42)

tol = 1e-10
const pivs = (
    FullPivot(), 
    RookPivot(; maxsteps = 2, rng = MersenneTwister(1))
)

L = 25
g_mps = named_grid((L, 1))
g_comb = named_comb_tree((2, L ÷ 2))
g_bt = named_binary_tree(Int(sqrt(L)))
# most significant digits nearest the tree center
bt_sorted = sort(collect(vertices(g_bt)); by = v -> eccentricity(g_bt, v), alg = MergeSort)
const graphs_1D = (
    (g = g_mps, s = continuous_siteinds(g_mps; map_dimension = 1)),
    (g = g_comb, s = continuous_siteinds(g_comb; map_dimension = 1)),
    (g = g_bt, s = continuous_siteinds(g_bt; map_dimension = 1)),
)
const graphs_2D = (
    (g = g_mps, s = continuous_siteinds(g_mps; map_dimension = 2)),
    (g = g_comb, s = continuous_siteinds(g_comb; map_dimension = 2)),
    (g = g_bt, s = continuous_siteinds(g_bt, [bt_sorted[d:2:end] for d in 1:2])),
)

@testset "test interpolative decomposition" begin
    rng = MersenneTwister(99)

    # ID on matrices
    # C*Z ≈ M
    for piv in pivs
        M = randn(rng, 9, 14)
        C, Z, piv_cols, err = interpolative(M; pivotsearch = piv, cutoff = 0.0)

        @test size(C, 2) == size(Z, 1) == length(piv_cols)
        @test C * Z ≈ M
        @test err <= tol
    end

    # C is made of columns of M
    M = randn(rng, Float64, 10, 4) * randn(rng, Float64, 4, 20)
    C, Z, piv_cols, _ = interpolative(M; cutoff = tol)

    @test length(piv_cols) == 4
    @test allunique(piv_cols)
    @test all(1 .<= piv_cols .<= size(M, 2))
    @test C ≈ M[:, piv_cols]
    # Z reproduces the pivot columns exactly: Z[:, piv_cols] == I
    @test Z[:, piv_cols] ≈ Matrix(I, length(piv_cols), length(piv_cols))
    @test C * Z ≈ M

    # ID on ITensors
    rng = MersenneTwister(11)
    s1, s2, s3 = Index(2, "Digit"), Index(2, "Digit"), Index(2, "Digit")
    T = random_itensor(s1, s2, s3)

    C, Z, err = interpolative(T, [s1, s2]; cutoff = tol)

    b = only(setdiff(inds(Z), [s1, s2]))
    @test haspivots(b)
    @test b in inds(C)
    @test issetequal(setdiff(inds(C), [b]), [s3])
    @test issetequal(setdiff(inds(Z), [b]), [s1, s2])
    @test dim(b) <= 2

    @test norm(C * Z - T) <= tol
    @test err <= tol

    # pivot bookkeeping
    C, Z, _ = interpolative(T, [s1, s2]; cutoff = tol)
    b = only(setdiff(inds(Z), [s1, s2]))
    for piv in space(b)
        @test length(piv) == 2
        @test issetequal(first.(piv), [s1, s2])
        @test all(1 <= last(p) <= dim(first(p)) for p in piv)
    end
end

@testset "test pivot search" begin
    rng = MersenneTwister(7)

    # FullPivot finds the global maximum
    M = randn(rng, 11, 13)
    val, idx = searchpivot(M, FullPivot())
    @test val == maximum(abs, M)
    @test abs(M[idx]) == val

    # RookPivot is a local maximum in its row and column
    for _ in 1:20
        M = randn(rng, 8, 10)
        val, idx = searchpivot(M, RookPivot(; maxsteps = 100, rng))
        i, j = Tuple(idx)
        @test val == abs(M[i, j])
        @test val == maximum(abs, @view M[i, :])
        @test val == maximum(abs, @view M[:, j])
        @test val <= maximum(abs, M)
    end

    # test that fixed rngs work for RookPivot
    M = randn(MersenneTwister(3), 10, 10)
    a = searchpivot(M, RookPivot(; maxsteps = 3, rng = MersenneTwister(5)))
    b = searchpivot(M, RookPivot(; maxsteps = 3, rng = MersenneTwister(5)))
    @test a == b
end

@testset "test PivotIndex" begin
    s1, s2 = Index(2, "Digit,x"), Index(2, "Digit,y")
    pivs = [[s1 => 1, s2 => 1], [s1 => 2, s2 => 1], [s1 => 2, s2 => 2]]
    b = Index(pivs; tags = "Link")

    @test b isa PivotIndex
    @test haspivots(b)
    @test !haspivots(s1)
    @test !haspivots(Index(4))
    @test dim(b) == 3
    @test dim(b) == length(space(b))
    @test !hasqns(b)
    @test space(b) == pivs
    @test space(b)[2] == [s1 => 2, s2 => 1]

    d = dag(b)
    @test d isa PivotIndex
    @test dim(d) == dim(b)
end


@testset "test tci" begin

    rng = MersenneTwister(12)
    xs = [rand(rng, 0:255)/256 for _ in 1:5]
    ys = [rand(rng, 0:255)/256 for _ in 1:5]
    @testset "exp(x)" begin
        for graph in graphs_1D, piv in pivs
            f = x -> exp(sum(x))
            ψ = tci(f, graph.g, graph.s; pivotsearch = piv, nsweeps = 3, cutoff = tol, maxdim = 10)

            @test nv(ψ) == nv(graph.g)
            for x in xs
                @test evaluate(ψ, x) ≈ f(x) atol = 1.0e-6
            end
            # exp(x) factorizes over digits on any tree
            @test maxvirtualdim(ψ) == 1
            @test integrate(ψ) ≈ exp(1) - 1 atol = 1e-4
        end
    end

    @testset "cos(x)" begin
        for graph in graphs_1D, piv in pivs
            f = x -> cos(sum(x))
            ψ = tci(f, graph.g, graph.s; pivotsearch = piv, nsweeps = 4, cutoff = tol, maxdim = 10)

            for x in xs
                @test evaluate(ψ, x) ≈ f(x) atol = 1.0e-6
            end
            @test maxvirtualdim(ψ) <= 2
            @test abs(integrate(ψ)) ≈ sin(1) atol = 1.0e-4
        end
    end

    @testset "Σₖ sin(2πkx)/k" begin
        freqs = (1, 4, 16, 64)
        for graph in graphs_1D, piv in pivs
            f = x -> sum(sin(2 * pi * k * sum(x)) / k for k in freqs)
            ψ = tci(f, graph.g, graph.s; pivotsearch = piv, nsweeps = 6, cutoff = tol, maxdim = 12)

            # exp(2πikx) is rank 1 for every k, so each sin term is exactly rank 2
            @test maxvirtualdim(ψ) <= 2 * length(freqs)
            for x in xs
                @test evaluate(ψ, x) ≈ f(x) atol = 1.0e-6
            end
            @test abs(integrate(ψ)) <= 1.0e-4
        end
    end

    @testset "1/(1+x+y)" begin
        for graph in graphs_2D, piv in pivs
            f = x -> 1 / (1 + x[1] + x[2])
            pts = ([0.0, 0.0], [1 / 4, 1 / 2], [7 / 8, 1 / 8], [15 / 16, 15 / 16])

            ψ = tci(f, graph.g, graph.s; pivotsearch = piv, nsweeps = 5, cutoff = tol, maxdim = 20)
            err = maximum(abs(evaluate(ψ, p) - f(p)) for p in pts)

            for x in xs, y in ys
                @test evaluate(ψ, [x, y]) ≈ f([x, y]) atol = 1.0e-6
            end
            @test maxvirtualdim(ψ) > 2
            @test integrate(ψ) ≈ 3 * log(3) - 4 * log(2) atol = 1e-4
        end
    end
end