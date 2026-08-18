using NamedGraphs.NamedGraphGenerators: named_comb_tree
using ITensorNumericalAnalysis: continuous_siteinds, tci, evaluate

function main(;
    pivotsearch = FullPivot(),
    nsweeps = 5,
    mindim = 5,
    maxdim = 15,
    cutoff = 1e-12
)
    L = 8
    map_dimension = 2
    g = named_comb_tree((L, map_dimension))
    imap = continuous_siteinds(g; map_dimension)
    f = x -> exp(-x[1]^2 - 2.0*x[2]^2)
    ψ = tci(
        f, 
        g,
        imap; 
        pivotsearch,
        nsweeps,
        mindim,
        maxdim,
        cutoff,
        outputlevel = 1
    )
    rng = MersenneTwister(1234)
    pts = [(rand(rng, 0:255)/256, rand(rng, 0:255)/256) for _ in 1:100]
    errs = [abs(evaluate(ψ, [x,y]) - f([x,y])) for (x,y) in pts]
    @show maximum(errs), sum(errs)/length(errs)
end

main(;)