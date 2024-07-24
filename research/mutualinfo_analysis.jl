
using SpecialFunctions
using Plots
using LaTeXStrings
using Random

using NPZ

Random.seed!(1234)

include("mutualinfo.jl")

function main()
    modes = ["RandPlaneWaves"]
    for mode in modes
        if mode == "Airy"
            ω = 100
            f = x -> airyai(- ω*only(x))
            dim = 1
        elseif mode == "Weierstrass"
            ks = weirstrass_coefficients(100, 3)
            f = x -> calulate_weirstrass(only(x), ks)
            dim = 1
        elseif mode == "Laguerre"
            ks = laguerre_coefficients(40)
            f = x -> evaluate_polynomial(only(x), ks)
            dim = 1
        elseif mode == "RandPlaneWaves"
            nterms = 30
            kxs, kys, kzs = [i*randn() for i in 1:nterms], [i*randn() for i in 1:nterms], [i*randn() for i in 1:nterms]
            f = x -> sum([cos(kxs[i]*x[1] + kys[i]*x[2] + kzs[i]*x[3]) for i in 1:nterms])
            dim = 3
        end
        nsamples = 1000
        L = 20
        d = 5
        m = generate_mi_matrix(f, nsamples, L, dim)
        corrs = m[1, d, 1, :]
        print(corrs)
        plot([i for i in 1:length(corrs)], corrs)

        #file_name = "/Users/jtindall/Files/Data/TCI/"*"MutualInfo"*mode*"L$(L)Nsamples$(nsamples).npz"
        #npzwrite(file_name, m = m)
    end
end

main()


