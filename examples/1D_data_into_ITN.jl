using ITensorNumericalAnalysis

using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: NamedGraph
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using Random: Random
using ITensorNetworks: maxlinkdim
using Plots

function eval_L2(s, ψ, f)
  x_vals = grid_points(s, 1)
  error = 0
  # (∑|ψ(x) - f(x)|^2)
  for (i, x) in enumerate(x_vals)
    error += (real(evaluate(ψ, [x])) - f(x))^2
  end
  return error
end

Random.seed!(1234)

L = 10
g = named_comb_tree((1, L))
s = continuous_siteinds(g; map_dimension=1)

f(x) = 1 * exp(-(x - 0.5)^2 / 0.01) + 5 * x^2
#f(x) = sin(30π*x)
noise_level = 0.0
x_vals = [collect(range(0.0, 0.5; length=50)); collect(range(0.51, 1.0; length=50))]
data = [f(x) + noise_level * randn() for x in x_vals]

### Plot one result
ψ_f = data_itn(s, data, x_vals; mode="fourier", cutoff=1e-3, max_coeffs=300)
println("length of coeffs vector: $(length(cf))")
ψ_f = truncate(ψ_f; cutoff=1e-10)
x_vals_ψ = grid_points(s, 1)
f_vals = zeros(length(x_vals_ψ))
# c_vals = zeros(length(x_vals))
for (i, x) in enumerate(x_vals_ψ)
  f_vals[i] = real(evaluate(ψ_f, [x]))
  #c_vals[i] = real(evaluate(ψ_c, [x]))
end
plot1 = plot(x_vals, data; label="true data")
plot!(x_vals_ψ, f_vals; label="fourier approx")

### Plot of error vs # terms
# max_terms = 201
# eval_range = 1:10:max_terms
# l = length(eval_range)

# fourier_L2 = zeros(l)

# for (j, i) in enumerate(eval_range)

#     println("$i terms")

#     #local ψ_c = function_itn(s, f, cutoff=1e-10, max_coeffs=i, mode="chebyshev")
#     local ψ_f, n_f = data_itn(s, data; cutoff=0.0, max_coeffs=i, mode="fourier")

#     #ψ_c = truncate(ψ_c; cutoff=1e-10)
#     ψ_f = truncate(ψ_f; cutoff=1e-10)

#     f = LinearInterpolation(x_vals, data, extrapolation_bc=Interpolations.Line())
#     fourier_L2[j] = eval_L2(s, ψ_f, f)
# end

# plot(eval_range, fourier_L2, label="Fourier L2", title="noisy data", xlabel="number of terms", ylabel="loss")
