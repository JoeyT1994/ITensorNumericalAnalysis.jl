using ITensorNumericalAnalysis

using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: NamedGraph
using NamedGraphs.GraphsExtensions: rename_vertices
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_binary_tree
using Random
using ITensorNetworks: maxlinkdim
using Plots

Random.seed!(1234)

L = 10
#comb tree
g = named_comb_tree((1, L))

#binary tree
# g = named_binary_tree(L)
# g = rename_vertices(v -> join(["$i," for i in v]), g)

#random tree
# g = NamedGraph(SimpleGraph(uniform_tree(L)))
# g = rename_vertices(v -> (v, 1), g)

s = continuous_siteinds(g; map_dimension=1)

function eval_L1(s, ψ, f)
  x_vals = grid_points(s, 1)
  error = 0
  # (∑|ψ(x) - f(x)|)
  for (i, x) in enumerate(x_vals)
    error += abs(real(evaluate(ψ, [x])) - f(x))
  end
  return error
end

function eval_L2(s, ψ, f)
  x_vals = grid_points(s, 1)
  error = 0
  # (∑|ψ(x) - f(x)|^2)
  for (i, x) in enumerate(x_vals)
    error += (real(evaluate(ψ, [x])) - f(x))^2
  end
  return error / length(x_vals)
end

function L2_error(truth, est)
  if length(truth) != length(est)
    throw("not equal length!")
  end
  return sum((truth .- est) .^ 2) / length(truth)
end

### Potential functions to choose from

#piecewise function
#f(x) = x < 0.75 ? x < 0.25 ? 0 : sin(4π * (x - 0.25)) : 0
#gaussian
#f(x) = exp(-(x - 0.5)^2 / 0.01)

#almost 1/x function. this breaks it if the cutoff is not fairly large/max_coeffs fairly small
# f(x) = 1/(.01+x) + 5*x^2

#f(x) = sin(1/(.01+x))
#f(x) = cos(π*x) + 5*x
#f(x) = sin(20*x^(1/7))

#
# Sum of random Gaussians
#
Ng = 40
ω = 0.001
ws = [ω * rand() for n in 1:Ng]
xs = [rand() for n in 1:Ng]
hs = [randn() for n in 1:Ng]
step_size = 2.5 * rand()
step_location = 0.4
function f(x::Number)
  val = sum(g -> hs[g] * exp(-(x[1] - xs[g])^2 / ws[g]), 1:Ng)
  val += (x[1] > step_location) ? step_size : 0.0
  return val
end

### Plotting f(x) vs ITN encodings using Fourier or Chebyshev decompositions

cheb_max = 25
fourier_max = 100
cheb_cut = 1e-3
fourier_cut = 1e-3

ψ_c = function_itn(
  s, f; cutoff=cheb_cut, max_coeffs=cheb_max, mode="chebyshev", by_mag=false
)
ψ_f = function_itn(s, f; cutoff=fourier_cut, max_coeffs=fourier_max, mode="fourier")



println("truncating...")
ψ_c = truncate(ψ_c; cutoff=1e-10)
ψ_f = truncate(ψ_f; cutoff=1e-10)

println("chebyshev maxlinkdim $(maxlinkdim(ψ_c))")
println("fourier maxlinkdim $(maxlinkdim(ψ_f))")

fourier_L1 = eval_L1(s, ψ_f, f)
fourier_L2 = eval_L2(s, ψ_f, f)
cheb_L1 = eval_L1(s, ψ_c, f)
cheb_L2 = eval_L2(s, ψ_c, f)
println("Fourier L2: $fourier_L2")
println("Chebyshev L2: $cheb_L2")

x_vals = grid_points(s, 1)
f_vals = [real(evaluate(ψ_f, [x])) for x in x_vals]
c_vals = [real(evaluate(ψ_c, [x])) for x in x_vals]
true_vals = [f(x) for x in x_vals]

# println("Fourier L2 (second method): $(L2_error(true_vals,f_vals))")
# println("Cheb L2 (second method): $(L2_error(true_vals, c_vals))")

println("Plotting f(x)")
plot(x_vals, true_vals; label="true function", title="f(x)")
plot!(x_vals, f_vals; label="Fourier ")
plot!(x_vals, c_vals; label="Chebyshev")

### Plots of Loss vs # terms and Maxlinkdim vs # terms

# max_terms = 25
# eval_range = 1:3:max_terms
# l = length(eval_range)

# cheb_L1 = zeros(l)
# cheb_L2 = zeros(l)
# fourier_L1 = zeros(l)
# fourier_L2 = zeros(l)
# cheb_mld = zeros(l)
# fourier_mld = zeros(l)

# for (j, i) in enumerate(eval_range)
#   println("$i terms")

#   local ψ_c = function_itn(s, f; cutoff=1e-10, max_coeffs=i, mode="chebyshev")
#   local ψ_f = function_itn(s, f; cutoff=1e-10, max_coeffs=i, mode="fourier")

#   ψ_c = truncate(ψ_c; cutoff=1e-10)
#   ψ_f = truncate(ψ_f; cutoff=1e-10)

#   cheb_L1[j] = eval_L1(s, ψ_c, f)
#   cheb_L2[j] = eval_L2(s, ψ_c, f)
#   fourier_L1[j] = eval_L1(s, ψ_f, f)
#   fourier_L2[j] = eval_L2(s, ψ_f, f)
#   cheb_mld[j] = maxlinkdim(ψ_c)
#   fourier_mld[j] = maxlinkdim(ψ_f)
# end

# plot1 = plot(
#   eval_range,
#   cheb_L1;
#   label="Cheb L1",
#   title="f(x) = exp(-(x-0.5)^2/.01)",
#   xlabel="number of terms",
#   ylabel="loss",
# )
# plot!(eval_range, cheb_L2; label="Cheb L2")
# plot!(eval_range, fourier_L1; label="Fourier L1")
# plot!(eval_range, fourier_L2; label="Fourier L2")

# plot2 = plot(
#   eval_range,
#   cheb_mld;
#   label="Chebyshev",
#   title="f(x) = exp(-(x-0.5)^2/.01)",
#   xlabel="number of terms",
#   ylabel="max bond dimension",
# )
# plot!(eval_range, fourier_mld; label="Fourier")

# display(plot1, plot2, layout=(1,2))
