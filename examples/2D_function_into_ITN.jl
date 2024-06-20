using ITensorNumericalAnalysis

using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: NamedGraph
using NamedGraphs.GraphsExtensions: rename_vertices
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_binary_tree
using Random
using ITensorNetworks: maxlinkdim
using Plots

Random.seed!(1234)

function eval_L2(f_eval, ψ_eval)
  return sum((f_eval .- ψ_eval) .^ 2)
end

L = 10

#comb tree
g = named_comb_tree((2, L ÷ 2))

#uniform tree
# g = NamedGraph(SimpleGraph(uniform_tree(L)))
# g = rename_vertices(v -> (v, 1), g)

#binary tree
# g = named_binary_tree(4)
# g = rename_vertices(v -> join(["$i," for i in v]), g)

s = continuous_siteinds(g; map_dimension=2)

x_vals = grid_points(s, 1)
y_vals = grid_points(s, length(x_vals), 2)

### Example Test Functions to use
f(x, y) = exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.05)
#f(x, y) = exp(x) - exp(y)
#f(x, y) = cos(20 * (x + y)) + sin(50 * (x - y))

ψ_f, cf = function_itn(s, f; mode="fourier", cutoff=1e-3, max_coeffs=100)
ψ_c, cc = function_itn(s, f; mode="chebyshev", cutoff=1e-2, max_coeffs=36, by_mag=false)

println("maxlinkdim of ψ_f after truncation: $(maxlinkdim(ψ_f))")
println("maxlinkdim of ψ_c after truncation: $(maxlinkdim(ψ_c))")

f_eval = [f(x, y) for x in x_vals, y in y_vals]
println("f_eval done")
ψ_f_eval = [
  real(ITensorNumericalAnalysis.evaluate(ψ_f, [x, y])) for x in x_vals, y in y_vals
]
println("ψ_f_eval done")
ψ_c_eval = [
  real(ITensorNumericalAnalysis.evaluate(ψ_c, [x, y])) for x in x_vals, y in y_vals
]
println("ψ_c_eval done")

fourier_loss = eval_L2(f_eval, ψ_f_eval)
chebyshev_loss = eval_L2(f_eval, ψ_c_eval)
println("fourier L2 error:  $fourier_loss")
println("chebyshev L2 error: $chebyshev_loss")

plot1 = heatmap(
  x_vals, y_vals, f_eval; title="true function", xlabel="x", ylabel="y", zlabel="f(x,y)"
) #true function
plot2 = heatmap(
  x_vals, y_vals, ψ_f_eval; title="fourier TN", xlabel="x", ylabel="y", zlabel="ψ(x,y)"
) #fourier TN approximation
plot3 = heatmap(
  x_vals, y_vals, ψ_c_eval; title="chebyshev TN", xlabel="x", ylabel="y", zlabel="ψ(x,y)"
) #chebyshev TN approximation

combined_plot = plot(plot1, plot2, plot3; layout=(1, 3), size=(1200, 280))
display(combined_plot)
