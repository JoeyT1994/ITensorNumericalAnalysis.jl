using ITensorNumericalAnalysis

using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: NamedGraph
using NamedGraphs.GraphsExtensions: rename_vertices
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_binary_tree
using Random
using ITensorNetworks: maxlinkdim
using Plots
using Interpolations

Random.seed!(1234)

function eval_L2(s, f, ψ_f)
  x_vals = grid_points(s, 1)
  y_vals = grid_points(s, 2)
  error_sq = [
    (f(x, y) - real(ITensorNumericalAnalysis.evaluate(ψ_f, [x, y])))^2 for x in x_vals,
    y in y_vals
  ]
  N = length(error_sq)
  return sum(error_sq) / N
end

L = 8
g = named_comb_tree((2, L ÷ 2))

# g = NamedGraph(SimpleGraph(uniform_tree(L)))
# g = rename_vertices(v -> (v, 1), g)

# g = named_binary_tree(3)
# g = rename_vertices(v -> join(["$i," for i in v]), g)

s = continuous_siteinds(g; map_dimension=2)

### Example Test Functions to generate data off of
#f(x, y) = exp( -( (x-0.5)^2 + (y-0.5)^2 )/.05)
f(x, y) = sin(2π * x) + sin(2π * y)

xrange = 0:0.05:1
yrange = 0:0.05:1
noise_level = 0.0
data = [f(x, y) + noise_level * randn() for x in xrange, y in yrange]

elapsed_time = @elapsed begin
  ψ_f = data_itn(s, data, (xrange, yrange); mode="fourier", cutoff=1e-3, max_coeffs=20)
end
println("ψ_f time: ", elapsed_time)

elapsed_time = @elapsed begin
  ψ_c = data_itn(
    s, data, (xrange, yrange); mode="chebyshev", cutoff=1e-3, max_coeffs=20, by_mag=false
  )
end
println("ψ_c time: ", elapsed_time)

ψ_f = truncate(ψ_f; cutoff=1e-10)
println("maxlinkdim of ψ_f after truncation: $(maxlinkdim(ψ_f))")

ψ_c = truncate(ψ_c; cutoff=1e-10)
println("maxlinkdim of ψ_c after truncation: $(maxlinkdim(ψ_c))")

x_vals = grid_points(s, 1)
y_vals = grid_points(s, length(x_vals), 2)
#f_eval = [f(x,y) for x in x_vals, y in y_vals]
ψ_f_eval = [
  real(ITensorNumericalAnalysis.evaluate(ψ_f, [x, y])) for x in x_vals, y in y_vals
]
ψ_c_eval = [
  real(ITensorNumericalAnalysis.evaluate(ψ_c, [x, y])) for x in x_vals, y in y_vals
]

g = LinearInterpolation((xrange, yrange), data; extrapolation_bc=Interpolations.Line())
fourier_loss = eval_L2(s, g, ψ_f)
chebyshev_loss = eval_L2(s, g, ψ_c)
println("fourier L2 error:  $fourier_loss")
println("chebyshev L2 error: $chebyshev_loss")

plot1 = heatmap(xrange, yrange, data; title="true data", xlabel="x", ylabel="y") #true data
plot2 = heatmap(
  x_vals, y_vals, ψ_f_eval; title="fourier TN", xlabel="x", ylabel="y", zlabel="ψ(x,y)"
) #fourier TN approximation
plot3 = heatmap(
  x_vals, y_vals, ψ_c_eval; title="chebyshev TN", xlabel="x", ylabel="y", zlabel="ψ(x,y)"
) #chebyshev TN approximation

combined_plot = plot(plot1, plot2, plot3; layout=(1, 3), size=(1200, 280))
display(combined_plot)
