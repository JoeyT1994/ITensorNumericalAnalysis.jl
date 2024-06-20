using ITensorNumericalAnalysis

using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: NamedGraph
using NamedGraphs.GraphsExtensions: rename_vertices
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using Random
using ITensorNetworks: maxlinkdim
using Plots
using Images: imresize
using FileIO: load

function eval_L2(f_eval, ψ_eval)
  N = length(f_eval)
  return sum((f_eval .- ψ_eval) .^ 2) / N
end

L = 5
g = named_comb_tree((2, L))
s = continuous_siteinds(g; map_dimension=2)

img = load("examples/Statue_Of_Liberty.png")
img_matrix = Array{Gray}(img)
img_float_matrix = (x -> float(x)).(Float32.(img_matrix))
data = imresize(img_float_matrix, (2^L, 2^L))
data = reverse(data; dims=1)

println("size of matrix: $(size(data))")

elapsed_time = @elapsed begin
  ψ_f, cf = data_itn(s, data; mode="fourier", cutoff=1e-4, max_coeffs=100, by_mag=true)
end
println("constructing ψ_f: ", elapsed_time, " seconds")

# elapsed_time = @elapsed begin
#     ψ_c, cc = data_itn(s, data; mode="chebyshev", cutoff=1e-3, max_coeffs = 100)
# end
# println("constructing ψ_c: ", elapsed_time, " seconds")

#### Plot Error vs Max Bond Dim

# bondrange = 1:4:50
# loss = zeros(length(bondrange))
# x_vals = grid_points(s, 1)
# y_vals = grid_points(s, 2)
# for (i, bond) in enumerate(bondrange)
#   ψ_bond = truncate(ψ_f; maxdim=bond)
#   ψ_bond_eval = [
#     real(ITensorNumericalAnalysis.evaluate(ψ_bond, [x, y])) for x in x_vals, y in y_vals
#   ]
#   loss[i] = eval_L2(data, ψ_bond_eval)
# end

# plot(
#   bondrange,
#   loss;
#   xlabel="max bond dimension",
#   ylabel="L2 error",
#   title="64 x 64 statue of liberty image",
# )

### Compare images side by side

ψ_f = truncate(ψ_f; cutoff=1e-16)
# ψ_c = truncate(ψ_c, cutoff=1e-16)

println("size of coeffs of ψ_f: $(size(cf))")
println("maxlinkdim of ψ_f: $(maxlinkdim(ψ_f))")
# println("size of coeffs of ψ_c: $(size(cc))")
# println("maxlinkdim of ψ_c: $(maxlinkdim(ψ_c))")

x_vals = grid_points(s, 1)
y_vals = grid_points(s, 2)

elapsed_time = @elapsed begin
  ψ_f_eval = [
    real(ITensorNumericalAnalysis.evaluate(ψ_f, [x, y])) for x in x_vals, y in y_vals
  ]
end
println("ψ_f evaluation elapsed time: ", elapsed_time, " seconds")

# elapsed_time = @elapsed begin
#     ψ_c_eval = [
#     real(ITensorNumericalAnalysis.evaluate(ψ_c, [x, y])) for x in x_vals, y in y_vals
#     ]
# end
# println("ψ_c evaluation elapsed time: ", elapsed_time, " seconds")

fourier_loss = eval_L2(data, ψ_f_eval)
# chebyshev_loss = eval_L2(data, ψ_c_eval)
println("fourier L2 error:  $fourier_loss")
# println("chebyshev L2 error: $chebyshev_loss")

plot1 = heatmap(x_vals, y_vals, data; title="true image")
plot2 = heatmap(x_vals, y_vals, ψ_f_eval; title="fourier TN") #fourier TN approximation
# plot3 = heatmap(x_vals, y_vals, ψ_c_eval; title="chebyshev TN")

combined = plot(plot1, plot2; layout=(1, 2), size=(700, 280))
