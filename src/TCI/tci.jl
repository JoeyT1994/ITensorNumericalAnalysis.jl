using Graphs: is_tree, vertices
using ITensors: dim
using ITensorNetworks: ITensorNetwork, ttn
using Dictionaries: Dictionary

random_initial_pivot(s::IndsNetworkMap) = [v => rand(1:dim(ind)) for v in vertices(s)]

#f should be an ndimensional function that maps a vector of scalars of length ndimensional to a scalar
function interpolate(f, s::IndsNetworkMap; kwargs...)
  @assert is_tree(s)

  vs, vs_new = collect(vertices(s)),
  [(vertex_dimension(s, v), vertex_digit(s, v)) for v in vs]
  forward_dict, backward_dict = Dictionary(vs, vs_new), Dictionary(vs_new, vs)
  s_renamed = rename_vertices(v -> forward_dict[v], s)

  tn = const_itn(s_renamed; linkdim=1)
  initial_pivot =
    random_initial_pivot(s_renamed),     #Call out to function in ITensorsTCI.jl is here
    tn = interpolate(
      input -> f(input_to_scalars(input)), ttn(itensornetwork(tn)); initial_pivot, kwargs...
    )

  tn = rename_vertices(v -> backward_dict[v], tn)

  return ITensorNetworkFunction(ITensorNetwork(tn), s)
end

#Takes a vector of [(dimension, digit) => bit] and converts to vector of scalars
function input_to_scalars(input)
  ndims = maximum(first.(input))
  x = zeros(ndims)
  for pair in input
    (i, j) = pair[1]
    bit = pair[2] - 1
    x[i] += bit / 2.0^j
  end
  return x
end
