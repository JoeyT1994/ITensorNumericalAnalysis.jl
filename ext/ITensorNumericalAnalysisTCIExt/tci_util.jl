using Graphs: is_tree, vertices
using ITensors: dim
using ITensorNetworks: ITensorNetwork, ttn
using Dictionaries: Dictionary

using ITensorTCI: ITensorTCI
using ITensorNumericalAnalysis:
  IndsNetworkMap,
  vertex_dimension,
  vertex_digit,
  rename_vertices,
  const_itn,
  ITensorNetworkFunction,
  itensornetwork,
  base,
  dimension

random_initial_pivot(s::IndsNetworkMap) = [v => rand(1:dim(s[v])) for v in vertices(s)]

#f should be an ndimensional function that maps a vector of scalars of length ndimensional to a scalar
function ITensorTCI.interpolate(
  f::Function, s::IndsNetworkMap; initial_state=nothing, initial_pivot=nothing, kwargs...
)
  @assert is_tree(s)

  vs = collect(vertices(s))
  vs_new = [(vertex_dimension(s, v), vertex_digit(s, v)) for v in vs]
  forward_dict, backward_dict = Dictionary(vs, vs_new), Dictionary(vs_new, vs)
  s_renamed = rename_vertices(v -> forward_dict[v], s)

  if isnothing(initial_state)
    tn = const_itn(s_renamed; linkdim=1)
  else
    tn = rename_vertices(v -> forward_dict[v], initial_state)
  end

  if isnothing(initial_pivot)
    initial_pivot = random_initial_pivot(s_renamed)
  else
    # manually rename
    # assuming from calculate_ind_values
    initial_pivot = [forward_dict[v] => initial_pivot[only(s[v])] + 1 for v in vertices(s)]
  end

  tn = ITensorTCI.interpolate(
    input -> f(input_to_scalars(input; ndims=dimension(s), base=float(base(s)))),
    ttn(itensornetwork(tn));
    initial_pivot,
    kwargs...,
  )

  tn = rename_vertices(v -> backward_dict[v], tn)

  return ITensorNetworkFunction(ITensorNetwork(tn), s)
end

#Takes a vector of [(dimension, digit) => bit] and converts to vector of scalars
function input_to_scalars(input; ndims, base=2.0)
  x = zeros(ndims)
  for pair in input
    (i, j) = pair[1]
    bit = pair[2] - 1
    x[i] += bit / base^j
  end
  return x
end
