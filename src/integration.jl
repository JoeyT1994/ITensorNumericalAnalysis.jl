using ITensorNetworks:
  ITensorNetworks, ITensorNetwork, AbstractITensorNetwork, scalar, inner, TreeTensorNetwork
using ITensors: ITensor

" Naive integration of a function in all dimensions ∫₀¹f({r})d{r} "
function integrate(
  fitn::ITensorNetworkFunction; alg=default_contraction_alg(), take_sum=false, kwargs...
)
  fitn = copy(fitn)
  s = indsnetwork(indsnetworkmap(fitn))
  c = take_sum ? 1.0 : (1.0 / base(s))
  for v in vertices(fitn)
    indices = inds(s, v)
    fitn[v] = fitn[v] * ITensor(eltype(fitn[v]), c, indices...)
  end
  return scalar(itensornetwork(fitn); alg, kwargs...)
end

" Naive integration of a operator applied to a function in all dimensions ∫₀¹ (operator*f)({r})d{r} "
function integrate(
  operator::AbstractITensorNetwork,
  fitn::ITensorNetworkFunction;
  alg=default_contraction_alg(),
  kwargs...,
)
  s = indsnetwork(indsnetworkmap(fitn))
  # create basic integrator to apply to the operator|fitn> state
  ∑ = ITensorNetwork(eltype(first(fitn)), v -> [0.5, 0.5], s)
  return inner(∑, operator, itensornetwork(fitn); alg, kwargs...)
end

""" Partial integration of function over specified dimensions. By default reduce the resulting network down to a new, smaller one """
function partial_integrate(
  fitn::ITensorNetworkFunction, dims::Vector{Int}; merge_vertices=true, take_sum=false
)
  s = indsnetworkmap(fitn)
  new_imap = copy(indexmap(s))
  fitn = copy(itensornetwork(fitn))
  c = take_sum ? 1.0 : (1.0 / base(s))
  for v in dimension_vertices(s, dims)
    sinds_dim = filter(i -> dimension(s, i) ∈ dims, s[v])
    for sind in sinds_dim
      fitn[v] *= ITensor(eltype(fitn[v]), c, sind)
      new_imap = rem_index(new_imap, sind)
    end
  end
  if merge_vertices
    fitn = merge_internal_tensors(fitn)
  end
  new_inmap = IndsNetworkMap(siteinds(fitn), new_imap)
  return ITensorNetworkFunction(fitn, new_inmap)
end
