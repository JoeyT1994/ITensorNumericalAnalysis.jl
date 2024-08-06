using ITensorNetworks:
  ITensorNetworks, ITensorNetwork, AbstractITensorNetwork, scalar, inner, TreeTensorNetwork
using ITensors: ITensor

" Naive integration of a function in all dimensions ∫₀¹f({r})d{r} "
function integrate(fitn::ITensorNetworkFunction; alg=default_contraction_alg(), kwargs...)
  fitn = copy(fitn)
  s = indsnetwork(indsnetworkmap(fitn))
  for v in vertices(fitn)
    indices = inds(s, v)
    fitn[v] = fitn[v] * ITensor(eltype(fitn[v]), 0.5, indices...)
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

function partial_integrate(
  fitn::ITensorNetworkFunction, dims::Vector{Int}; reduce_network=true
)
  s = indsnetworkmap(fitn)
  new_imap = copy(indexmap(s))
  fitn = copy(itensornetwork(fitn))
  for v in dimension_vertices(s, dims)
    sinds_dim = filter(i -> dimension(s, i) ∈ dims, s[v])
    for sind in sinds_dim
      fitn[v] *= ITensor(eltype(fitn[v]), 0.5, sind)
      new_imap = rem_index(new_imap, sind)
    end
  end
  if reduce_network
    fitn = merge_internal_tensors(fitn)
  end
  new_inmap = IndsNetworkMap(siteinds(fitn), new_imap)
  return ITensorNetworkFunction(fitn, new_inmap)
end
