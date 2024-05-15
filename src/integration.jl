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
