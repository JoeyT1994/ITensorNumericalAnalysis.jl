"""Build the order L tensor corresponding to fx(x): x ∈ [0,1]."""
function build_full_rank_tensor(L::Int64, fx::Function)
  inds = [Index(2, "$i") for i in 1:L]
  dims = Tuple([2 for i in 1:L])
  array = zeros(dims)
  for i in 0:(2^(L) - 1)
    xis = digits(i; base=2, pad=L)
    x = sum([xis[i] / (2^i) for i in 1:L])
    array[Tuple(xis + ones(Int64, (L)))...] = fx(x)
  end

  return ITensor(array, inds)
end

function c_tensor(phys_ind::Index, virt_inds::Vector)
  inds = vcat(phys_ind, virt_inds)
  @assert allequal(ITensors.dim.(virt_inds))
  #Build tensor to be delta on inds and independent of phys_ind
  T = ITensor(0.0, inds...)
  for i in 1:ITensors.dim(phys_ind)
    for j in 1:ITensors.dim(first(virt_inds))
      ind_array = [v => j for v in virt_inds]
      T[phys_ind => i, ind_array...] = 1.0
    end
  end

  return T
end

function copy_tensor_network(s::IndsNetwork, linkdim::Int64)
  tn = randomITensorNetwork(s; link_space=linkdim)
  for v in vertices(tn)
    virt_inds = setdiff(inds(tn[v]), Index[only(s[v])])
    tn[v] = c_tensor(only(s[v]), virt_inds)
  end

  return tn
end

function operate(
  operator::AbstractITensorNetwork,
  ψ::ITensorNetworkFunction;
  truncate_kwargs=(;),
  kwargs...,
)
  ψ_tn = TTN(itensornetwork(ψ))
  ψO_tn = noprime(ITensors.contract(operator, ψ_tn; init=prime(copy(ψ_tn)), kwargs...))
  ψO_tn = truncate(ψO_tn; truncate_kwargs...)

  return ITensorNetworkFunction(ITensorNetwork(ψO_tn), bit_map(ψ))
end
