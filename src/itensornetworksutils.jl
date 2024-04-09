using ITensors: Index, dim, inds
using ITensorNetworks: random_tensornetwork, IndsNetwork

"""Build the order L tensor corresponding to fx(x): x ∈ [0,1]."""
function build_full_rank_tensor(L::Int64, fx::Function; base::Int64=2)
  inds = [Index(base, "$i") for i in 1:L]
  dims = Tuple([base for i in 1:L])
  array = zeros(dims)
  for i in 0:(base^(L) - 1)
    xis = digits(i; base, pad=L)
    x = sum([xis[i] / (base^i) for i in 1:L])
    array[Tuple(xis + ones(Int64, (L)))...] = fx(x)
  end

  return ITensor(array, inds)
end

function c_tensor(phys_ind::Index, virt_inds::Vector)
  inds = vcat(phys_ind, virt_inds)
  @assert allequal(dim.(virt_inds))
  #Build tensor to be delta on inds and independent of phys_ind
  T = ITensor(0.0, inds...)
  for i in 1:dim(phys_ind)
    for j in 1:dim(first(virt_inds))
      ind_array = [v => j for v in virt_inds]
      T[phys_ind => i, ind_array...] = 1.0
    end
  end

  return T
end

function copy_tensor_network(s::IndsNetwork; linkdim::Int64=1)
  tn = random_tensornetwork(s; link_space=linkdim)
  for v in vertices(tn)
    virt_inds = setdiff(inds(tn[v]), Index[only(s[v])])
    tn[v] = c_tensor(only(s[v]), virt_inds)
  end

  return tn
end
