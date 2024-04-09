using Graphs: AbstractGraph
using ITensors: ITensors, Index, dim, inds, siteinds
using ITensorNetworks: random_tensornetwork, IndsNetwork, vertex_tag

"""Build the order L tensor corresponding to fx(x): x ∈ [0,1], default decomposition is binary"""
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

"""Build the tensor C such that C_{phys_ind, virt_inds...} = delta_{virt_inds...}"""
function c_tensor(phys_ind::Index, virt_inds::Vector)
  inds = vcat(phys_ind, virt_inds)
  @assert allequal(dim.(virt_inds))
  T = ITensor(0.0, inds...)
  for i in 1:dim(phys_ind)
    for j in 1:dim(first(virt_inds))
      ind_array = [v => j for v in virt_inds]
      T[phys_ind => i, ind_array...] = 1.0
    end
  end

  return T
end

"""Tag for a vertex based on its dimension (Dim) and digit (N)"""
function digit_tag(bm::BitMap, v)
  dig = digit(bm, v)
  dim = dimension(bm, v)
  return "N $dig, Dim $dim"
end

"""Generate network of physical indices given a graph and a bitmap"""
function ITensors.siteinds(g::AbstractGraph, bm::BitMap)
  is = IndsNetwork(g)
  for v in vertices(g)
    is[v] = [Index(base(bm), "Digit, $(digit_tag(bm, v)), V$(vertex_tag(v))")]
  end
  return is
end
