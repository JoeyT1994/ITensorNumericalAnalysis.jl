using Graphs: AbstractGraph
using ITensors: ITensors, Index, dim, inds, siteinds
using ITensorNetworks: IndsNetwork, random_tensornetwork, vertex_tag

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

function ITensors.inds(s::IndsNetwork, v)
  return s[v]
end

function ITensors.inds(s::IndsNetwork, verts::Vector)
  return reduce(vcat, [inds(s, v) for v in verts])
end

function ITensors.inds(s::IndsNetwork)
  return inds(s, vertices(s))
end

function continuous_siteinds(g::AbstractGraph; base=2)
  is = IndsNetwork(g)
  for v in vertices(g)
    is[v] = [Index(base, "Digit, V$(vertex_tag(v))")]
  end
  return is
end

function base(s::IndsNetwork)
  indices = inds(s)
  dims = dim.(indices)
  @assert all(d -> d == first(dims), dims)
  return first(dims)
end
