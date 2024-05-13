using Graphs: AbstractGraph
using ITensors:
  ITensors,
  Index,
  delta,
  dim,
  inds,
  siteinds,
  @OpName_str,
  @SiteType_str,
  val,
  state,
  ValName,
  StateName,
  SiteType,
  op
using ITensorNetworks: IndsNetwork, random_tensornetwork, vertex_tag

# reuse Qudit definitions for now

function ITensors.val(::ValName{N}, ::SiteType"Digit") where {N}
  return parse(Int, String(N)) + 1
end

function ITensors.state(::StateName{N}, ::SiteType"Digit", s::Index) where {N}
  n = parse(Int, String(N))
  st = zeros(dim(s))
  st[n + 1] = 1.0
  return ITensor(st, s)
end

function ITensors.op(::OpName"D+", ::SiteType"Digit", s::Index)
  d = dim(s)
  o = zeros(d, d)
  o[2, 1] = 1
  return ITensor(o, s, s')
end
function ITensors.op(::OpName"D-", ::SiteType"Digit", s::Index)
  d = dim(s)
  o = zeros(d, d)
  o[1, 2] = 1
  return ITensor(o, s, s')
end
function ITensors.op(::OpName"Ddn", ::SiteType"Digit", s::Index)
  d = dim(s)
  o = zeros(d, d)
  o[1, 1] = 1
  return ITensor(o, s, s')
end
function ITensors.op(::OpName"Dup", ::SiteType"Digit", s::Index)
  d = dim(s)
  o = zeros(d, d)
  o[2, 2] = 1
  return ITensor(o, s, s')
end

"""Build the order L tensor corresponding to fx(x): x ∈ [0,1], default decomposition is binary"""
function build_full_rank_tensor(L::Int, fx::Function; base::Int=2)
  inds = [Index(base, "$i") for i in 1:L]
  dims = Tuple([base for i in 1:L])
  array = zeros(dims)
  for i in 0:(base^(L) - 1)
    xis = digits(i; base, pad=L)
    x = sum([xis[i] / (base^i) for i in 1:L])
    array[Tuple(xis + ones(Int, (L)))...] = fx(x)
  end

  return ITensor(array, inds)
end

"""Build the tensor C such that C_{phys_ind, virt_inds...} = delta_{virt_inds...}"""
function c_tensor(phys_inds::Vector, virt_inds::Vector)
  @assert allequal(dim.(virt_inds))
  T = delta(virt_inds)
  for ind in phys_inds
    T = T * ITensor(1.0, ind)
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
  return inds(s, collect(vertices(s)))
end

function base(s::IndsNetwork)
  indices = inds(s)
  dims = dim.(indices)
  @assert all(d -> d == first(dims), dims)
  return first(dims)
end

function digit_siteinds(g::AbstractGraph; base=2, is_complex = false)
  is = IndsNetwork(g)
  for v in vertices(g)
    if !is_complex
      is[v] = [Index(base, "Digit, V$(vertex_tag(v))")]
    else
      is[v] = [Index(base, "Digit Re, V$(vertex_tag(v))"), Index(base, "Digit Im, V$(vertex_tag(v))")]
    end
  end
  return is
end
