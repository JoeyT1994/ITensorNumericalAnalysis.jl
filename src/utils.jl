using Graphs: AbstractGraph
using ITensors: ITensors, Index, dim, inds
using ITensorNetworks: IndsNetwork, ITensorNetworks, random_tensornetwork

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
function c_tensor(eltype::Type, phys_inds::Vector, virt_inds::Vector)
  @assert allequal(dim.(virt_inds))
  T = delta(eltype, virt_inds)
  T = T * ITensor(1, phys_inds...)
  return T
end

function c_tensor(phys_inds::Vector, virt_inds::Vector)
  return c_tensor(Float64, phys_inds, virt_inds)
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

#ToDo: generalize beyond 2-site
#ToDo: remove concept of orthogonality center for generality
function ITensorNetworks.current_ortho(sweep_plan, which_region_update)
  regions = first.(sweep_plan)
  region = regions[which_region_update]
  current_verts = support(region)
  if !isa(region, AbstractEdge) && length(region) == 1
    return only(current_verts)
  end
  # look forward
  other_regions = filter(
    x -> !(issetequal(x, current_verts)), support.(regions[(which_region_update + 1):end])
  )
  # find the first region that has overlapping support with current region 
  ind = findfirst(x -> !isempty(intersect(support(x), support(region))), other_regions)
  if isnothing(ind)
    # look backward
    other_regions = reverse(
      filter(
        x -> !(issetequal(x, current_verts)),
        support.(regions[1:(which_region_update - 1)]),
      ),
    )
    ind = findfirst(x -> !isempty(intersect(support(x), support(region))), other_regions)
  end
  @assert !isnothing(ind)
  future_verts = union(support(other_regions[ind]))
  # return ortho_ceter as the vertex in current region that does not overlap with following one
  overlapping_vertex = intersect(current_verts, future_verts)
  nonoverlapping_vertex = only(setdiff(current_verts, overlapping_vertex))
  return nonoverlapping_vertex
end
