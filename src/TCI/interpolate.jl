using Graphs: AbstractEdge, is_tree, nv
using ITensors: array, commonind, factorize, hastags, pause, permute, uniqueinds, order
using ITensorNetworks:
  ITensorNetworks,
  AbstractTTN,
  IndsNetwork,
  ITensorNetwork,
  TreeTensorNetwork,
  alternating_update,
  commonind,
  edgetype,
  orthogonalize,
  set_ortho_region,
  siteinds,
  support,
  tags,
  default_transform_operator,
  default_sweep_plans,
  ttn
using NamedGraphs.GraphsExtensions: is_leaf_vertex, incident_edges
using Observers: observer

function interpolate_extracter(state, projected_operator, region; internal_kwargs)
  @assert !(region isa AbstractEdge)
  gauge_center = first(region)
  state = interpolative_gauge(state, gauge_center)
  local_tensor = prod(state[v] for v in region)
  return state, projected_operator, local_tensor
end

function interpolate_updater(
  init;
  state!,
  projected_operator!,
  outputlevel,
  which_sweep,
  sweep_plan,
  which_region_update,
  internal_kwargs,
)
  ttnf = projected_operator![]
  ttn = state![]
  region = first(sweep_plan[which_region_update])
  site_inds = reduce(vcat, [ITensorNetworks.siteinds(ttn)[v] for v in region])
  site_ranges = [1:dim(s) for s in site_inds]
  link_inds = setdiff(inds(init), site_inds)
  link_ranges = [1:dim(l) for l in link_inds]
  Π = permute(init, site_inds..., link_inds...)
  A = array(Π)
  inf_norm_error = 0.0
  for link_vals in Iterators.product(link_ranges...)
    link_pivs = vcat([space(l)[lval] for (l, lval) in zip(link_inds, link_vals)]...)
    for site_vals in Iterators.product(site_ranges...)
      site_pivs = [ind => s for (ind, s) in zip(site_inds, site_vals)]
      arg = vcat(site_pivs, link_pivs)
      val = ttnf(arg)
      prev_val = A[site_vals..., link_vals...]
      A[site_vals..., link_vals...] = val
      inf_norm_error = max(inf_norm_error, abs(val - prev_val))
    end
  end
  return Π, (; inf_norm_error)
end

function interpolate_inserter(
  state::AbstractTTN,
  Pi::ITensor,
  region;
  maxdim=nothing,
  mindim=nothing,
  cutoff=nothing,
  internal_kwargs,
)
  state = copy(state)
  if length(ITensorNetworks.ortho_region(state)) != 1
    state = interpolative_gauge(state, first(support(region)))
    ortho_vert = first(support(region))
  else
    ortho_vert = only(ITensorNetworks.ortho_region(state))
  end

  center_vert = only(setdiff(support(region), [ortho_vert]))
  e = edgetype(state)(ortho_vert, center_vert)
  col_inds = uniqueinds(state[ortho_vert], state[center_vert])
  site_inds = vcat(ITensorNetworks.siteinds(state)[ortho_vert], ITensorNetworks.siteinds(state)[center_vert])
  #TODO: try to include tags(state,e), but resulting in extra quotation marks?
  ltags = "Link"
  C, Z, _ = interpolative(
    Pi, col_inds, site_inds; col_vertex=ortho_vert, tags=ltags, maxdim, mindim, cutoff
  )
  state[ortho_vert] = Z
  state[center_vert] = C
  state = set_ortho_region(state, [center_vert])
  return state, nothing
end

function interpolate_extracter_V2(
  state, projected_operator, region, gauge_center; internal_kwargs
)
  @assert !(region isa AbstractEdge)
  state = interpolative_gauge(state, gauge_center)

  @assert length(region) == 2
  vertex1,vertex2 = region
  site1 = only(siteinds(state,vertex1))
  site2 = only(siteinds(state,vertex2))

  col_inds1 = setdiff(uniqueinds(state[vertex1], state[vertex2]),[site1])
  C1, Z1, _ = interpolative_V2(state[vertex1], col_inds1; col_vertex=vertex1)

  col_inds2 = setdiff(uniqueinds(state[vertex2], state[vertex1]),[site2])
  C2, Z2, _ = interpolative_V2(state[vertex2], col_inds2; col_vertex=vertex2)

  C = C1*C2

  local_state = (;C,site1,site2,vertex1,vertex2,Z1,Z2)

  @assert order(C) <= 4
  return state, projected_operator, local_state
end

function interpolate_updater_V2(
  local_state;
  state!,
  projected_operator!,
  outputlevel,
  which_sweep,
  sweep_plan,
  which_region_update,
  internal_kwargs,
)
  Π0 = local_state[:C]
  (; time) = @timed begin
    ttnf = projected_operator![]
    ttn = state![]
    region = first(sweep_plan[which_region_update])
    site_inds = [only(siteinds(ttn, v)) for v in region]
    #site_inds = (local_state[:site1],local_state[:site2])
    site_ranges = [1:dim(s) for s in site_inds]
    link_inds = setdiff(inds(Π0), site_inds)
    link_ranges = [1:dim(l) for l in link_inds]
    Π = permute(Π0, site_inds..., link_inds...)
    A = array(Π)
    inf_norm_error = 0.0
    for link_vals in Iterators.product(link_ranges...)
      link_pivs = vcat([space(l)[lval] for (l, lval) in zip(link_inds, link_vals)]...)
      for site_vals in Iterators.product(site_ranges...)
        site_pivs = [v => s for (v, s) in zip(region, site_vals)]
        arg = vcat(site_pivs, link_pivs)
        @assert length(arg) == nv(ttn)
        val = ttnf(arg)
        prev_val = A[site_vals..., link_vals...]
        A[site_vals..., link_vals...] = val
        inf_norm_error = max(inf_norm_error, abs(val - prev_val))
      end
    end
  end
  local_state = merge(local_state,(;C=Π))
  return local_state, (; inf_norm_error, updater_time=time)
end

function interpolate_inserter_V2(
  state::AbstractTTN,
  local_state,
  region,
  ortho_vert;
  maxdim=nothing,
  mindim=nothing,
  cutoff=nothing,
  internal_kwargs,
)
  inserter_time = @timed begin
    state = copy(state)
    center_vert = only(setdiff(support(region), [ortho_vert]))
    e = edgetype(state)(ortho_vert, center_vert)

    Π = local_state.C
    Z1,Z2 = local_state.Z1, local_state.Z2
    ortho_Z,center_Z = local_state.vertex1==ortho_vert ? (Z1,Z2) : (Z2,Z1)
    col_inds = [siteinds(state,ortho_vert)...,commoninds(Π,ortho_Z)...]

    #TODO: try to include tags(state,e), but resulting in extra quotation marks?
    ltags = "Link"
    idtime = @timed begin
      C, Z, _ = interpolative_V2(
        Π, col_inds; col_vertex=ortho_vert, tags=ltags, maxdim, mindim, cutoff
      )
    end
    insert_time = @timed begin
      state[ortho_vert] = ortho_Z*Z
      state[center_vert] = center_Z*C
      state = set_ortho_region(state, [center_vert])
    end
  end
  return state,
  (; idtime=idtime.time, insert_time=insert_time.time, inserter_time=inserter_time.time)
end

function interpolate_sweep_printer(; outputlevel, state, which_sweep, sweep_time, kwargs...)
  outputlevel >= 1 || return nothing
  println("After sweep $which_sweep :")
  println(" maxlinkdim= $(ITensorNetworks.maxlinkdim(state))")
  println(" cpu_time= $(round(sweep_time; digits=3))")
  return flush(stdout)
end

function interpolate_region_printer(;
  outputlevel,
  state,
  sweep_plan,
  spec,
  which_region_update,
  which_sweep,
  inf_norm_error,
  kwargs...,
)
  outputlevel >= 2 || return nothing
  region = first(sweep_plan[which_region_update])
  println("Sweep $which_sweep, region=$(region), max error= $inf_norm_error")
  return flush(stdout)
end

random_initial_pivot(s::AbstractIndexMap) = [only(siteinds(s)[v]) => rand(1:dim(siteinds(s)[v])) for v in vertices(s)]

function random_initial_pivot(tn::AbstractTensorNetwork)
  s = siteinds(tn)
  return random_initial_pivot(s)
end

function interpolate(f, s::AbstractIndexMap, g; initial_pivot=random_initial_pivot(s), initial_state = const_tnf(g, s; bond_dimension = 1), kws...)
  input_f = input -> f(calculate_p_pairs(s, input))
  @assert is_tree(g)
  ttn, info = interpolate(input_f, ITensorNetworks.ttn(initial_state); initial_pivot, kws...)
  tn = TensorNetworkQuantumSimulator.TensorNetwork(ttn)
  return TensorNetworkFunction(tn, s), info
end

function interpolate(
  f,
  init_tn::TreeTensorNetwork;
  cutoff=0.0,
  extracter=interpolate_extracter,
  initial_pivot=random_initial_pivot(init_tn),
  inserter=interpolate_inserter,
  maxdim=typemax(Int),
  mindim=1,
  nsweeps,
  nsites=2,
  region_printer=interpolate_region_printer,
  sweep_printer=interpolate_sweep_printer,
  updater=interpolate_updater,
  use_caching=true,
  kw...,
)
  root_vertex = first(leaf_vertices(init_tn))
  init_tn = interpolative_gauge(init_tn, root_vertex)

  region(; which_region_update, sweep_plan, kw...) = first(sweep_plan[which_region_update])
  sweep(; sweep_plan, which_sweep, kw...) = which_sweep
  error(; inf_norm_error, kw...) = inf_norm_error
  region_observer! = observer(region, error, sweep)

  ttnf = NetworkFunction(f, initial_pivot; use_caching)
  return alternating_update(
    ttnf,
    init_tn;
    region_observer!, 
    cutoff,
    extracter,
    inserter,
    maxdim,
    mindim,
    nsweeps,
    nsites,
    region_printer,
    sweep_printer,
    updater,
    kw...,
  ), region_observer!
end