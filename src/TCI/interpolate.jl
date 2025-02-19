using Graphs: AbstractEdge, is_tree, nv
using ITensors: array, commonind, factorize, hastags, pause, permute, uniqueinds
using ITensorNetworks:
  ITensorNetworks,
  AbstractTTN,
  IndsNetwork,
  ITensorNetwork,
  TreeTensorNetwork,
  alternating_update,
  commonind,
  edgetype,
  maxlinkdim,
  orthogonalize,
  set_ortho_region,
  siteinds,
  support,
  tags,
  default_transform_operator,
  default_sweep_plans
using NamedGraphs.GraphsExtensions: is_leaf_vertex, incident_edges

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
  site_inds = reduce(vcat, [siteinds(ttn, v) for v in region])
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
  ortho_vert = only(ortho_region(state))
  center_vert = only(setdiff(support(region), [ortho_vert]))
  e = edgetype(state)(ortho_vert, center_vert)
  col_inds = uniqueinds(state[ortho_vert], state[center_vert])
  site_inds = vcat(siteinds(state, ortho_vert), siteinds(state, center_vert))
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

function interpolate_sweep_printer(; outputlevel, state, which_sweep, sweep_time, kwargs...)
  outputlevel >= 1 || return nothing
  println("After sweep $which_sweep :")
  println(" maxlinkdim= $(maxlinkdim(state))")
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

random_initial_pivot(s::IndsNetworkMap) = [ind => rand(1:dim(ind)) for ind in inds(s)]

function random_initial_pivot(tn::AbstractITensorNetwork)
  s = siteinds(tn)
  return random_initial_pivot(s)
end

function interpolate(f, s::IndsNetworkMap; initial_pivot=random_initial_pivot(s), kws...)
  input_f = input -> f(calculate_p(s, input))
  @assert is_tree(s)
  tn = const_itn(s; linkdim=1)
  tn = interpolate(input_f, ttn(itensornetwork(tn)); initial_pivot, kws...)
  return ITensorNetworkFunction(ITensorNetwork(tn), s)
end

function interpolate(
  f,
  init_tn::TreeTensorNetwork;
  cutoff=0.0,
  extracter=interpolate_extracter,
  initial_pivot=random_initial_pivot(init_tn),
  inserter=interpolate_inserter,
  maxdim=typemax(Int),
  mindim=0,
  nsweeps,
  nsites=2,
  region_printer=interpolate_region_printer,
  sweep_printer=interpolate_sweep_printer,
  updater=interpolate_updater,
  use_caching=true,
  extracter_kwargs=(;),
  updater_kwargs=(;),
  transform_operator_kwargs=(;),
  transform_operator=default_transform_operator(),
  kws...,
)
  root_vertex = first(leaf_vertices(init_tn))
  init_tn = interpolative_gauge(init_tn, root_vertex)

  ttnf = NetworkFunction(f, initial_pivot; use_caching)
  sweep_plans = default_sweep_plans(
    nsweeps,
    init_tn;
    root_vertex,
    extracter,
    extracter_kwargs,
    updater_kwargs,
    inserter_kwargs=(; cutoff, maxdim, mindim),
    transform_operator,
    transform_operator_kwargs,
    updater,
    inserter,
    nsites,
  )
  return alternating_update(
    ttnf, init_tn, sweep_plans; region_printer, sweep_printer, kws...
  )
end
