module ITensorNumericalAnalysis

function __init__()
  include(joinpath(pkgdir(ITensorNumericalAnalysis), "src", "fixes.jl"))
  return nothing
end

include("utils.jl")
include("digit_inds.jl")
include("IndexMaps/abstractindexmap.jl")
include("IndexMaps/realindexmap.jl")
include("IndexMaps/complexindexmap.jl")
include("indsnetworkmap.jl")
include("polynomialutils.jl")
include("itensornetworkfunction.jl")
include("elementary_functions.jl")
include("elementary_operators.jl")
include("integration.jl")

export continuous_siteinds
export ITensorNetworkFunction, itensornetwork, dimension_vertices
export AbstractIndexMap,
  RealIndexMap,
  ComplexIndexMap,
  default_dimension_vertices,
  dimension_inds,
  calculate_p,
  calculate_ind_values,
  dimension,
  dimensions,
  grid_points
export IndsNetworkMap,
  continuous_siteinds,
  complex_continuous_siteinds,
  real_continuous_siteinds,
  indsnetwork,
  indexmap,
  indexmaptype,
  vertex_dimension,
  vertex_digit,
  vertices_dimensions,
  vertices_digits
export const_itensornetwork,
  exp_itensornetwork,
  cosh_itensornetwork,
  sinh_itensornetwork,
  tanh_itensornetwork,
  cos_itensornetwork,
  sin_itensornetwork,
  get_edge_toward_root,
  polynomial_itensornetwork,
  random_itensornetwork,
  laplacian_operator,
  first_derivative_operator,
  second_derivative_operator,
  third_derivative_operator,
  fourth_derivative_operator,
  identity_operator,
  delta_p,
  map_to_zero_operator,
  map_to_zeros,
  const_plane_op
export const_itn,
  poly_itn, cosh_itn, sinh_itn, tanh_itn, exp_itn, sin_itn, cos_itn, rand_itn
export evaluate
export operate, operator_proj, multiply

end
