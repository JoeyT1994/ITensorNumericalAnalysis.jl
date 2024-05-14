module ITensorNumericalAnalysis

include("utils.jl")
include("indexmap.jl")
include("indsnetworkmap.jl")
include("polynomialutils.jl")
include("itensornetworkfunction.jl")
include("elementary_functions.jl")
include("elementary_operators.jl")
include("integration.jl")

export continuous_siteinds
export ITensorNetworkFunction, itensornetwork, dimension_vertices
export IndexMap,
  default_dimension_map,
  dimension_inds,
  calculate_xyz,
  calculate_x,
  calculate_ind_values,
  dimension,
  dimensions,
  grid_points
export IndsNetworkMap,
  continuous_siteinds,
  indsnetwork,
  indexmap,
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
  delta_x,
  delta_xyz
export const_itn,
  poly_itn, cosh_itn, sinh_itn, tanh_itn, exp_itn, sin_itn, cos_itn, rand_itn
export calculate_fx, calculate_fxyz
export operate, operator_proj, multiply

end
