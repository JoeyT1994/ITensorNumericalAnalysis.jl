module ITensorNumericalAnalysis

#function __init__()
#  include(joinpath(@__DIR__, "fixes.jl"))
#  return nothing
#end

include("utils.jl")
include("IndexMaps/digit_inds.jl")
include("IndexMaps/abstractindexmap.jl")
include("IndexMaps/realindexmap.jl")
include("IndexMaps/complexindexmap.jl")
include("polynomialutils.jl")
include("tensornetworkfunction.jl")
include("elementary_functions.jl")
# include("elementary_operators.jl")
# include("integration.jl")

export continuous_siteinds, complex_continuous_siteinds
# export tensorNetworkFunction, tensornetwork, dimension_vertices
export AbstractIndexMap,
  RealIndexMap,
  ComplexIndexMap,
  default_dimension_vertices,
  dimension_inds,
  calculate_p,
  calculate_ind_values,
  dimension,
  dimensions,
  grid_points,
  indexmap,
  dimension_vertices,
  vertex_dimension,
  vertex_digit,
  vertices_dimensions,
  vertices_digits
export TensorNetworkFunction,
  evaluate,
  const_tnf,
  exp_tnf,
  cosh_tnf,
  sinh_tnf,
  cos_tnf,
  sin_tnf
# export IndsNetworkMap,
#   continuous_siteinds,
#   complex_continuous_siteinds,
#   real_continuous_siteinds,
#   indsnetwork,
#   indexmap,
#   indexmaptype,
#   vertex_dimension,
#   vertex_digit,
#   vertices_dimensions,
#   vertices_digits
# export const_tensornetwork,
#   exp_tensornetwork,
#   cosh_tensornetwork,
#   sinh_tensornetwork,
#   tanh_tensornetwork,
#   cos_tensornetwork,
#   sin_tensornetwork,
#   get_edge_toward_root,
#   polynomial_tensornetwork,
#   random_tensornetwork,
#   laplacian_operator,
#   first_derivative_operator,
#   second_derivative_operator,
#   third_derivative_operator,
#   fourth_derivative_operator,
#   identity_operator,
#   delta_p,
#   map_to_zero_operator,
#   map_to_zeros,
#   const_plane_op
# export const_itn,
#   poly_itn, cosh_itn, sinh_itn, tanh_itn, exp_itn, sin_itn, cos_itn, rand_itn
# export evaluate
# export operate, operator_proj, multiply

end
