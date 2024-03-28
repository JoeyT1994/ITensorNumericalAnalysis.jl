module TensorNetworkFunctionals

using ITensors
using ITensorNetworks
using NamedGraphs
using EllipsisNotation
using Dictionaries
using Graphs

using SplitApplyCombine: group
using ITensorNetworks: delta_network
using NamedGraphs: add_edges, random_bfs_tree, rem_edges

include("bitmaps.jl")
include("itensornetworkfunction.jl")
include("itensornetworks_elementary_functions.jl")
include("itensornetworks_elementary_operators.jl")
include("itensornetworksutils.jl")

export ITensorNetworkFunction
export BitMap,
  default_dimension_map, vertex, calculate_xyz, calculate_x, calculate_bit_values, dimension
export const_itensornetwork,
  exp_itensornetwork,
  cosh_itensornetwork,
  sinh_itensornetwork,
  tanh_itensornetwork,
  cos_itensornetwork,
  sin_itensornetwork,
  get_edge_toward_root,
  polynomial_itensornetwork,
  Laplacian_operator,
  derivative_operator
export const_itn, poly_itn, cosh_itn, sinh_itn, tanh_itn, exp_itn, sin_itn, cos_itn
export calculate_fx, calculate_fxyz
export operate

end
