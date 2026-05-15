module ITensorNumericalAnalysis


include("imports.jl")
include("utils.jl")
include("IndexMaps/digit_inds.jl")
include("IndexMaps/abstractindexmap.jl")
include("IndexMaps/realindexmap.jl")
include("IndexMaps/complexindexmap.jl")
include("polynomialutils.jl")
include("tensornetworkfunction.jl")
include("elementary_functions.jl")
include("elementary_operators.jl")
include("integration.jl")
include("tci/interpolate.jl")
include("tci/interpolative.jl")
include("tci/interpolative_gauge.jl")
include("tci/lu.jl")
include("tci/networkfunction.jl")
include("tci/pivot_index.jl")

export continuous_siteinds, complex_continuous_siteinds
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
    sin_tnf,
    tanh_tnf,
    poly_tnf,
    delta_p,
    integrate,
    partial_integrate,
    operator_proj,
    forward_shift_op,
    backward_shift_op,
    first_derivative_operator,
    second_derivative_operator,
    third_derivative_operator,
    fourth_derivative_operator,
    identity_operator,
    map_to_zero_operator,
    map_to_zeros,
    const_plane_op,
    multiply,
    operate,
    delta_kernel,
    reduced_indexmap

end
