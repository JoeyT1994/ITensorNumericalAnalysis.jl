include("itensornetworks_functions.jl")

function calculate_x(xis::Dict, vertex_map::Dict; a::Float64 = 1.0)
    @assert keys(vertex_map) == keys(xis)
    return sum([a*xis[v]/(2^vertex_map[v]) for v in keys(xis)])
end

function calculate_xis(x::Float64, vertex_map::Dict; a::Float64 = 1.0, print_x = false)
    x_rn = x/a
    xis = Dict()
    sorted_vertex_map = sort(vertex_map; byvalue = true)
    for v in keys(sorted_vertex_map)
        if(x_rn >= 1.0 / (2^vertex_map[v]))
            xis[v] = 1
            x_rn -= 1.0 / (2^vertex_map[v])
        else
            xis[v] = 0
        end
    end

    x_bitstring = calculate_x(xis, vertex_map; a)
    (print_x) && println("Actual value of x is $x but bitstring rep. is $x_bitstring")
    return xis
end
