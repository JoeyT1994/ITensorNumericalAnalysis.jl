include("itensornetworks_elementary_functions.jl")

function calculate_x(xis::Dict, vertex_map::Dict)
  @assert keys(vertex_map) == keys(xis)
  return sum([xis[v] / (2^vertex_map[v]) for v in keys(xis)])
end

function calculate_xis(x::Float64, vertex_map::Dict; print_x=false)
  x_rn = copy(x)
  xis = Dict()
  sorted_vertex_map = sort(vertex_map; byvalue=true)
  for v in keys(sorted_vertex_map)
    if (x_rn >= 1.0 / (2^vertex_map[v]))
      xis[v] = 1
      x_rn -= 1.0 / (2^vertex_map[v])
    else
      xis[v] = 0
    end
  end

  x_bitstring = calculate_x(xis, vertex_map)
  (print_x) && println("Actual value of x is $x but bitstring rep. is $x_bitstring")
  return xis
end

"""Build the order L tensor corresponding to fx(x): x ∈ [0,1]."""
function build_full_rank_tensor(L::Int64, fx::Function)
  inds = [Index(2, "$i") for i in 1:L]
  dims = Tuple([2 for i in 1:L])
  array = zeros(dims)
  for i in 0:(2^(L) - 1)
    xis = digits(i; base=2, pad=L)
    x = sum([xis[i] / (2^i) for i in 1:L])
    array[Tuple(xis + ones(Int64, (L)))...] = fx(x)
  end

  return ITensor(array, inds)
end
