function calc_error(exact_vals::Vector, approx_vals::Vector)
  @assert length(exact_vals) == length(approx_vals)

  m = minimum(exact_vals) + 1
  exact_vals = copy(exact_vals .+ m)
  approx_vals = copy(approx_vals .+ m)
  eps = 0
  for (i, e) in enumerate(exact_vals)
    eps += abs((e - approx_vals[i]) / e)
  end
  return eps / length(exact_vals)
end
