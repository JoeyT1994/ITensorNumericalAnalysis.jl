using Random: randn

function bessel_coefficients(alpha::Int64, nterms::Int64; k=1.0)
  coeffs = [0.0 for i in 1:alpha]
  c = (1.0 / factorial(alpha)) * ((0.5 * k)^alpha)
  append!(coeffs, c)
  for i in 1:(nterms - 1)
    c *= (-1.0 / (i * (alpha + i))) * (0.5 * 0.5 * k * k)
    append!(coeffs, 0.0)
    append!(coeffs, c)
  end
  return coeffs
end

function random_polynomial_coeffs(nterms::Int64; k=1.0, W::Float64=1.0)
  coeffs = Float64[]
  k_power = 1.0
  for i in 1:nterms
    c = W * randn() * k_power
    k_power *= k
    append!(coeffs, c)
  end
  return coeffs
end

function evaluate_polynomial(x::Float64, coefficients::Vector)
  out = first(coefficients)
  xpower = x
  for c in coefficients[2:length(coefficients)]
    out += c * xpower
    xpower *= x
  end
  return out
end

function weirstrass_coefficients(nterms::Int64, a::Float64, b::Float64)
  c, k = 1.0, Float64(pi)
  cs = Float64[c]
  ks = Float64[k]
  for i in 2:nterms
    c *= a
    k *= b
    append!(cs, c)
    append!(ks, k)
  end
  return cs, ks
end

function weirstrass_coefficients_V2(nterms::Int64, a::Float64)
    return [Float64(pi) * i^a for i in 1:nterms]
  end

function weirstrass_itn_V2(s::IndsNetworkMap, ks::Vector)
  ψ = sin_itn(s; k=first(ks), c=1.0 / first(ks))
  for i in 2:length(ks)
    ψ = ψ + sin_itn(s; k=ks[i], c= 1.0 /ks[i])
  end
  return ψ
end

function calulate_weirstrass_V2(x::Float64, ks::Vector)
  out = 0
  for i in 1:length(cs)
    out += (1.0 /ks[i]) * sin(ks[i] * x)
  end
  return out
end
