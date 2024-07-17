using Random: randn
using SpecialFunctions

function bessel_coefficients(nterms::Int64, a, k=1.0)
  coeffs = [0.0 for i in 1:a]
  c = (1.0 / factorial(a)) * ((0.5 * k)^a)
  append!(coeffs, c)
  for i in 1:(nterms - 1)
    c *= (-1.0 / (i * (a + i))) * (0.5 * 0.5 * k * k)
    append!(coeffs, 0.0)
    append!(coeffs, c)
  end
  return coeffs
end

function laguerre_coefficients(nterms::Int64)
  c = 1.0
  coeffs = []
  append!(coeffs, c)
  for i in 1:(nterms - 1)
    c *= -1 * (nterms - i + 1) / (i*i)
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

function weirstrass_coefficients(nterms::Int64, a)
    return [Float64(pi) * i^a for i in 1:nterms]
  end

function weirstrass_itn(s::IndsNetworkMap, ks::Vector)
  ψ = sin_itn(s; k=first(ks), c=1.0 / first(ks))
  for i in 2:length(ks)
    ψ = ψ + sin_itn(s; k=ks[i], c= 1.0 /ks[i])
  end
  return ψ
end

function calulate_weirstrass(x::Float64, ks::Vector)
  out = 0
  for i in 1:length(ks)
    out += (1.0 /ks[i]) * sin(ks[i] * x)
  end
  return out
end

function build_random_planewaves(s::IndsNetworkMap, nterms, As, kxs, kys, kzs; dimension = 1)
  itns = [exp_itn(s; k = 1.0im * kxs[i], c = As[i], dimension = 1)*exp_itn(s; k = 1.0im * kys[i], dimension = 2)*exp_itn(s; k = 1.0im * kzs[i], dimension = 3) for i in 1:nterms]
  return reduce(+, reduce(vcat,itns))
end

function build_spherical_laplacian_solution(s::IndsNetworkMap, coeffs)
  r, r_sq, r_cub = poly_itn(s, [0.0, 1.0]; dimension = 1), poly_itn(s, [0.0, 0.0, 1.0]; dimension = 1), poly_itn(s, [0.0, 0.0, 0.0, 1.0]; dimension = 1)
  c_theta, s_theta = cos_itn(s; k = pi, dimension = 2), sin_itn(s; k = pi, dimension = 2)
  c_phi, s_phi, c_2phi = cos_itn(s; k =2.0*pi, dimension = 3), sin_itn(s; k = 2.0*pi, dimension = 3), cos_itn(s; k =4.0*pi, dimension = 3)
  ψ = r * s_theta * s_phi * const_itn(s; c = coeffs[1])
  ψ += r_cub * s_theta * s_phi * const_itn(s; c = coeffs[2])
  ψ += r_sq * s_theta * s_phi * c_theta * const_itn(s; c = coeffs[3])
  ψ += r_sq * s_theta * s_phi * c_phi * s_theta * const_itn(s; c = coeffs[4])
  ψ += r_cub * s_theta * s_phi * c_phi * s_theta * c_theta * const_itn(s; c = coeffs[5])
  ψ += r_cub * s_theta * s_phi * s_theta * s_theta *  c_2phi * const_itn(s; c = coeffs[6])
  return ψ
end

function calculate_spherical_laplacian_solution(x, y, z, coeffs)
  r, r_sq, r_cub = x, x*x, x*x*x 
  c_theta, s_theta = cos(pi*y), sin(pi*y)
  c_phi, s_phi, c_2phi = cos(2*pi*z), sin(2*pi*z), cos(4*pi*z)
  out = r * s_theta * s_phi * coeffs[1]
  out += r_cub * s_theta * s_phi * coeffs[2]
  out += r_sq * s_theta * s_phi * c_theta * coeffs[3]
  out += r_sq * s_theta * s_phi * c_phi * s_theta * coeffs[4]
  out += r_cub * s_theta * s_phi * c_phi * s_theta * c_theta * coeffs[5]
  out += r_cub * s_theta * s_phi * s_theta * s_theta *  c_2phi * coeffs[6]
  return out
end

function in_mandelbrot(c; maxiter = 10000)
  zi = 0
  for i in 1:maxiter
    zi = zi*zi + c
    if abs(zi) > 2
      return i / maxiter
    end
  end
  return 1.0
end