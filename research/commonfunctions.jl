function bessel_coefficients(alpha::Int64, nterms::Int64; k = 1.0)
    coeffs = [0.0 for i in 1:alpha]
    c = (1.0 /factorial(alpha))*((0.5*k)^alpha)
    append!(coeffs, c)
    for i in 1:(nterms - 1)
        c *= (-1.0 / (i*(alpha + i))) * (0.5*0.5*k*k)
        append!(coeffs, 0.0)
        append!(coeffs, c)
    end
    return coeffs
end