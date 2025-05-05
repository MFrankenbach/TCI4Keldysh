#=
Written by ChatGPT. Could come in useful at some point.
=#

using LinearAlgebra

struct CubicSpline
    x::Vector{Float64}
    a::Vector{Float64}
    b::Vector{Float64}
    c::Vector{Float64}
    d::Vector{Float64}

    function CubicSpline(x::Vector{Float64}, y::Vector{Float64})
        n = length(x)
        @assert n == length(y) "Input vectors must have the same length."
        @assert issorted(x) "x values must be sorted in increasing order."

        h = diff(x)
        α = zeros(n)
        for i in 2:n-1
            α[i] = 3 * (y[i+1] - y[i]) / h[i] - 3 * (y[i] - y[i-1]) / h[i-1]
        end

        l = ones(n)
        μ = zeros(n)
        z = zeros(n)

        for i in 2:n-1
            l[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * μ[i-1]
            μ[i] = h[i] / l[i]
            z[i] = (α[i] - h[i-1] * z[i-1]) / l[i]
        end

        c = zeros(n)
        b = zeros(n-1)
        d = zeros(n-1)

        for j in n-1:-1:1
            c[j] = z[j] - μ[j] * c[j+1]
            b[j] = (y[j+1] - y[j]) / h[j] - h[j] * (c[j+1] + 2*c[j]) / 3
            d[j] = (c[j+1] - c[j]) / (3 * h[j])
        end

        return new(x, y[1:end-1], b, c[1:end-1], d)
    end
end

function (spline::CubicSpline)(xq::Float64)
    if abs(xq - spline.x[1])<1.e-12
        return spline.a[1]
    end
    idx = if abs(xq - spline.x[end])<1.e-12
        lastindex(spline.x)-1
    else
        idx = searchsortedlast(spline.x, xq)
        if idx == 0 || idx >= length(spline.x)
            throw(ArgumentError("Query point $(xq) is out of bounds."))
        end
        idx
    end
    
    h = xq - spline.x[idx]
    return spline.a[idx] + spline.b[idx] * h + spline.c[idx] * h^2 + spline.d[idx] * h^3
end

function check_spline()
    x = [1.0, 1.1, 1.5, 2.0] 
    y = [1.3, -1.0, 0.0, 2.0] 
    sp = CubicSpline(x,y)
    return sp
end
