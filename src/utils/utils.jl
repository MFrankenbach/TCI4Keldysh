function maxabs(arr::AbstractArray)
    return maximum(abs.(arr))
end

function quadtrapz(xs::Vector, ys::Vector)
    return sum((ys[2:end]+ys[1:end-1]).*(xs[2:end]-xs[1:end-1]))/2
end

"""
Make the discrete spectral data ("Adisc") of a partial spectral function
compact, by removing the rows, columns, or slices of "Adisc" that contain
zeros only.
"""
function compactAdisc(
    ωdisc   ::Vector{Float64}, 
    Adisc   ::Array{Float64}
    )
    # Make the discrete spectral data ("Adisc") of a partial spectral function
    # compact by removing the rows, columns, or slices of "Adisc" that contain
    # zeros only.

    # remove singleton dimensions
    Adisc = dropdims(Adisc,dims=tuple(findall(size(Adisc).==1)...))
    nO = ndims(Adisc)

    # Sanity check
    sz = size(Adisc)
    if !all(sz .== (length(ωdisc) * ones(Int, nO)))
        error("ERR: size of Adisc is inconsistent with the size of odisc.")
    end

    oks = [BitVector(undef, length(ωdisc)) for i in 1:nO]
    AdiscIsZero = abs.(Adisc) .< (maxabs(Adisc) * 1.e-14)
    for i in 1:nO
        # find slices that contain only zeros:
        all!(reshape(oks[i], (ones(Int, i-1)..., length(ωdisc))), AdiscIsZero)
    end
    Adisc = Adisc[[.!oks[i] for i in 1:nO]...]
    ωdiscs = [deepcopy(ωdisc[.!oks[i]]) for i in 1:nO]

    return oks, ωdiscs, Adisc
end

function get_ω_binwidths(ωs::Vector{Float64}) ::Vector{Float64}
    Δωs = [ωs[2] - ωs[1]; (ωs[3:end] - ωs[1:end-2]) / 2; ωs[end] - ωs[end-1]]  # width of the frequency bin
    return Δωs 
end



"""
    hilbert(x)
    hilbert(x, n)
Analytic signal, computed using the Hilbert transform.
`y = hilbert(x)` gives the analytic signal `y = x + i*xi` where `xi` is the
Hilbert transform of vector `x`. 
If `x` is a matrix, then `hilbert` operates along columns.

**Example**
```julia
x = randn(10,10)
y = hilbert(x)
```
"""
function hilbert_fft(x::AbstractArray{Float64,d}; dims::Int=1) where {d}
    if d > 2
        throw(ArgumentError("Currently only Vector and Matrix are supported for signal array ys."))
    end

    x_ = copy(x)

    # work along columns
    #size(x_,1)==1 ? x_ = permutedims(x_,[2 1]) : nothing

    #if(n>0 && n<size(x_,1))
    #    x_ = x_[1:n,:]
    #elseif(n>0 && n>size(x_,1))
    #    x_ = cat(1,x_,zeros(n-size(x_,1),size(x_,2)))
    #else
        n = size(x_,dims)
    #end

    xf = fft(x_,dims)
    h = reshape(zeros(Int64,n), (ones(Int, dims-1)..., n))      # represents the step function in time --> product in time corresponds to convolution with retarded kernel in frequency space
    if n>0 && n % 2 == 0
        #even, nonempty
        h[1:div(n,2)+1] .= 1
        h[2:div(n,2)] .= 2
    elseif n>0
        #odd, nonempty
        h[1] = 1
        h[2:div(n + 1,2)] .= 2
    end
    x_ = ifft(xf .* h, dims)

    # restore to original shape if necessary
    #size(x,1)==1 ? x_ = permutedims(x_,[2 1]) : nothing

    return x_

end



"""
my_hilbert_trafo

Computes the Hilbert transform of signal ys (sampled on a grid xs_in) for output frequencies xs_out.
We assume that the signal ys is piecewise linear.
If ys is multidimensional, the transform is computed along the direction dims=1.

The Hilbert transform is:
h(x_out) = P.V. ∫ dx y(x) / (x_out - x)
where P.V. stands for a principal value integral.

NOTE ON CONVENTIONS:    Unlike the standard definition of Hilbert transforms we don't divide the result 
                        by a global factor of π.

NOTE ON PERFORMANCE / CONVERGENCE:
 - bottlenecks: construction of L (uncomment @time to see details)
 - convergence: for slowly decaying ys ∝ 1/x one needs a huge frequency range for xs_in
                analyze convergence in frequency range and mesh density with maxdev_in_hilbertTrafo_sinc(...) and maxdev_in_hilbertTrafo_rat(...)
                for high-frequency tails in spectrum: try out linear least squares fit to obtain coeffs for 1/ω and 1/ω²
"""
function my_hilbert_trafo(
    xs_out::Vector{Float64},    # real output frequencies xs_out[o]
    xs_in::Vector{Float64},     # real  input frequencies xs_in[i]
    ys::Array{Float64,d}        # piecewisely linear signal ys[i,j] where dim=1 is sampled on frequencies xs_in
                                # and j represents the trailing dimensions in ys
                                # assume 1/x-decay outside of the box
    ) where {d}
    # As y(x) is piecewise linear we compute h(xs_out[o]) piecewisely, giving h(xs_out[o]) = ∑_i H[o,i,j] with
    #       H[o,i,j] = \int_{x[i]}^{x[i+1]} dx y⃗(x) / (xs_out[o] - x) 
    # On the i-th segment we have y⃗(x) = a[i,j] * (x - x[i]) + y[i,j]
    #       with a[i,j] = (y[i+1,j] - y[i,j]) / (x[i+1] - x[i])
    # such that the integral contribution from the i-th segment gives 
    # H[o,i,j] = - (x[i+1] - x[i]) * a[i,j] - ln( |x_out[o] - x[i+1]| / |x_out[o] - x[i]| ) * (y[i,j] + (xs_out[o] - x[i]) a[i,j])
    #          = - Δys[i,j]                .- ΔL[o,i] (y[i,j] + (xs_out[o] - x[i]) a[i,j])
    # with  Δxs_in[i] = xs_in[i+1] - xs_in[i]
    #       L[o,i]  = ln|xs_out[o] - xs_in[i]|
    #       ΔL[o,i] = L[o,i+1] - L[o,i]
    #
    # EDGE CASES:
    #   x_out == x[i]   -->     Then lim_{ϵ→0} ϵ ln(ϵ) = 0  and the divergent terms form segment i and i-1 cancel exactly!

    if d > 2
        throw(ArgumentError("Currently only Vector and Matrix are supported for signal array ys."))
    end

    #print("Time for construction of M: \t")
    #@time M = xs_out .- xs_in'
    M = xs_out .- xs_in'
    #print("Time for construction of L: \t")
    #@time L = log.(abs.(M))
    L = log.(abs.(M))
    ## clean L from terms where xs_out[i] == xs_in[j]
    is_divergentL = .!isfinite.(L)
    (@view L[is_divergentL]) .= 0.

    #print("Time for construction of ΔL: \t")
    #@time ΔL= diff(L, dims=2)                 
    ΔL= diff(L, dims=2)                 


    #print("Time for construction of Δy: \t")
    #@time Δys = diff(ys, dims=1)              
    #Δxs_in = diff(xs_in)                
    Δys = diff(ys, dims=1)              
    Δxs_in = diff(xs_in)                

    #print("Time for construction of a: \t")
    #@time a = Δys ./ Δxs_in
    a = Δys ./ Δxs_in

    #print("Time for construction of term1: \t")
    #@time term1 = - (ys[1,:] - ys[end,:])
    #print("Time for construction of term2: \t")
    #@time term2 = ΔL * ys[1:end-1,:]
    #print("Time for construction of term3: \t")
    #@time term3 = (ΔL .* (@view M[:,1:end-1])) * a
    term1 = - (ys[1,:] - ys[end,:])
    term2 = ΔL * ys[1:end-1,:]
    term3 = (ΔL .* (@view M[:,1:end-1])) * a

    #println("shapes of terms:\t", size(term1), "\t", size(term2), "\t", size(term3))

    result = term1' .- term2 .- term3
    #result = - (ys[1,:] - ys[end,:]) .- ΔL * ys[1:end-1,:] - (ΔL .* (xs_out .- xs_in[1:end-1]')) * a

    return result
end


function load_Adisc(path::String, Ops::Vector{String}, flavor_idx::Int)
    f = matopen(joinpath(path, "PSF_(("*mapreduce(*,*,Ops, ["," for i in 1:length(Ops)])[1:end-1]*")).mat"), "r")
    try 
        keys(f)
    catch
        keys(f)
    end
    Adisc = read(f, "Adisc")[flavor_idx]
    Adisc = dropdims(Adisc,dims=tuple(findall(size(Adisc).==1)...))

    return Adisc 
end


function load_ωdisc(path::String, Ops::Vector{String})
    f = matopen(joinpath(path, "PSF_(("*mapreduce(*,*,Ops, ["," for i in 1:length(Ops)])[1:end-1]*")).mat"), "r")
    try 
        keys(f)
    catch
        keys(f)
    end
    ωdisc  = read(f, "odisc")[:]
    return ωdisc 
end