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