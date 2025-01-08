

"""    
    getAcont_mp(ωdisc::Vector{Float64}, Adisc::Array{Float64}, sigmak::Vector{Float64}, γ::Float64; kwargs...)

Broaden multipoint partial spectral function (PSF) data with log-Gaussian-type and linear broadening. 

# Arguments
1. ωdisc   ::Vector{Float64} :      Logarithimic frequency bins. 
                                    Here the original frequency values from the differences
                                    b/w energy eigenvalues are shifted to the closest bins.
2. Adisc   ::Array{Float64}  :      Spectral function in layout |ωdisc|x|sigmak|.
3. sigmak  ::Vector{Float64} :      Sensitivity of logarithmic position of spectral
                                    weight to z-shift, i.e. |d[log(|ω|)]/dz|. These values will
                                    be used to broaden discrete data. (σ _{ij} or σ_k in
                                    Lee2016.)
4. γ       ::Float64         :      Parameter for secondary linear broadening kernel. (γ in Lee2016.)

# Keyword arguments 
This function accepts the same options as for "getAcont". A difference
from the standard usage of "getAcont" is that the default values of
'emin', 'emax', and 'estep' are set to 1e-6, 10, and 16, respectively.
This is necessary since this function broadens two- or higher dimensional
data; with the default values of "getAcont", your memory might blow up.

# Returns
1. ωcont   ::Vector{Float64}:    continous frequencies
2. Acont   ::Vector{Float64}:    continous spectral data


# Output
1. ocont ::Vector{Float64} :    One-dimensional frequency grid for "Acont", used to
                                describe the frequency space for "Acont" along all dimensions.
2. Acont ::Vector{Float64} :    Multi-dimensional array that represent the broadened spectral data.
"""
function getAcont_mp(
    ωdisc   ::Vector{Float64},  
    Adisc   ::Array{Float64},   
    sigmak  ::Vector{Float64},
    γ       ::Float64           
    ; 
    kwargs...                   # keyword arguments to be passed to 1D broadening function
    )

    D, _, Adisc, Kernels, ωcont = _prepare_broadening_mp(ωdisc, Adisc, sigmak, γ; kwargs...)
    Acont = contract_1D_Kernels_w_Adisc_mp(Kernels, Adisc)

    return ωcont, Acont
end

"""
Logarithmic grid with estep points
"""
# function Acont_grid(ωdisc::VectorArray{Float64}, sigmak::Vector{Float64}, γ::Float64)
function get_Acont_grid(;estep=_ESTEP_DEFAULT(), emin=1.e-12, emax=1.e4)
    # lemin = floor(log10(emin))
    # lemax = ceil(log10(emax))
    # xs = LinRange(lemin, lemax, estep)
    xs = (Int(floor(log10(emin)*estep)):Int(ceil(log10(emax)*estep))) ./ estep
    ocont_p = 10.0 .^ xs
    return vcat(-reverse(ocont_p), [0.0], ocont_p)
end

"""
Compute broadened Keldysh kernel on given 1D frequency grid.
"""
function compute_broadened_kernel(
    ωdisc::Vector{Float64},
    sigmak::Array{Float64},
    γ::Float64;
    ωs_ext::Vector{Float64},
    broadening_kwargs...
)
    
    # broadening kernels
    ωcont_in = get_Acont_grid(;broadening_kwargs...)
    ωcont, Kernel = getAcont(
        ωdisc,
        Matrix{Float64}(LinearAlgebra.I, (ones(Int, 2).*length(ωdisc))...),
        sigmak .+ zeros(length(ωdisc)),
        γ;
        ωcont=ωcont_in,
        broadening_kwargs...
        )

    # hilbert transform
    @show size(Kernel)
    @show extrema(ωs_ext)
    @show extrema(ωcont)
    kernel = -im * π * my_hilbert_trafo(ωs_ext, ωcont, Kernel)
    return kernel
end

function _prepare_broadening_mp(
    ωdisc   ::Vector{Float64},  # Logarithimic frequency bins. 
                                # Here the original frequency values from the differences
                                # b/w energy eigenvalues are shifted to the closest bins.
                                # 
    Adisc   ::Array{Float64},   # Spectral function in layout |ωdisc|x|sigmak|.
    sigmak  ::Vector{Float64},  # Sensitivity of logarithmic position of spectral
                                # weight to z-shift, i.e. |d[log(|omega|)]/dz|. These values will
                                # be used to broaden discrete data. (\sigma_{ij} or \sigma_k in
                                # Lee2016.)
    γ       ::Float64           # Parameter for secondary linear broadening kernel. (\gamma
                                # in Lee2016.)
    ; 
    ωconts  ::NTuple{D,Vector{Float64}},
    kwargs...
) where{D}
    # Check if sigmak is a single number
    if length(sigmak) != 1
        error("ERR: Logarithmic broadening width parameter 'sigmak' should be a single number.")
    end
    # remove singleton dimensions
    Adisc = dropdims(Adisc,dims=tuple(findall(size(Adisc).==1)...))

    # dimensions that need to be broadened
    if !(1 <= D <=3)
        throw(ArgumentError("Only 1-/2-/3-dimensional Adisc supported."))
    end
    ωdiscs = ntuple(x -> ωdisc, D)
    #sz = [prod(size(x)) for x in ωdiscs]
    ωcont_lengths = length.(ωconts)
    ωcont_largest = ωconts[argmax(ωcont_lengths)]
    # Broadening kernels, obtained by creating Acont with identity matrix Adisc
    ωcont, Kernel = getAcont(ωdisc, Matrix{Float64}(LinearAlgebra.I, (ones(Int, 2).*length(ωdisc))...), sigmak .+ zeros(length(ωdisc)), γ; ωcont=ωcont_largest, kwargs...)
    
    # Delete rows/columns that contain only zeros
    AdiscIsZero_oks, ωdiscs, Adisc = compactAdisc(ωdisc, Adisc)
    # AdiscIsZero_oks, ωdiscs, Adisc = compactAdisc(ωdisc, ones(Float64, size(Adisc)))
    ishifts = ntuple(i -> div(length(ωcont_largest) - length(ωconts[i]), 2), D)
    Kernels = [Kernel[1+ishifts[i]:end-ishifts[i], .!AdiscIsZero_oks[i]] for i in 1:D]
    return D, ωdiscs, Adisc, Kernels, ωcont
end


"""
    BroadenedPSF(ωdisc::Vector{Float64}, Adisc::Array{Float64,D}, sigmak::Vector{Float64}, γ::Float64; kwargs...)

Stores a broadened D-dimensional PSF in form of a TuckerDecomposition{Float64,D}. \n
Kernels are precomputed, such that e.g. for D=2 we get Acontᵢⱼ = ∑ᵦᵧ Kᵢᵦ Kⱼᵧ Adiscᵦᵧ. \n
A BroadenedPSF can be evaluated at indices 1:length(ωconts[d]) with 1≤d≤D.
```
"""
function BroadenedPSF(
    ωdisc   ::Vector{Float64},  # Logarithimic frequency bins. 
                                # Here the original frequency values from the differences
                                # b/w energy eigenvalues are shifted to the closest bins.
                                # 
    Adisc   ::Array{Float64,D}, # Spectral function in layout |ωdisc|x|sigmak|.
    sigmak  ::Vector{Float64},  # Sensitivity of logarithmic position of spectral
                                # weight to z-shift, i.e. |d[log(|omega|)]/dz|. These values will
                                # be used to broaden discrete data. (\sigma_{ij} or \sigma_k in
                                # Lee2016.)
    γ       ::Float64           # Parameter for secondary linear broadening kernel. (\gamma
                                # in Lee2016.)
    ; 
    ωconts  ::NTuple{D,Vector{Float64}},
    kwargs...
) where{D}
    @TIME _, ωdiscs, Adisc, Kernels, _ = _prepare_broadening_mp(ωdisc, Adisc, sigmak, γ; ωconts, kwargs...) "Prepare broadening."
    # omdisc = ωdiscs[1]
    # println("KKKKKKKKKKKKKKKKKKKKKK")
    # zomid = findfirst(om -> abs(om)<1.e-12, omdisc)
    # zomcontid = findfirst(om -> abs(om)<1.e-12, ωconts[1])
    # @show size.(Kernels)
    # display(Kernels[1][zomcontid-5:zomcontid+5,zomid])
    # plot(ωconts[1], Kernels[1][:,zomid])
    # savefig("kernel.pdf")
    # println("KKKKKKKKKKKKKKKKKKKKKK")
    return TuckerDecomposition(Adisc, Kernels; ωs_center=ωdiscs, ωs_legs=[ωconts...])
end

"""
Broadening as in symmetric estimators paper (Lihm et al. 2024), appendix E2, Sec. 2
"""
function default_broadening_γσ(T::Float64)
    # γ, σ_k
    return (0.5*T, [0.3])
end