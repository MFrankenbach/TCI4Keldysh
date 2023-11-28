
"""
PartialCorrelator_reg

partial correlator ̃Gₚ(ω₁, ω₂, ...) = ∫dε₁dε₂ ̃K(ω₁-ε₁) ̃K( ω₂- ε₂)... Sₚ(ε₁,ε₂,...)
"""
struct PartialCorrelator_reg{D}
    Adisc   ::  Array{Float64,D}                # discrete PSF data; best: compactified with compactAdisc(...)
    Kernels ::  NTuple{D,Matrix{ComplexF64}}    # regular kernels
    ωs      ::  NTuple{D,Vector{ComplexF64}}    # external complex frequencies
    isBos   ::  NTuple{D,Bool}                  # indicates whether ω_i/ε_i is a bosonic frequency


    ####################
    ### Constructors ###
    ####################
    function PartialCorrelator_reg(Adisc::Array{Float64,D}, ωdisc::Vector{Float64}, isBos::NTuple{D,Bool}, ωs::Vararg{Vector{ComplexF64},D}) where {D}

        # Delete rows/columns that contain only zeros
        AdiscIsZero_oks, ωdiscs, Adisc = compactAdisc(ωdisc, Adisc)
        # Then pray that Adisc has no contributions for which the kernels diverge:
        Kernels = ntuple(i -> get_regular_1DKernel(ωs[i], ωdiscs[i]) ,D)
        return new{D}(Adisc, Kernels, (ωs..., ), isBos)
    end
    function PartialCorrelator_reg(Acont::BroadenedPSF{D}, isBos::NTuple{D,Bool}, ωs::Vararg{Vector{ComplexF64},D}) where {D}
        δωcont = get_ω_binwidths(Acont.ωcont)
        # 1.: rediscretization of broadening kernel
        # 2.: contraction with regular kernel
        Kernels = ntuple(i -> get_regular_1DKernel(ωs[i], Acont.ωcont) * (δωcont .* Acont.Kernels[i]), D)
        return new{D}(Acont.Adisc, Kernels, (ωs..., ), isBos)
    end

end


function get_regular_1DKernel(ωs::Vector{ComplexF64}, ωdisc::Vector{Float64})

    println("is there a zero in the ωdisc?: ", any(ωdisc.≈0.))
    Kernel = 1. ./ (ωs .- ωdisc')

    is_divergent = .!isfinite.(Kernel)
    (@view Kernel[is_divergent]) .= 0. # This corresponds the result with a Adisc(0) δ(ω) broadened to --> Adisc(0)/Δx0 1_[x[-1]/2,x[1]/2]  where 1_[a,b](x) is the rectangular function which gives one if x∈[a,b]
    @assert any(isfinite.(Kernel))

    return Kernel
end