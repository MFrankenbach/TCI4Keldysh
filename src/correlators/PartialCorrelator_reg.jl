
"""
PartialCorrelator_reg

Regular partial correlator ̃Gₚ(ω₁, ω₂, ...) = ∫dε₁dε₂ ̃K(ω₁-ε₁) ̃K( ω₂- ε₂)... Sₚ(ε₁,ε₂,...)

Members:
* formalism:: String                          # "MF" or "KF"
* Adisc   ::  Array{Float64,D}                # discrete PSF data; best: compactified with compactAdisc(...)
* ωdiscs  ::  Vector{Vector{Float64}}         # discrete frequencies for 
* Kernels ::  Vector{Matrix{ComplexF64}}      # regular kernels
* ωs_ext  ::  NTuple{D,Vector{ComplexF64}}    # external complex frequencies
* ωs_int  ::  NTuple{D,Vector{ComplexF64}}    # internal complex frequencies
* ωconvMat::  SMatrix{D,D,Int}                # matrix encoding frequency conversion in terms of indices ~ i_ωs_int = ωconvMat * i_ωs_ext + ωconvOff
* ωconvOff::  SVector{D,Int}                  # Offset encoding frequency conversion in terms of indices



Kernel for formalism == "MF":  ̃K(ω₁-ε₁) = 1/(ω₁-ε₁)            [= regular kernel]
                        "KF":  ̃K(ω₁-ε₁) = 1/(ω₁-ε₁ +im 0^+)    [= retarded kernel]
Pointwise evaluation of PartialCorrelator_reg Gp: 
 * without frequency conversion (any formalism) --> Gp[i,j,...]
 * with    frequency conversion 
        for MF --> Gp(i,j,...) returns regular part of partial correlator
        for KF --> evaluate_with_ωconversion_KF(...) returns all useful combinations of retarded/advanced kernels ̃∏_{αᵢ=R/A} K^{αᵢ}

precompute_all_values:
 * "MF": computes values for all external frequencies ωs_ext (and includes anomalous contributions!)
 * "KF": not implemented yet
"""
mutable struct PartialCorrelator_reg{D} <: AbstractTuckerDecomp{D}
    formalism:: String                          # "MF" or "KF"
    tucker      ::  TuckerDecomposition{ComplexF64,D}
    #Adisc   ::  Array{ComplexF64,D}             # discrete PSF data; best: compactified with compactAdisc(...)
    Adisc_anoβ::Array{ComplexF64,D}             # anomalous part of the discrete PSF data (only used for computing the anomalous contribution ∝ β)
    #ωdiscs  ::  Vector{Vector{Float64}}         # discrete frequencies for 
    #Kernels ::  Vector{Matrix{ComplexF64}}      # regular kernels
    ωs_ext  ::  NTuple{D,Vector{Float64}}    # external complex frequencies
    #ωs_int  ::  NTuple{D,Vector{ComplexF64}}    # internal complex frequencies
    ωconvMat::  SMatrix{D,D,Int}                # matrix encoding frequency conversion in terms of indices ~ i_ωs_int = ωconvMat * i_ωs_ext + ωconvOff
    ωconvOff::  SVector{D,Int}                  # Offset encoding frequency conversion in terms of (one-based!) indices

    ####################
    ### Constructors ###
    ####################
    function PartialCorrelator_reg(formalism::String, Adisc::Array{Float64,D}, ωdisc::Vector{Float64}, ωs_ext::NTuple{D,Vector{Float64}}, ωconv::Matrix{Int}; is_compactAdisc::Bool=true) where {D}
        if !(formalism == "MF" || formalism == "KF")
            throw(ArgumentError("formalism must be MF when unbroadened Adisc is input."))
        end
        if DEBUG()
            println("Constructing PartialCorrelator_reg (WITHOUT broadening).")
        end

        ωs_int, ωconvMat, ωconvOff = _trafo_ω_args(ωs_ext, ωconv)
        if is_compactAdisc
        # Delete rows/columns that contain only zeros
        _, ωdiscs, Adisc = compactAdisc(ωdisc, Adisc)
        else
            ωdiscs, Adisc = [ωdisc for _ in 1:D], Adisc
        end
        @TIME Kernels = [get_regular_1D_MF_Kernel(ωs_int[i], ωdiscs[i]) for i in 1:D] "Precomputing 1D kernels (for MF)."
        if formalism == "MF"
            Adisc_anoβ = deepcopy(Adisc)
        end
        tucker = TuckerDecomposition(Adisc, Kernels; ωs_center=ωdiscs, ωs_legs=[ωs_int...])
        return new{D}(formalism, tucker, Adisc_anoβ, ωs_ext, ωconvMat, ωconvOff)
    end
    function PartialCorrelator_reg(formalism::String, Acont::AbstractTuckerDecomp{D}, ωs_ext::NTuple{D,Vector{Float64}}, ωconv::Matrix{Int}) where {D}
        if !(formalism == "MF" || formalism == "KF")
            throw(ArgumentError("formalism must be MF or KF."))
        end
        if DEBUG()
            println("Constructing PartialCorrelator_reg (WITH broadening).")
        end

        ωs_int, ωconvMat, ωconvOff = _trafo_ω_args(ωs_ext, ωconv)
        #println("ωconvMat, ωconvOff: ", ωconvMat, ωconvOff)
        δωcont = get_ω_binwidths(Acont.ωconts[1])
        tucker = deepcopy(Acont)
        if formalism == "MF"
            # 1.: rediscretization of broadening kernel
            # 2.: contraction with regular kernel
            @TIME tucker.legs = [get_regular_1D_MF_Kernel(ωs_int[i], Acont.ωcont) * (get_ω_binwidths(Acont.ωconts[i]) .* Acont.legs[i]) for i in 1:D] "Constructing 1D Kernels (for MF)."
            Adisc_anoβ = deepcopy(Acont.center)
        else
            # check that grid is equidistant:
            if maximum(abs.(diff(δωcont) )) > 1e-10
                throw(ArgumentError("ωcont must be an equidistant grid."))
            end
            # compute retarded 1D kernels
            @TIME tucker.legs = [-im * π * hilbert_fft(Acont.legs[i]; dims=1) for i in 1:D] "Hilbert trafo (for KF)."
            Adisc_anoβ = Array{ComplexF64,D}(undef, ones(Int,D))
        end
        return new{D}(formalism, tucker, Adisc_anoβ, ωs_ext, ωconvMat, ωconvOff)
    end

end


"""
_trafo_ω_args

Compute internal frequencies from external ωs.
Also compute the index transformation matrices:
External indices have the ranges OneTo.(length.(ωs))


Internal indices have the ranges OneTo.(length.(ωs_new))


"""
function _trafo_ω_args(ωs::NTuple{D,Vector{T}}, ωconv::Matrix{Int}) where{D,T}
    function grids_are_fine(grids)
        Δgrids = [diff(g) for g in grids]
        is_equidistant_symmetric = [(grids[i][1] == -grids[i][end]) && (maximum(abs.(diff(Δgrids[i]))) < 1.e-10) for i in eachindex(ωs)]
        all_spacings_identical = D > 1 ? maximum(abs.(diff([Δgrids[i][1] for i in eachindex(ωs)]))) < 1.e-10 : true
        return all(is_equidistant_symmetric) && all_spacings_identical
    end

    # check if all grids are equally spaced and symmetric around 0:
    if !grids_are_fine(ωs)
        throw(ArgumentError("PartialCorrelator_reg requires equidistant symmetric grid."))
    end

    Δω = diff(ωs[1])[1]
    ωs_max = [ωs[i][end] for i in eachindex(ωs)]
    ωs_int_max = abs.(ωconv) * ωs_max
    ωs_int = ntuple(i -> collect(LinRange(-ωs_int_max[i], ωs_int_max[i], 1 + trunc(Int, real(2. * ωs_int_max[i] / Δω) + 0.1   ))), D)
    # check spacing of ωs_int:
    @assert all( [ωs_int[i][2] - ωs_int[i][1] ≈ Δω for i in eachindex(ωs)])

    Nωs_ext = length.(ωs)
    ωconvMat = trunc.(Int, ωconv)
    ωconvOff = ntuple(i -> sum([ωconvMat[i,j] == -1 ? Nωs_ext[j] : -ωconvMat[i,j] for j in 1:D]) + 1, D)

    return ωs_int, ωconvMat, ωconvOff
end

function get_regular_1D_MF_Kernel(ωs::Vector{Float64}, ωdisc::Vector{Float64})

    Kernel = 1. ./ (im .* ωs .- ωdisc')

    is_divergent = .!isfinite.(Kernel)
    (@view Kernel[is_divergent]) .= 0. # This corresponds the result with a Adisc(0) δ(ω) broadened to --> Adisc(0)/Δx0 1_[x[-1]/2,x[1]/2]  where 1_[a,b](x) is the rectangular function which gives one if x∈[a,b]
    @assert any(isfinite.(Kernel))

    return Kernel
end


###############################
#### Pointwise evaluation: ####
###############################

function (Gp :: PartialCorrelator_reg{D})(w   :: Vararg{Int, D} )   ::ComplexF64 where {D}
    return evaluate_with_ωconversion(Gp, w...)
end

function evaluate_with_ωconversion(
    Gp  :: PartialCorrelator_reg{D},
    w   :: Vararg{Union{Int, Colon}, D}
    )   :: Union{ComplexF64, AbstractArray{ComplexF64}} where {D}
    return Gp.tucker[(Gp.ωconvMat * SA[w...] + Gp.ωconvOff)...]
end




function evaluate_with_ωconversion_KF(Gp::PartialCorrelator_reg{D}, idx::Vararg{Int,D})  ::Vector{ComplexF64} where{D}
    return evaluate_without_ωconversion_KF(Gp, (Gp.ωconvMat * SA[idx...] + Gp.ωconvOff)...)
end

function evaluate_without_ωconversion_KF(Gp::PartialCorrelator_reg{D}, idx::Vararg{Int,D})  ::Vector{ComplexF64} where {D}
    res = Gp.tucker.center
    sz_Adisc = size(res)
    for i in 1:D
        #println("i: ", i)
        #res = view(psf.Kernels[i], idx[i]:idx[i], :) * reshape(res, (psf.sz[i], prod(psf.sz[i+1:D])))
        res = view(Gp.tucker.legs[i], idx[i]:idx[i], :) * reshape(res, (sz_Adisc[i], prod(sz_Adisc[i+1:D])*i))    # version for Kernels[idx_ext, idx_int]
        res = reshape(res, (prod(sz_Adisc[i+1:D]), i))
        res = cat(res, conj.(res[:,1]), dims=2)
        #j = D - i + 1
        #res = reshape(res, (prod(psf.sz[1:j-1]), psf.sz[j])) * view(psf.Kernels[j], idx[j], :)
        #res = reshape(res, (prod(sz_Adisc[1:j-1]), sz_Adisc[j])) * view(Gp.Kernels[j], idx[j], :)    # version for Kernels[idx_int, idx_ext]
    end
    #println("length(res): ", length(res))
    return reshape(res, D+1)
end



##################################
#### precomputing all values (WITH frequency rotation): ####
##################################

function precompute_all_values_MF(
    Gp :: PartialCorrelator_reg{D},
) ::Array{ComplexF64,D} where{D}
    
    @assert Gp.formalism == "MF"

    data_unrotated = contract_1D_Kernels_w_Adisc_mp(Gp.tucker.legs, Gp.tucker.center)  # contributions from regular kernel
    
    # compute anomalous contribution for Matsubara formalism
    ## check for ω=0 entries in ωs_int:
    for d in 1:D
        is_zero_ωs_int = abs.(Gp.tucker.ωs_legs[d]) .< 1.e-10
        is_zero_ωdisc = abs.(Gp.tucker.ωs_center[d]) .< 1.e-10
        if any(is_zero_ωs_int) && any(is_zero_ωdisc)  ## currently only support single bosonic frequency
            #println("Has anomalous contribution: ")
            Adisc_ano = dropdims(Gp.tucker.center[[Colon() for _ in 1:d-1]..., is_zero_ωdisc, [Colon() for _ in d+1:D]...], dims=d)

            Adisc_anoβ_temp = dropdims(Gp.Adisc_anoβ[[Colon() for _ in 1:d-1]..., is_zero_ωdisc, [Colon() for _ in d+1:D]...], dims=d)
            #println("size of Adisc_ano: ", size(Adisc_ano))
            # compute anomalous contribution:
            Kernels_ano = [Gp.tucker.legs[1:d-1]..., Gp.tucker.legs[d+1:D]...]
            # add β/2 contribution?
            β = 2. * π / abs(Gp.tucker.ωs_legs[1][2]-Gp.tucker.ωs_legs[1][1])
            #println("β: ", β)
            values_ano = β* contract_1D_Kernels_w_Adisc_mp(Kernels_ano, Adisc_anoβ_temp)
            for dd in 1:D-1
                Kernels_tmp = [Kernels_ano...]
                #println("maxima before: ", maximum(abs.(Kernels_tmp[dd])))
                Kernels_tmp[dd] = Kernels_tmp[dd].^2
                #println("maxima after: ", maximum(abs.(Kernels_tmp[dd])))
                values_ano .+= contract_1D_Kernels_w_Adisc_mp(Kernels_tmp, Adisc_ano)
            end
            values_ano .*= -0.5
    
            #myview = view(data_unrotated, [Colon() for _ in 1:d-1]..., argmax(is_zero_ωs_int), [Colon() for _ in d+1:D]...)
            #println("size of view: ", size(myview))
            #println("size of values_ano = ", size(values_ano))
            view(data_unrotated, [Colon() for _ in 1:d-1]..., argmax(is_zero_ωs_int), [Colon() for _ in d+1:D]...) .+= values_ano
        end
    end
    

    ## perform frequency rotation:
    strides_internal = [stride(data_unrotated, i) for i in 1:D]'
    strides4rot = ((strides_internal * Gp.ωconvMat)...,)
    offset4rot = sum(strides4rot) - sum(strides_internal) + strides_internal * Gp.ωconvOff
    sv = StridedView(data_unrotated, (length.(Gp.ωs_ext)...,), strides4rot, offset4rot)


    return sv[[Colon() for _ in 1:D]...]
end


function precompute_all_values_KF(
    Gp :: PartialCorrelator_reg{D},
) where{D}
    
    @assert Gp.formalism == "KF"

    data_unrotated = contract_KF_Kernels_w_Adisc_mp(Gp.tucker.legs, Gp.tucker.center)  # contributions from fully-retarded kernels

    ## perform frequency rotation:
    strides_internal = [stride(data_unrotated, i) for i in 1:D]'
    strides4rot = ((strides_internal * Gp.ωconvMat)...,)
    offset4rot = sum(strides4rot) - sum(strides_internal) + strides_internal * Gp.ωconvOff
    sv = StridedView(data_unrotated, (length.(Gp.ωs_ext)..., D+1), (strides4rot..., strides(data_unrotated)[D+1]), offset4rot)


    return sv[[Colon() for _ in 1:D+1]...]
end