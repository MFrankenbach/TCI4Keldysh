"""
This file is concerned with representing and evaluating partial correlators in both formalisms (Matsubara/Keldysh).
It provides the basic struct to represent a partial correlator.
"""

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
    T           ::Float64
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
    isFermi ::  SVector{D,Bool}                 # encodes whether the i-th dimension of tucker encodes a bosonic or fermionic frequency (only relevant in MF)

    ####################
    ### Constructors ###
    ####################
    function PartialCorrelator_reg(T::Float64, formalism::String, Adisc::Array{Float64,D}, ωdisc::Vector{Float64}, ωs_ext::NTuple{D,Vector{Float64}}, ωconvMat::Matrix{Int}; is_compactAdisc::Bool=true) where {D}
        if !(formalism == "MF" || formalism == "KF")
            throw(ArgumentError("formalism must be MF when unbroadened Adisc is input."))
        end
        if DEBUG()
            println("Constructing PartialCorrelator_reg (WITHOUT broadening).")
        end

        ωs_int, ωconvOff, isFermi = _trafo_ω_args(ωs_ext, ωconvMat)
        if is_compactAdisc
        # Delete rows/columns that contain only zeros
        _, ωdiscs, Adisc = compactAdisc(ωdisc, Adisc)
        else
            ωdiscs, Adisc = [ωdisc for _ in 1:D], Adisc
        end
        @VERBOSE "Size ωs_int: $(length.(ωs_int))\n"
        @VERBOSE "Size ωdisc: $(length.(ωdiscs))\n"
        @TIME Kernels = [get_regular_1D_MF_Kernel(ωs_int[i], ωdiscs[i]) for i in 1:D] "Precomputing 1D kernels (for MF)."
        if formalism == "MF" && !all(isFermi)
            i_ωbos = argmax(.!isFermi)
            # all entries where bosonic frequency is (almost) zero
            Adisc_anoβ = Adisc[[Colon() for _ in 1:i_ωbos-1]..., abs.(ωdiscs[i_ωbos]) .< 1e-8, [Colon() for _ in i_ωbos+1:D]...]
        else 
            Adisc_anoβ = Array{ComplexF64,D}(undef, zeros(Int, D)...)
        end
        tucker = TuckerDecomposition(Adisc, Kernels; ωs_center=ωdiscs, ωs_legs=[ωs_int...])
        
        return new{D}(T, formalism, tucker, Adisc_anoβ, ωs_ext, ωconvMat, ωconvOff, isFermi)
    end
    function PartialCorrelator_reg(
        T::Float64, formalism::String, Acont::AbstractTuckerDecomp{D}, ωs_ext::NTuple{D,Vector{Float64}}, ωconvMat::Matrix{Int}
        ; hilbert_method::String="linear_interpolation"
        ) where {D}
        if !(formalism == "MF" || formalism == "KF")
            throw(ArgumentError("formalism must be MF or KF."))
        end
        if DEBUG()
            println("Constructing PartialCorrelator_reg (WITH broadening).")
        end

        ωs_int, ωconvOff, isFermi = _trafo_ω_args(ωs_ext, ωconvMat)
        δωcont = get_ω_binwidths(Acont.ωs_legs[1])
        tucker = TuckerDecomposition(Acont.center .+ 0im, Acont.legs; ωs_center=Acont.ωs_center, ωs_legs=ωs_int)#deepcopy(Acont)
        if formalism == "MF"
            # 1.: rediscretization of broadening kernel
            # 2.: contraction with regular kernel
            @TIME tucker.legs = [get_regular_1D_MF_Kernel(ωs_int[i], Acont.ωs_legs) * (get_ω_binwidths(Acont.ωs_legs[i]) .* Acont.legs[i]) for i in 1:D] "Constructing 1D Kernels (for MF)."
        else
            # compute retarded 1D kernels
            tucker.legs = if hilbert_method=="linear_interpolation"
                # check that frequency grid is smaller than Acont grid to interpolate real part
                for i in 1:D
                    @assert (maximum(ωs_int[i]) <= maximum(Acont.ωs_legs[i])) "Cannot interpolate real part of kernel: $(maximum.(Acont.ωs_legs)) vs $(maximum.(ωs_int))"
                    @assert (minimum(ωs_int[i]) >= minimum(Acont.ωs_legs[i])) "Cannot interpolate real part of kernel: $(minimum.(Acont.ωs_legs)) vs $(minimum.(ωs_int))"
                end
                # [-im * π * my_hilbert_trafo(ωs_int[i], Acont.ωs_legs[i], Acont.legs[i]) for i in 1:D]
                hts = Vector{Matrix{ComplexF64}}(undef,D)
                Threads.@threads for i in 1:D
                    hts[i] = -im * π * my_hilbert_trafo(ωs_int[i], Acont.ωs_legs[i], Acont.legs[i])
                end
                hts
            elseif hilbert_method=="fft"
                # check that grid is equidistant:
                error("TODO: fix hilbert_fft")
                if maximum(abs.(diff(δωcont) )) > 1e-10
                    throw(ArgumentError("ωcont must be an equidistant grid."))
                end
                [-im * π * hilbert_fft(Acont.legs[i]; dims=1) for i in 1:D]
            else
                error("Invalid hilbert transform method $(hilbert_method)")
            end
        end
        Adisc_anoβ = Array{ComplexF64,D}(undef, zeros(Int, D)...)

        return new{D}(T, formalism, tucker, Adisc_anoβ, ωs_ext, ωconvMat, ωconvOff, isFermi)
    end

end

"""
Construct (D+1)-point partial correlator with frequency grids of size 2^R / 2^R+1
"""
function PartialCorrelator_reg(D::Int, T::Float64, R::Int, path::String, Ops::Vector{String}, ωconvMat::Matrix{Int}, formalism::String="MF"; flavor_idx=1, nested_ωdisc=false)
    if formalism!="MF"
        error("only Matsubara available here")
    end

    Adisc = load_Adisc(path, Ops, flavor_idx)
    ωdisc = load_ωdisc(path, Ops; nested_ωdisc=nested_ωdisc)

    Nhalf = 2^(R-1)
    ombos = MF_grid(T, Nhalf, false)
    omfer = MF_grid(T, Nhalf, true)
    ωs_ext = ntuple(i -> i>1 ? omfer : ombos, D)

    return PartialCorrelator_reg(T, formalism, Adisc, ωdisc, ωs_ext, ωconvMat)
end

"""
Only if Gp is intact can it be evaluated correctly.
"""
function intact(Gp::PartialCorrelator_reg{D}) where {D}
    return !Gp.tucker.modified
end

function update_frequency_args!(Gp::PartialCorrelator_reg{D}) where{D}
    Gp.tucker.ωs_legs, Gp.ωconvOff, Gp.isFermi = _trafo_ω_args(Gp.ωs_ext, Gp.ωconvMat)

    update_kernels!(Gp)

    return nothing
end

function update_kernels!(Gp::PartialCorrelator_reg{D}) where{D}
    @TIME Gp.tucker.legs = [get_regular_1D_MF_Kernel(Gp.tucker.ωs_legs[i], Gp.tucker.ωs_center[i]) for i in 1:D] "Precomputing 1D kernels (for MF)."
    return nothing
end

"""
_trafo_ω_args

Compute internal frequencies from external ωs.
Also compute the index transformation matrices:
External indices have the ranges OneTo.(length.(ωs))


Internal indices have the ranges OneTo.(length.(ωs_new))


"""
function _trafo_ω_args(ωs::NTuple{D,Vector{T}}, ωconvMat::AbstractMatrix{Int}) where{D,T}
    function grids_are_fine(grids)
        tol = 1.e-10
        Δgrids = [diff(g) for g in grids]
        is_equidistant_symmetric = [abs(grids[i][1] + grids[i][end])<1.e-12 && (maximum(abs.(diff(Δgrids[i]))) < tol) for i in eachindex(ωs)]
        all_spacings_identical = D > 1 ? maximum(abs.(diff([Δgrids[i][1] for i in eachindex(ωs)]))) < tol : true
        return all(is_equidistant_symmetric) && all_spacings_identical
    end

    # check if all grids are equally spaced and symmetric around 0:
    if !grids_are_fine(ωs)
        throw(ArgumentError("PartialCorrelator_reg requires equidistant symmetric grid."))
    end

    Δω = diff(ωs[1])[1]
    ωs_max = [ωs[i][end] for i in eachindex(ωs)]
    ωs_int_max = abs.(ωconvMat) * ωs_max
    ωs_int = [collect(LinRange(-ωs_int_max[i], ωs_int_max[i], 1 + trunc(Int, real(2. * ωs_int_max[i] / Δω) + 0.1   ))) for i in 1:D]
    # check spacing of ωs_int:
    @assert all( [ωs_int[i][2] - ωs_int[i][1] ≈ Δω for i in eachindex(ωs)])

    Nωs_ext = length.(ωs)
    ωconvOff = ntuple(i -> sum([ωconvMat[i,j] == -1 ? Nωs_ext[j] : -ωconvMat[i,j] for j in 1:D]) + 1, D)

    isFermi = mod.(length.(ωs_int), 2) .== 0

    return ωs_int, ωconvOff, isFermi
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

"""
Evaluate regular part with frequency transformation.
"""
function (Gp :: PartialCorrelator_reg{D})(w   :: Vararg{Int, D} )   ::ComplexF64 where {D}
    return evaluate_with_ωconversion(Gp, w...)
end

"""
Evaluate regular part with frequency transformation.
Prune tucker center.
"""
function (Gp :: PartialCorrelator_reg{D})(tucker_cut::Int, w::Vararg{Int, D})::ComplexF64 where {D}
    return evaluate_with_ωconversion(tucker_cut, Gp, w...)
end

"""
Evaluate regular part with frequency transformation.
Prune tucker center.
"""
function evaluate_with_ωconversion(
    tucker_cut::Int,
    Gp  :: PartialCorrelator_reg{D},
    w   :: Vararg{Union{Int, Colon}, D}
    )   :: Union{ComplexF64, AbstractArray{ComplexF64}} where {D}
    return Gp.tucker(tucker_cut, (Gp.ωconvMat * SA[w...] + Gp.ωconvOff)...)
end

"""
Evaluate regular part with frequency transformation.
"""
function evaluate_with_ωconversion(
    Gp  :: PartialCorrelator_reg{D},
    w   :: Vararg{Union{Int, Colon}, D}
    )   :: Union{ComplexF64, AbstractArray{ComplexF64}} where {D}
    # return Gp.tucker[(Gp.ωconvMat * SA[w...] + Gp.ωconvOff)...]
    return Gp.tucker((Gp.ωconvMat * SA[w...] + Gp.ωconvOff)...)
end

"""
Evaluate anomalous contribution
-0.5(β + ∑_{i≠j}1/(i⋅ωi - ϵi)) ∏_{i≠j}1/(i⋅ωi - ϵi)
where is the bosonic index.

Here we take D external frequencies and return zero(ComplexF64) if the internal bosonic frequency is nonzero.
"""
function evaluate_ano_with_ωconversion(
    Gp :: PartialCorrelator_reg{D},
    w :: Vararg{Int, D}
    ) where {D}
    bos_idx = get_bosonic_idx(Gp)
    w_int = Gp.ωconvMat * SA[w...] + Gp.ωconvOff
    beta = 1.0/Gp.T
    if abs(Gp.tucker.ωs_legs[bos_idx][w_int[bos_idx]]) > 1.e-2*Gp.T
        # not in anomalous range
        return zero(ComplexF64)
    else
        # IN anomalous range
        ret = zero(ComplexF64)
        for c in CartesianIndices(Gp.Adisc_anoβ)
            # compute anomalous kernel
            sum = beta
            prod = one(ComplexF64)
            for i in 1:D
                if i==bos_idx
                    continue
                else
                    sum += Gp.tucker.legs[i][w_int[i], c[i]]
                    prod *= Gp.tucker.legs[i][w_int[i], c[i]]
                end
            end
            ret += sum * prod * Gp.Adisc_anoβ[c]
        end
        return -0.5 * ret
    end
end

"""
struct designed to evaluate anomalous term of PartialCorrelator efficiently, i.e., compute:
    -0.5*(β + sum_i≠j 1/(iω_j - ϵ_j)) ⋅ ∏_i≠j 1/(iω_j - ϵ_j)
* Adisc_ano_red: needed when number of PartialCorrelators has been reduced: Then
Adisc_ano = Adisc_ano(Gp1) + (-1)^(D-1) * Adisc_ano(Gp2)
Adisc_ano_red = Adisc_ano(Gp1) + (-1)^(D) * Adisc_ano(Gp2)
"""
struct AnomalousEvaluator{T,D,N}
    beta::Float64
    bos_zero_idx::Int
    nonbos_idx::NTuple{N,Int}
    Adisc_ano::Array{T,N}
    Adisc_ano_red::Array{T,N}
    ωlegs::Vector{Matrix{T}}
    ωlegs_sq::Vector{Matrix{T}} # pointwise squared
    ωconvMat::SMatrix{D,D,Int}
    ωconvOff::SVector{D,Int}

    function AnomalousEvaluator(Gp::PartialCorrelator_reg{D}) where {D}
        @assert ano_term_required(Gp) "Trying to build AnomalousEvaluator when anomalous term is not required"
        bos_idx = get_bosonic_idx(Gp)
        nonbos_idx = ntuple(i -> (i<bos_idx ? i : i+1), D-1)
        bos_zero_idx = findfirst(w -> abs(Gp.tucker.ωs_legs[bos_idx][w]) < 1.e-2*Gp.T, 1:length(Gp.tucker.ωs_legs[bos_idx]))

        # permute bos_idx to first index
        perm = collect(1:D)
        perm[1] = bos_idx
        perm[bos_idx] = 1
        ωconvMat = Gp.ωconvMat[perm, :]
        ωconvOff = Gp.ωconvOff[perm]
        beta = 1.0/Gp.T

        Adisc_ano = copy(dropdims(Gp.Adisc_anoβ; dims=(bos_idx,)))
        ano_slice = Vector{Any}(undef, D)
        ano_slice[:] .= Colon()
        zero_idx = findfirst(ϵ -> abs(ϵ)<1.e-14, Gp.tucker.ωs_center[bos_idx])
        ano_slice[bos_idx] = zero_idx:zero_idx
        Adisc_ano_red = dropdims(Gp.tucker.center[ano_slice...]; dims=(bos_idx,))
        ωlegs = copy.([Gp.tucker.legs[i] for i in nonbos_idx])
        ωlegs_sq = [Gp.tucker.legs[i] .^ 2 for i in nonbos_idx]
        T = eltype(Adisc_ano)

        # check
        for i in eachindex(ωlegs)
            @assert size(ωlegs[i], 2)==size(Adisc_ano, i)
            @assert size(ωlegs[i], 2)==size(Adisc_ano_red, i)
        end

        return new{T, D, D-1}(beta, bos_zero_idx, nonbos_idx, Adisc_ano, Adisc_ano_red, ωlegs, ωlegs_sq, ωconvMat, ωconvOff)
    end
end

"""
4-point anomalous term
"""
function (anev::AnomalousEvaluator{T,3,2})(w::Vararg{Int,3}) where {T}
    # always has bosonic index first
    w_int = anev.ωconvMat * SA[w...] + anev.ωconvOff
    if w_int[1] == anev.bos_zero_idx
        # IN anomalous range
        bterm = anev.beta * transpose(anev.ωlegs[1][w_int[2],:]) * anev.Adisc_ano * anev.ωlegs[2][w_int[3],:]
        lsqterm = transpose(anev.ωlegs_sq[1][w_int[2],:]) * anev.Adisc_ano_red * anev.ωlegs[2][w_int[3],:]
        rsqterm = transpose(anev.ωlegs[1][w_int[2],:]) * anev.Adisc_ano_red * anev.ωlegs_sq[2][w_int[3],:]
        return -0.5 * (bterm + lsqterm + rsqterm)
    else
        # not in anomalous range
        return zero(T)
    end
end

"""
3-point anomalous term
"""
function (anev::AnomalousEvaluator{T,2,1})(w::Vararg{Int,2}) where {T}
    # always has bosonic index first
    w_int = anev.ωconvMat * SA[w...] + anev.ωconvOff
    if w_int[1] == anev.bos_zero_idx
        # IN anomalous range
        bterm = anev.beta * transpose(anev.Adisc_ano) * anev.ωlegs[1][w_int[2],:]
        sqterm = transpose(anev.Adisc_ano_red) * anev.ωlegs_sq[1][w_int[2],:]
        return -0.5 * (bterm + sqterm)
    else
        # not in anomalous range
        return zero(T)
    end
end

"""
2-point anomalous term
"""
function (anev::AnomalousEvaluator{T,1,0})(w::Int) where {T}
    w_int = anev.ωconvMat * SA[w...] + anev.ωconvOff
    if w_int[1] == anev.bos_zero_idx
        return -0.5 * anev.beta * anev.Adisc_ano[begin]
    else
        return zero(T)
    end
end


function evaluate_with_ωconversion_KF(Gp::PartialCorrelator_reg{D}, idx::Vararg{Int,D})  ::Vector{ComplexF64} where{D}
    return evaluate_without_ωconversion_KF(Gp, (Gp.ωconvMat * SA[idx...] + Gp.ωconvOff)...)
end

"""
Evaluate Partial Keldysh correlator on given retarded component iR=1...(D+1)
"""
function evaluate_with_ωconversion_KF(Gp::PartialCorrelator_reg{D}, iR::Int, idx::Vararg{Int,D})::ComplexF64 where{D}
    return evaluate_without_ωconversion_KF(Gp, iR, (Gp.ωconvMat * SA[idx...] + Gp.ωconvOff)...)
end

"""
Evaluate Partial Keldysh correlator on given retarded component iR=1...(D+1): Generic fallback
"""
function evaluate_without_ωconversion_KF(Gp::PartialCorrelator_reg{D}, iR::Int, idx::Vararg{Int,D})::ComplexF64 where {D}
    return evaluate_without_ωconversion_KF(Gp, idx...)[iR]
end


"""
Evaluate Partial Keldysh correlator on given retarded component iR=1...4: Direct loop\\
            K^[iR=1](ω₁,ω₂,ω₃,ω₄)  =       K^R(ω₁)K^R(ω₂)K^R(ω₃)  \\
            K^[iR=2](ω₁,ω₂,ω₃,ω₄)  =       K^A(ω₁)K^R(ω₂)K^R(ω₃)  \\
            K^[iR=3](ω₁,ω₂,ω₃,ω₄)  =       K^A(ω₁)K^A(ω₂)K^R(ω₃)  \\
            K^[iR=4](ω₁,ω₂,ω₃,ω₄)  =       K^A(ω₁)K^A(ω₂)K^A(ω₃)  \\
Stored kernel is retarded (K^R).\\
CAREFUL: Assumes tucker center is real-valued.
"""
function evaluate_without_ωconversion_KF(Gp::PartialCorrelator_reg{3}, iR::Int, idx::Vararg{Int,3})::ComplexF64

    #=
    ret = zero(ComplexF64)    
    n1, n2, n3 = size(Gp.tucker.center)
    @inbounds for k in 1:n3
        ret3 = zero(ComplexF64)
        for j in 1:n2
            ret2 = zero(ComplexF64)
            for i in 1:n1
                ret2 += Gp.tucker.legs[1][idx[1], i] * Gp.tucker.center[i, j, k]
            end
            if iR==2
                ret2 = conj(ret2)
            end
            ret3 += ret2 * Gp.tucker.legs[2][idx[2], j] 
        end
        if iR==3
            ret3 = conj(ret3)
        end
        ret += ret3 * Gp.tucker.legs[3][idx[3], k] 
    end

    if iR==4
        return conj(ret)
    else
        return ret
    end
    =#

    # # much slower
    # @inbounds N = size(Gp.tucker.legs[3])[2]
    # @inbounds @views temp = [LinearAlgebra.BLAS.dotu(Gp.tucker.legs[1][idx[1],:], LinearAlgebra.BLAS.gemv('N',Gp.tucker.center[:,:,i],Gp.tucker.legs[2][idx[2],:])) for i in 1:N]
    # @inbounds @views return LinearAlgebra.BLAS.dotu(temp, Gp.tucker.legs[3][idx[3],:])

    res = Gp.tucker.center
    sz_Adisc = size(res)
    D = 3
    @inbounds for i in 1:D
        res = view(Gp.tucker.legs[i], idx[i]:idx[i], :) * reshape(res, (sz_Adisc[i], prod(sz_Adisc[i+1:D])))    # version for Kernels[idx_ext, idx_int]
        if i==iR-1
            res = conj.(res)
        end
        res = reshape(res, prod(sz_Adisc[i+1:D]))
        # res = cat(res, conj.(res[:,1]), dims=2)
    end
    return only(res)
end


function evaluate_without_ωconversion_KF(Gp::PartialCorrelator_reg{D}, idx::Vararg{Int,D})  ::Vector{ComplexF64} where {D}
    res = Gp.tucker.center
    sz_Adisc = size(res)
    for i in 1:D
        #res = view(psf.Kernels[i], idx[i]:idx[i], :) * reshape(res, (psf.sz[i], prod(psf.sz[i+1:D])))
        res = view(Gp.tucker.legs[i], idx[i]:idx[i], :) * reshape(res, (sz_Adisc[i], prod(sz_Adisc[i+1:D])*i))    # version for Kernels[idx_ext, idx_int]
        res = reshape(res, (prod(sz_Adisc[i+1:D]), i))
        res = cat(res, conj.(res[:,1]), dims=2)
        #j = D - i + 1
        #res = reshape(res, (prod(psf.sz[1:j-1]), psf.sz[j])) * view(psf.Kernels[j], idx[j], :)
        #res = reshape(res, (prod(sz_Adisc[1:j-1]), sz_Adisc[j])) * view(Gp.Kernels[j], idx[j], :)    # version for Kernels[idx_int, idx_ext]
    end
    return reshape(res, D+1)
end



##################################
#### precomputing all values (WITH frequency rotation): ####
##################################

function precompute_reg_div_values_MF_without_ωconv(d::Int, Adisc_ano::Array{ComplexF64,D}, legs::Vector{Matrix{ComplexF64}}) where{D}


    @assert D == length(legs)-1 "Incompatible D=$D and length(legs)=$(length(legs))"
    #println("size of Adisc_ano: ", size(Adisc_ano))
    # compute anomalous contribution:
    Kernels_ano = [legs[1:d-1]..., legs[d+1:D+1]...]

    values_ano = zeros(ComplexF64, size.(Kernels_ano, 1)...)
    for dd in 1:D
        Kernels_tmp = [Kernels_ano...]
        #println("maxima before: ", maximum(abs.(Kernels_tmp[dd])))
        Kernels_tmp[dd] = Kernels_tmp[dd].^2
        #println("maxima after: ", maximum(abs.(Kernels_tmp[dd])))
        values_ano .+= contract_1D_Kernels_w_Adisc_mp(Kernels_tmp, Adisc_ano)
    end
    values_ano .*= -0.5

    return values_ano
end

function precompute_reg_values_MF_without_ωconv(
    Gp :: PartialCorrelator_reg{D},
) ::Array{ComplexF64,D} where{D}
    
    @assert Gp.formalism == "MF"


    data_unrotated = contract_1D_Kernels_w_Adisc_mp(Gp.tucker.legs, Gp.tucker.center)  # contributions from regular kernel

    
    # compute anomalous contribution for Matsubara formalism
    ## check for ω=0 entries in ωs_int:
    if !all(Gp.isFermi)    # if bosonic frequencies exist
        d = argmax(.!Gp.isFermi) # find bosonic direction
        
        is_zero_ωs_int = abs.(Gp.tucker.ωs_legs[d]) .< 1.e-10
        is_zero_ωdisc = abs.(Gp.tucker.ωs_center[d]) .< 1.e-10
        if any(is_zero_ωs_int) && any(is_zero_ωdisc)  ## currently only support single bosonic frequency
            #println("Has anomalous contribution: ")
            Adisc_ano = dropdims(Gp.tucker.center[[Colon() for _ in 1:d-1]..., is_zero_ωdisc, [Colon() for _ in d+1:D]...], dims=d)

            values_ano = precompute_reg_div_values_MF_without_ωconv(d, Adisc_ano, Gp.tucker.legs)
            #println("diff between the two value_ano's: ", maximum(abs.(values_ano - values_ano2)))
            #println("magnitude of values_ano = ", maximum(abs.(values_ano)))
            #println("magnitude of rest = ", maximum(abs.(data_unrotated)))

    
            #myview = view(data_unrotated, [Colon() for _ in 1:d-1]..., argmax(is_zero_ωs_int), [Colon() for _ in d+1:D]...)
            #println("size of view: ", size(myview))
            #println("size of values_ano = ", size(values_ano))
            view(data_unrotated, [Colon() for _ in 1:d-1]..., argmax(is_zero_ωs_int), [Colon() for _ in d+1:D]...) .+= values_ano
        end
    end
    

    return data_unrotated
end


function precompute_ano_values_MF_without_ωconv(
    Gp :: PartialCorrelator_reg{D}
) ::Array{ComplexF64, D} where{D}
    
    @assert Gp.formalism == "MF"

    values_ano = Array{ComplexF64, D}(undef, zeros(Int, D)...)

    # compute anomalous contribution for Matsubara formalism
    ## check for ω=0 entries in ωs_int:
    if length(Gp.Adisc_anoβ) > 0 && !all(Gp.isFermi)    # if bosonic frequencies and anomalous PSF exist
        d = argmax(.!Gp.isFermi) # find bosonic direction
    
        is_zero_ωs_int = abs.(Gp.tucker.ωs_legs[d]) .< 1.e-10
        #is_zero_ωdisc = abs.(Gp.tucker.ωs_center[d]) .< 1.e-10
        if any(is_zero_ωs_int)  ## currently only support single bosonic frequency
            
            #Adisc_anoβ_temp = dropdims(Gp.Adisc_anoβ[[Colon() for _ in 1:d-1]..., is_zero_ωdisc, [Colon() for _ in d+1:D]...], dims=d)
            #println("size of Adisc_ano: ", size(Adisc_ano))
            # compute anomalous contribution:
            Kernels_ano = [Gp.tucker.legs[1:d-1]..., Gp.tucker.legs[d+1:D]...]
            # add β/2 contribution?
            β = 1/Gp.T
            #println("β: ", β)
            values_ano = -0.5 * β* (D == 1 ? Gp.Adisc_anoβ : contract_1D_Kernels_w_Adisc_mp(Kernels_ano, dropdims(Gp.Adisc_anoβ, dims=d)))
            values_ano = reshape(values_ano, (size(values_ano)[1:d-1]..., 1, size(values_ano)[d:D-1]...))
    
            #myview = view(data_unrotated, [Colon() for _ in 1:d-1]..., argmax(is_zero_ωs_int), [Colon() for _ in d+1:D]...)
            #println("size of view: ", size(myview))
            #println("size of values_ano = ", size(values_ano))
            #view(data_unrotated, [Colon() for _ in 1:d-1]..., argmax(is_zero_ωs_int), [Colon() for _ in d+1:D]...) .+= values_ano
        end
    end
    

    return values_ano
end

function precompute_all_values_MF_without_ωconv(
    Gp :: PartialCorrelator_reg{D},
) ::Array{ComplexF64,D} where{D}
    
    @assert Gp.formalism == "MF"

    data_unrotated = precompute_reg_values_MF_without_ωconv(Gp)
    data_ano = precompute_ano_values_MF_without_ωconv(Gp)
    if length(data_ano) > 0
        d = argmax(.!Gp.isFermi) # find bosonic direction
        is_zero_ωs_int = abs.(Gp.tucker.ωs_legs[d]) .< 1.e-10
        i_ωbos = argmax(is_zero_ωs_int)
        view(data_unrotated, [Colon() for _ in 1:d-1]..., i_ωbos:i_ωbos, [Colon() for _ in d+1:D]...) .+= data_ano
    end

    return data_unrotated
end


function precompute_all_values_MF(
    Gp :: PartialCorrelator_reg{D},
) ::Array{ComplexF64,D} where{D}
    
    @assert Gp.formalism == "MF"
    @assert intact(Gp) "TuckerDecomposition has been modified"

    data_unrotated = precompute_all_values_MF_without_ωconv(Gp)
    maxval1 = maximum(abs.(data_unrotated))
    #println("Gp val before rotation: ", maxval1)

    ## perform frequency rotation:
    strides_internal = [stride(data_unrotated, i) for i in 1:D]'
    strides4rot = ((strides_internal * Gp.ωconvMat)...,)
    offset4rot = sum(strides4rot) - sum(strides_internal) + strides_internal * Gp.ωconvOff
    sv = StridedView(data_unrotated, (length.(Gp.ωs_ext)...,), strides4rot, offset4rot)
    res = Array{ComplexF64,D}(sv[[Colon() for _ in 1:D]...])
    maxval2 = maximum(abs.(res))

    #println("Gp val after rotation: ", maxval2)
    #println("dev in maxvals due to rotation: $(maxval1 - maxval2)")


    return res
end

"""
Precompute values, including only regular kernel convolutions and frequency rotations
TODO: TEST
"""
function precompute_all_values_MF_noano(
    Gp::PartialCorrelator_reg{D}
    )::Array{ComplexF64,D} where {D}
    
    @assert intact(Gp) "TuckerDecomposition has been modified"
    data_unrotated = contract_1D_Kernels_w_Adisc_mp(Gp.tucker.legs, Gp.tucker.center)

    # perform frequency rotation:
    strides_internal = [stride(data_unrotated, i) for i in 1:D]'
    strides4rot = ((strides_internal * Gp.ωconvMat)...,)
    offset4rot = sum(strides4rot) - sum(strides_internal) + strides_internal * Gp.ωconvOff
    sv = StridedView(data_unrotated, (length.(Gp.ωs_ext)...,), strides4rot, offset4rot)
    res = Array{ComplexF64,D}(sv[[Colon() for _ in 1:D]...])

    return res
end

"""
Precompute values (kernel contraction and frequency rotation)
"""
function precompute_all_values(
    Gp::PartialCorrelator_reg{D}
    )::Array{ComplexF64,D} where {D}
    
    @assert intact(Gp) "TuckerDecomposition has been modified"
    data_unrotated = contract_1D_Kernels_w_Adisc_mp(Gp.tucker.legs, Gp.tucker.center)

    # perform frequency rotation:
    strides_internal = [stride(data_unrotated, i) for i in 1:D]'
    strides4rot = ((strides_internal * Gp.ωconvMat)...,)
    offset4rot = sum(strides4rot) - sum(strides_internal) + strides_internal * Gp.ωconvOff
    sv = StridedView(data_unrotated, (length.(Gp.ωs_ext)...,), strides4rot, offset4rot)
    res = Array{ComplexF64,D}(sv[[Colon() for _ in 1:D]...])

    return res
end

"""
Compute only anomalous values. For testing (very inefficient).
"""
function precompute_ano_values_MF(
    Gp::PartialCorrelator_reg{D}
    )::Array{ComplexF64,D} where {D}
    return precompute_all_values_MF(Gp) .- precompute_all_values_MF_noano(Gp)
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
    # println("")
    # @show length.(Gp.ωs_ext)
    # @show strides_internal
    # @show Gp.ωconvMat
    # @show strides4rot
    # @show strides(data_unrotated)[D+1]
    # @show offset4rot
    sv = StridedView(data_unrotated, (length.(Gp.ωs_ext)..., D+1), (strides4rot..., strides(data_unrotated)[D+1]), offset4rot)
    res = Array{ComplexF64,D+1}(sv[[Colon() for _ in 1:D+1]...])

    return res
end


function fit_tucker_center(Gp_data::Array{T,D}, legs::Vector{Matrix{T}}) where {T,D}
    @assert length(legs) == D "The number of legs ($(length(legs))) must match the dimensions of Gp_data (D=$D)"

    center = deepcopy(Gp_data)
    for d in 1:D

        if size(center, d) > 1 # dummy dimensions need no fitting
            # transform array to matrix:
            g, partialsize = Lehmann._tensor2matrix(center, Val(d))
            # fit coeffients by solving matrix equation:
            coeffs_t = legs[d] \ g
            # update center:
            center = Lehmann._matrix2tensor(coeffs_t, partialsize, Val(d))           
        end 
        
    end


    # check validity of fit:
    if DEBUG() && length(Gp_data) > 0 && all(size(Gp_data).>1)
        compare = contract_1D_Kernels_w_Adisc_mp(legs, center)
        absdev = maximum(abs.(Gp_data - compare))
        #println("deviation in fit: ", absdev)
        @assert absdev < 1e-10
    end


    return center
end


"""
    discreteLehmannRep4Gp!(Gp_in::TCI4Keldysh.PartialCorrelator_reg{D}; kwargs...)

Compress/represent D-dimensional partial correlator Gp_in with Discrete Lehmann Representation.
"""
function discreteLehmannRep4Gp!(Gp_in::PartialCorrelator_reg{D}; dlr_bos::DLRGrid, dlr_fer::DLRGrid) where{D}
    @assert !dlr_bos.isFermi
    @assert  dlr_fer.isFermi

    dlrgrids_here = [Gp_in.isFermi[i] ? dlr_fer : dlr_bos for i in 1:D]
    ωs_legs_before = deepcopy(Gp_in.tucker.ωs_legs)

    if DEBUG()
        Gp_data_before = precompute_reg_values_MF_without_ωconv(Gp_in) # .tucker[[Colon() for _ in 1:D]...]
        Gp_ano_data_before = precompute_ano_values_MF_without_ωconv(Gp_in) # .tucker[[Colon() for _ in 1:D]...] 
    end

    # overwrite Matsubara frequencies with those necessary for the DLR fit
    for d in 1:D
        Gp_in.tucker.ωs_legs[d] = dlrgrids_here[d].ωn
    end
    update_kernels!(Gp_in)
    Gp_data = precompute_reg_values_MF_without_ωconv(Gp_in) # Gp_in.tucker[[Colon() for _ in 1:D]...]
    Gp_ano_data = precompute_ano_values_MF_without_ωconv(Gp_in) / (-0.5*dlrgrids_here[1].β) # .tucker[[Colon() for _ in 1:D]...]
    #println("max dev:\t", maximum(abs.(Gp_data - Gp_data_before)))

    # update internal frequencies and kernels:
    for d in 1:D
        Gp_in.tucker.legs[d] = TCI4Keldysh.get_regular_1D_MF_Kernel(Gp_in.tucker.ωs_legs[d], dlrgrids_here[d].ω)
        #Gp_in.tucker.legs[d] = Lehmann.Spectral.kernelΩ(Float64, Val(Gp_in.isFermi[d]), Val(:none), collect(-Nωs_legs_div2[d]+1:Nωs_legs_div2[d]-1-Gp_in.isFermi[d]), dlrgrids_here[d].ω, dlrgrids_here[d].β, true)#-    dlrgrids_here[d].kernel_nc
        #Gp_in.tucker.legs[d] = Lehmann.Spectral.kernelΩ(Float64, Val(Gp_in.isFermi[d]), Val(:none), dlrgrids_here[d].n, dlrgrids_here[d].ω, dlrgrids_here[d].β, true)#-    dlrgrids_here[d].kernel_nc
        Gp_in.tucker.ωs_center[d] = dlrgrids_here[d].ω
    end

    Gp_in.tucker.center = fit_tucker_center(Gp_data, Gp_in.tucker.legs)
    Gp_in.Adisc_anoβ = fit_tucker_center(Gp_ano_data, Gp_in.tucker.legs)
    Gp_in.tucker.ωs_legs = ωs_legs_before
    update_frequency_args!(Gp_in)

    if DEBUG()
        Gp_data_after = precompute_reg_values_MF_without_ωconv(Gp_in) # Gp_in.tucker[[Colon() for _ in 1:D]...]  # 
        Gp_ano_data_after = precompute_ano_values_MF_without_ωconv(Gp_in) # .tucker[[Colon() for _ in 1:D]...]

        @VERBOSE "abs dev of Gp due to DLR compression (reg): $(maximum(abs.(Gp_data_after - Gp_data_before)) / maximum(abs.(Gp_data_before)))\n"
        if length(Gp_ano_data_before) > 0
            @VERBOSE "abs dev of Gp due to DLR compression (ano): $(maximum(abs.(Gp_ano_data_after - Gp_ano_data_before)) / maximum(abs.(Gp_ano_data_before)))\n"
        end
        
    end 
    return nothing
end



function permute_Adisc_indices!(Gp_in::PartialCorrelator_reg{D}, idx_order_new::Vector{Int}) ::Nothing where{D}
    
    Gp_in.ωconvMat = Gp_in.ωconvMat[idx_order_new, :]
    Gp_in.tucker.center = permutedims(Gp_in.tucker.center, idx_order_new)
    Gp_in.Adisc_anoβ = permutedims(Gp_in.Adisc_anoβ, idx_order_new)


    Gp_in.tucker.legs = Gp_in.tucker.legs[idx_order_new]
    Gp_in.tucker.ωs_legs = Gp_in.tucker.ωs_legs[idx_order_new]
    Gp_in.tucker.ωs_center = Gp_in.tucker.ωs_center[idx_order_new]

    return nothing
end


"""
    set_DLR_MFfreqs!(Gp::TCI4Keldysh.PartialCorrelator_reg{D}; dlr_bos, dlr_fer)

Set Matsubara frequencies in tucker.ωs_legs to sparse DLR frequencies.
"""
function set_DLR_MFfreqs!(Gp::TCI4Keldysh.PartialCorrelator_reg{D}; dlr_bos::DLRGrid, dlr_fer::DLRGrid) where{D}
    for d in 1:D
        Gp.tucker.ωs_legs[d] = Gp.isFermi[d] ? dlr_fer.ωn : dlr_bos.ωn
    end 
    return nothing
end


function partial_fraction_decomp(Gp_in::PartialCorrelator_reg{D}; idx1::Int, idx2::Int, dlr_bos::DLRGrid, dlr_fer::DLRGrid) where{D}

    @assert 1 ≤ idx1 ≤ idx2 ≤ D

    Gp_out1 = deepcopy(Gp_in)

    # reorder internal frequencies to bring idx1 and idx2 to the front
    begin
        idxs_rest = [collect(1:idx1-1); collect(idx1+1:idx2-1); collect(idx2+1:D)]
        idx_order_new = [[idx1, idx2]; idxs_rest]
        permute_Adisc_indices!(Gp_out1, idx_order_new)
    end

    Gp_out2 = deepcopy(Gp_out1)

    # decide whether the new frequency should be (v1 - v2) or (v2 - v1)
    # ==> make sure that the direction of fermionic frequency directions don't flip for anomalous contributions ∝ β
    prefac = Gp_in.isFermi[2] ? -1 : 1
    # frequencies for new PartialCorrelators from partial fraction decomposition
    Gp_out1.ωconvMat = vcat(vcat(prefac.*(Gp_in.ωconvMat[1:1,:]-Gp_in.ωconvMat[2:2,:]), Gp_in.ωconvMat[1:1,:]), Gp_in.ωconvMat[3:D,:])
    Gp_out2.ωconvMat = vcat(vcat(prefac.*(Gp_in.ωconvMat[1:1,:]-Gp_in.ωconvMat[2:2,:]), Gp_in.ωconvMat[2:2,:]), Gp_in.ωconvMat[3:D,:])


    # prepare frequency arguments to compute Kernel_t (see below) on sparse sampling points
    TCI4Keldysh.update_frequency_args!(Gp_out1)
    TCI4Keldysh.update_frequency_args!(Gp_out2)
    set_DLR_MFfreqs!(Gp_out1; dlr_bos, dlr_fer)
    set_DLR_MFfreqs!(Gp_out2; dlr_bos, dlr_fer)
    TCI4Keldysh.update_kernels!(Gp_out1)
    TCI4Keldysh.update_kernels!(Gp_out2)
    @assert maximum(abs.(Gp_out1.tucker.ωs_legs[1] - Gp_out2.tucker.ωs_legs[1])) < 1e-14
    @assert sum(.!Gp_out1.isFermi) ≤ 1 # it is not allowed to have more than one bosonic frequency!
    @assert sum(.!Gp_out2.isFermi) ≤ 1 # it is not allowed to have more than one bosonic frequency!
    

    # construct Kernel_{n,i,j} := K(i ωₙ - ϵᵢ + ϵⱼ)
    sz_center = size(Gp_out1.tucker.center)
    ωn1 = (Gp_out1.isFermi[1] ? dlr_fer : dlr_bos).ωn
    Kernel_t = 1. ./ (ωn1*im .- prefac * (Gp_in.tucker.ωs_center[1]' .- reshape(Gp_in.tucker.ωs_center[2], (1,1,sz_center[2]))))
    is_nan = .!isfinite.(Kernel_t)
    Kernel_t[is_nan] .= 0.



    # re-fit DLR-coefficients along free dimension for each partial fraction
    G_1 = dropdims(sum(reshape(Gp_out1.tucker.center, (1, sz_center...)) .* Kernel_t, dims=3), dims=3)
    G_2 = dropdims(sum(reshape(Gp_out1.tucker.center, (1, sz_center...)) .* Kernel_t, dims=2), dims=2)

    if sum(is_nan) > 0
        iω0 = argmax(is_nan)[1]
        Adisc_ano_temp = Gp_in.tucker.center[is_nan[iω0,:,:], [Colon() for _ in 3:D]...]
        
        G_ano = precompute_reg_div_values_MF_without_ωconv(1, Adisc_ano_temp, Gp_out1.tucker.legs)
        A_ano = fit_tucker_center(G_ano, Gp_out1.tucker.legs[2:end])
        view(G_1, iω0, [Colon() for _ in 2:D]...) .+= (+prefac)*A_ano
        view(G_2, iω0, [Colon() for _ in 2:D]...) .+= (-prefac)*A_ano

    end

#    println("max1: ", maximum(abs.(view(G_1, iω0, [Colon() for _ in 2:D]...))))

    Adisc_shift1 = reshape(Gp_out1.tucker.legs[1] \ reshape(G_1, (length(ωn1), div(length(G_1), length(ωn1)))), sz_center)
    Adisc_shift2 = reshape(Gp_out1.tucker.legs[1] \ reshape(G_2, (length(ωn1), div(length(G_2), length(ωn1)))), sz_center)
    # maximum(abs.(G_1 - Gp_out1.tucker.legs[1] * Adisc_shift1))
    # maximum(abs.(G_2 - Gp_out1.tucker.legs[1] * Adisc_shift2))
    Gp_out1.tucker.center = Adisc_shift1
    Gp_out2.tucker.center = Adisc_shift2
    if prefac == 1
        Gp_out1.tucker.center *= -1
    else
        Gp_out2.tucker.center *= -1
    end

    ## treat anomalous Adisc:
    if sum(.!Gp_in.isFermi) == 1 # if there is a bosonic frequency in the original Gp
        if sum(.!Gp_out1.isFermi)==0 && sum(.!Gp_out2.isFermi)==1 
            Gp_out1.Adisc_anoβ = Array{ComplexF64,D}(undef, zeros(Int, D)...) # delete Adisc_anoβ
            imax = argmax(.!Gp_out2.isFermi)
            isin = argmax(size(Gp_out2.Adisc_anoβ) .== 1)
            if imax != isin
                Gp_out2.Adisc_anoβ = permutedims(Gp_out2.Adisc_anoβ, (2,1,collect(3:D)...)) # permute dims if bosonic directions don't match
            end
        elseif sum(.!Gp_out1.isFermi)==1 && sum(.!Gp_out2.isFermi)==0 
            Gp_out2.Adisc_anoβ = Array{ComplexF64,D}(undef, zeros(Int, D)...) # delete Adisc_anoβ
            imax = argmax(.!Gp_out1.isFermi)
            isin = argmax(size(Gp_out1.Adisc_anoβ) .== 1)
            if imax != isin
                Gp_out1.Adisc_anoβ = permutedims(Gp_out1.Adisc_anoβ, (2,1,collect(3:D)...)) # permute dims if bosonic directions don't match
            end
        else
            @assert false
        end

    else    # if there are no bosonic frequencies in the original Gp, there is one in the new Gps:

    end

    TCI4Keldysh.update_frequency_args!(Gp_out1)
    TCI4Keldysh.update_frequency_args!(Gp_out2)

    if DEBUG()
        vals_orig = TCI4Keldysh.precompute_all_values_MF(Gp_in);
        vals_pfd1 = TCI4Keldysh.precompute_all_values_MF(Gp_out1)
        vals_pfd2 = TCI4Keldysh.precompute_all_values_MF(Gp_out2)
        diff = vals_pfd1+vals_pfd2-vals_orig
        maxdev = maximum(abs.(diff))
        println("Sup-norm deviation of original to decomposed Gps: \t", maxdev)
    end

    return Gp_out1, Gp_out2
end

"""
return nothing if no bosonic index exists
"""
function get_bosonic_idx(Gp::PartialCorrelator_reg) :: Union{Int, Nothing}
    return findfirst(.!Gp.isFermi)
end

"""
Determine whether a partial correlator requires an anomalous term.
"""
function ano_term_required(Gp::PartialCorrelator_reg) :: Bool
    bos_idx = get_bosonic_idx(Gp)
    return !isnothing(bos_idx) && !isnothing(findfirst(om -> abs(om)<=1.e-14, Gp.tucker.ωs_center[bos_idx]))
end

"""
Get idx of vanishing frequency in PSF grid
"""
function get_bosonic_freq_idx_Adisc(Gp::PartialCorrelator_reg) :: Union{Int, Nothing}
    bos_idx = findfirst(.!Gp.isFermi)
    if isnothing(bos_idx)
        return nothing
    else
        return findfirst(w -> abs.(w)<=1e-10, Gp.tucker.ωs_center[bos_idx]) 
    end
end

"""
Get idx of vanishing frequency in Greens function grid
"""
function get_bosonic_freq_idx_G(Gp::PartialCorrelator_reg) :: Union{Int, Nothing}
    bos_idx = findfirst(.!Gp.isFermi)
    if isnothing(bos_idx)
        return nothing
    else
        return findfirst(w -> abs.(w)<=1e-10, Gp.tucker.ωs_legs[bos_idx]) 
    end
end