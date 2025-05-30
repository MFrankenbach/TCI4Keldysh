#=
This file provides different methods to compute the selfenergy.
=#

"""
calc_Σ_MF_dir

Calculate the selfenergy Σ directly from a full propagator 'G' and an inverse bare propagator 'G0inv'.
"""
function calc_Σ_MF_dir(G::Vector{ComplexF64}, G0inv::Vector{ComplexF64}) ::Vector{ComplexF64}
    return G0inv .- 1. ./ G
end

"""
cf. eq (105) Lihm et. al.
"""
function calc_Σ_MF_aIE(G_aux::Vector{ComplexF64}, G::Vector{ComplexF64}) ::Vector{ComplexF64}
    return G_aux ./ G
end

"""
cf. eq (108) Lihm et. al.
"""
function calc_Σ_MF_sIE(G_QQ_aux::Vector{ComplexF64}, G_QF_aux::Vector{ComplexF64}, G_FQ_aux::Vector{ComplexF64}, G::Vector{ComplexF64}, Σ_H::Float64) ::Vector{ComplexF64}
    return G_QQ_aux .+ Σ_H .- G_QF_aux ./ G .* G_FQ_aux
end

"""
cf. eq (108) Lihm et. al.

Keldysh version. 2-point correlators come in shape:
(length(ωs_ext), 2, 2)
"""
function calc_Σ_KF_sIE(G_QQ_aux::Array{ComplexF64}, G_QF_aux::Array{ComplexF64}, G_FQ_aux::Array{ComplexF64}, G::Array{ComplexF64}, Σ_H) ::Array{ComplexF64}
    error("not tested")
    @assert size(G_QF_aux)==size(G_FQ_aux)==size(G_QQ_aux)
    Σ_sIE = zeros(ComplexF64, size(G_QQ_aux))
    for w in axes(G_QQ_aux, 1)
        Σ_sIE[w,:,:] = G_QQ_aux[w,:,:] .+ Σ_H .- G_QF_aux[w,:,:] ./ G[w,:,:] .* G_FQ_aux[w,:,:]
    end
    return Σ_sIE
end

"""
convenience overload
"""
function calc_Σ_MF_sIE(PSFpath, ω_fer::Vector{Float64}; flavor_idx::Int, T::Float64)
    Adisc_Σ_H = load_Adisc_0pt(PSFpath, "Q12")
    Σ_H = only(Adisc_Σ_H)
    G        = FullCorrelator_MF(PSFpath, ["F1", "F1dag"]; T, flavor_idx=flavor_idx, ωs_ext=(ω_fer,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_auxL    = FullCorrelator_MF(PSFpath, ["Q1", "F1dag"]; T, flavor_idx=flavor_idx, ωs_ext=(ω_fer,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_auxR    = FullCorrelator_MF(PSFpath, ["F1", "Q1dag"]; T, flavor_idx=flavor_idx, ωs_ext=(ω_fer,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_QQ_aux = FullCorrelator_MF(PSFpath, ["Q1", "Q1dag"]; T, flavor_idx=flavor_idx, ωs_ext=(ω_fer,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_data      = precompute_all_values(G)
    G_auxL_data  = precompute_all_values(G_auxL)
    G_auxR_data  = precompute_all_values(G_auxR)
    G_QQ_aux_data= precompute_all_values(G_QQ_aux)
    return calc_Σ_MF_sIE(G_QQ_aux_data, G_auxL_data, G_auxR_data, G_data, Σ_H)
end

"""
convenience overload
"""
function calc_Σ_MF_aIE(PSFpath::String, ω_fer::Vector{Float64}; flavor_idx::Int, T::Float64)

    G        = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["F1", "F1dag"]; T, flavor_idx=flavor_idx, ωs_ext=(ω_fer,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_data = TCI4Keldysh.precompute_all_values(G)

    G_auxL   = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q1", "F1dag"]; T, flavor_idx=flavor_idx, ωs_ext=(ω_fer,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_auxL_data = TCI4Keldysh.precompute_all_values(G_auxL)
    Σ_calcL = TCI4Keldysh.calc_Σ_MF_aIE(G_auxL_data, G_data)

    G_auxR   = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["F1", "Q1dag"]; T, flavor_idx=flavor_idx, ωs_ext=(ω_fer,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_auxR_data = TCI4Keldysh.precompute_all_values(G_auxR)
    Σ_calcR = TCI4Keldysh.calc_Σ_MF_aIE(G_auxR_data, G_data)

    return (Σ_calcL, Σ_calcR)
end

"""
convenience overload
Compute Σ with pointwise matrix inversion.
"""
function calc_Σ_KF_aIE(PSFpath::String, ω_fer::Vector{Float64}; mode::Symbol=:normal, flavor_idx::Int, sigmak::Vector{Float64}, γ::Float64, broadening_kwargs...)
    T = dir_to_T(PSFpath)
    ωconvMat = ωconvMat_K1()
    # precompute correlators
    if mode==:normal
        GL = FullCorrelator_KF(PSFpath, ["F1", "F1dag"]; T, flavor_idx=flavor_idx, ωs_ext=(ω_fer,), ωconvMat=ωconvMat, sigmak=sigmak, γ=[γ, 3γ], broadening_kwargs...)
        GR = FullCorrelator_KF(PSFpath, ["F1", "F1dag"]; T, flavor_idx=flavor_idx, ωs_ext=(ω_fer,), ωconvMat=ωconvMat, sigmak=sigmak, γ=[3γ, γ], broadening_kwargs...)
        G_aux_L = FullCorrelator_KF(PSFpath, ["F1", "Q1dag"]; T, flavor_idx=flavor_idx, ωs_ext=(ω_fer,), ωconvMat=ωconvMat, sigmak=sigmak, γ=[γ, 3γ], broadening_kwargs...)
        G_aux_R = FullCorrelator_KF(PSFpath, ["Q1", "F1dag"]; T, flavor_idx=flavor_idx, ωs_ext=(ω_fer,), ωconvMat=ωconvMat, sigmak=sigmak, γ=[3γ, γ], broadening_kwargs...)
        GL_data = precompute_all_values(GL)
        GR_data = precompute_all_values(GR)
        G_aux_L_data = precompute_all_values(G_aux_L)
        G_aux_R_data = precompute_all_values(G_aux_R)
    elseif mode==:fdt
        error("Mode $mode deprecated")
        # G_data = precompute_all_values_FDT(PSFpath, ["F1", "F1dag"], ω_fer; flavor_idx=flavor_idx, T=T, γ=γ, sigmak=sigmak, broadening_kwargs...)
        # G_aux_L_data = precompute_all_values_FDT(PSFpath, ["F1", "Q1dag"], ω_fer; flavor_idx=flavor_idx, T=T, γ=γ, sigmak=sigmak, broadening_kwargs...)
        # G_aux_R_data = precompute_all_values_FDT(PSFpath, ["Q1", "F1dag"], ω_fer; flavor_idx=flavor_idx, T=T, γ=γ, sigmak=sigmak, broadening_kwargs...)
    else
        error("Invalid mode $mode")
    end

    # deduce self-energies
    X = get_PauliX()
    I = diagm([1.0,1.0])
    Σ_L = zeros(ComplexF64, size(GL_data))
    for i in axes(Σ_L,1)
        Σ_L[i,:,:] .= X*G_aux_L_data[i,:,:]/GL_data[i,:,:]
    end
    Σ_R = zeros(ComplexF64, size(GR_data))
    for i in axes(Σ_R,1)
        Σ_R[i,:,:] .= (I/GR_data[i,:,:]) * G_aux_R_data[i,:,:] * X
    end
    return (Σ_L, Σ_R)
end

"""
convenience overload
Compute Σ by SYMMETRIC estimator of retarded self-energy and fluctuation-dissipation relation
"""
function calc_Σ_KF_sIE_viaR(PSFpath, ω_fer::Vector{Float64}; flavor_idx::Int, T::Float64, sigmak::Vector{Float64}, γ::Float64, broadening_kwargs...)
    G_FF_R = TCI4Keldysh.get_GR(PSFpath, ["F1", "F1dag"]; T, flavor_idx=flavor_idx, ωs_ext=ω_fer, sigmak=sigmak, γ, broadening_kwargs...)
    G_FQ_R = TCI4Keldysh.get_GR(PSFpath, ["F1", "Q1dag"]; T, flavor_idx=flavor_idx, ωs_ext=ω_fer, sigmak=sigmak, γ, broadening_kwargs...)
    G_QF_R = TCI4Keldysh.get_GR(PSFpath, ["Q1", "F1dag"]; T, flavor_idx=flavor_idx, ωs_ext=ω_fer, sigmak=sigmak, γ, broadening_kwargs...)
    G_QQ_R = TCI4Keldysh.get_GR(PSFpath, ["Q1", "Q1dag"]; T, flavor_idx=flavor_idx, ωs_ext=ω_fer, sigmak=sigmak, γ, broadening_kwargs...)
    Adisc_Σ_H = load_Adisc_0pt(PSFpath, "Q12")
    Σ_H = only(Adisc_Σ_H)
    ΣR_sIE = TCI4Keldysh.calc_Σ_MF_sIE(G_QQ_R, G_QF_R, G_FQ_R, G_FF_R, Σ_H)
    # use FDT
    Σ = reshape(hcat(2*im*tanh.(ω_fer/2/T).*imag.(ΣR_sIE),conj.(ΣR_sIE),ΣR_sIE,0 .*ΣR_sIE), length(ω_fer),2,2)
    return Σ
end

"""
convenience overload
Compute Σ by ASYMMETRIC estimator of retarded self-energy and fluctuation-dissipation relation
"""
function calc_Σ_KF_aIE_viaR(PSFpath, ω_fer::Vector{Float64}; flavor_idx::Int, T::Float64, sigmak::Vector{Float64}, γ::Float64, broadening_kwargs...)
    @warn "using FDR in self energy"
    G_FF_R = TCI4Keldysh.get_GR(PSFpath, ["F1", "F1dag"]; T, flavor_idx=flavor_idx, ωs_ext=ω_fer, sigmak=sigmak, γ, broadening_kwargs...)
    G_FQ_R = TCI4Keldysh.get_GR(PSFpath, ["F1", "Q1dag"]; T, flavor_idx=flavor_idx, ωs_ext=ω_fer, sigmak=sigmak, γ, broadening_kwargs...)
    G_QF_R = TCI4Keldysh.get_GR(PSFpath, ["Q1", "F1dag"]; T, flavor_idx=flavor_idx, ωs_ext=ω_fer, sigmak=sigmak, γ, broadening_kwargs...)

    ΣR_aIE_L = TCI4Keldysh.calc_Σ_MF_aIE(G_QF_R, G_FF_R)
    ΣR_aIE_R = TCI4Keldysh.calc_Σ_MF_aIE(G_FQ_R, G_FF_R)
    # use FDT
    Σ_R = reshape(hcat(2*im*tanh.(ω_fer/2/T).*imag.(ΣR_aIE_R),conj.(ΣR_aIE_R),ΣR_aIE_R,0 .*ΣR_aIE_R), length(ω_fer),2,2)
    Σ_L = reshape(hcat(2*im*tanh.(ω_fer/2/T).*imag.(ΣR_aIE_L),conj.(ΣR_aIE_L),ΣR_aIE_L,0 .*ΣR_aIE_L), length(ω_fer),2,2)
    return (Σ_L, Σ_R)
end


"""
precompute_Σs_MF(
    Σ_vec   ::Vector{ComplexF64},   vector of Σ data
    sizeK23 ::NTuple{N,Int},        size of K2/K3 data
    ωconvMat::Matrix{Int}           The rows describe how the fermionic frequencies depend on the frequency parametrization.
                                    Convention: The first row always corresponds to a bosonic frequency. The other rows to fermionic ones.

Precomputes the selfenergies needed for the symmetric estimators of K2 and 4p core vertex.
"""
function precompute_Σs(
    Σ_vec       ::Array{ComplexF64,N},   # vector of Σ data (convention: first dimension is fermionic frequency, trailing dimensions can be other stuff such as Keldysh indices)
    sizeK23     ::NTuple{N1,Int},        # size of K2/core data (convention: sizeK23[1] is bosonic frequency, sizeK23[2/3] is fermionic frequency if defined)
    ωconvMat    ::Matrix{Int},           # The rows describe how the fermionic frequencies depend on the frequency parametrization.
                                        # Convention: The first row always corresponds to a bosonic frequency. The other rows to fermionic ones.
    is_incoming ::NTuple{N2,Bool}        # True if the fermionic leg is incoming aka daggered
    ) where{N,N1, N2}
    println("size of Σ_vec:" , size(Σ_vec))
    # prepare stuff to return:
    N_out = size(ωconvMat, 2) + ndims(Σ_vec)-1 # dims of Σs_return
    Σs_return = Vector{Array{ComplexF64,N_out}}(undef, size(ωconvMat)[1])

    # precompute a normal and a slanted version:
    N_Σ = size(Σ_vec, 1)
    N_bos = sizeK23[1]
    N_fer = sizeK23[2]
    i_Σ_straight_l = div(N_Σ - N_fer, 2) + 1
    i_Σ_straight_h = N_Σ - i_Σ_straight_l + 1
    Σ_straight = Σ_vec[i_Σ_straight_l:i_Σ_straight_h, [Colon() for _ in 2:N]...]
    #println("size of Σ_straight", size(Σ_straight))
    Σ_slanted = Array{ComplexF64,2+ndims(Σ_vec)-1}(undef, N_bos, N_fer, size(Σ_vec)[2:end]...)
    println("size of Σ_slanted: ", size(Σ_slanted))
    iωbos0 = div(N_bos+1,2)
    for iωbos in 1:N_bos
        shift = -iωbos0+iωbos
        #println("size of Σ_vec[i_Σ_straight_l+shift:i_Σ_straight_h+shift, [Colon() for _ in 2:N]...]: ", size(Σ_vec[i_Σ_straight_l+shift:i_Σ_straight_h+shift, [Colon() for _ in 2:N]...]))
        Σ_slanted[iωbos,[Colon() for _ in 1:N]...] .= Σ_vec[i_Σ_straight_l+shift:i_Σ_straight_h+shift, [Colon() for _ in 2:N]...]
    end

    # iterate through ωconvMat and compute the correct thing:
    for i in 1:size(ωconvMat)[1]
        pos_ferm_freq = argmax(abs.(ωconvMat[i,2:end]))+1
        if ωconvMat[i,:1] == 0 # the i-th column directly corresponds to ONE fermionic frequency
            Σtmp = deepcopy(Σ_straight)
            if ωconvMat[i,pos_ferm_freq] == -1
                reverse!(Σtmp, dims=1)
            end
            new_shape = [ones(Int, size(ωconvMat)[2])..., size(Σ_vec)[2:end]...]
            new_shape[pos_ferm_freq] = N_fer
            
            if is_incoming[i]
                reverse!(Σtmp, dims=1)
            end 
        else                   # the i-th column is a combination of ONE fermionic and ONE bosonic frequency 
            Σtmp = deepcopy(Σ_slanted)
            if ωconvMat[i,pos_ferm_freq] == -1
                Σtmp = reverse(Σtmp, dims=2)
            end
            if ωconvMat[i,1] == -1
                Σtmp = reverse(Σtmp, dims=1)
            end
            new_shape = [ones(Int, size(ωconvMat)[2])..., size(Σ_vec)[2:end]...]
            new_shape[1] = N_bos
            new_shape[pos_ferm_freq] = N_fer
           
            if is_incoming[i]
                reverse!(Σtmp, dims=(1,2))
            end 
        end
        Σtmp = reshape(Σtmp, (new_shape...,))
        Σs_return[i] = Σtmp
    end

    return Σs_return
end