#=
This file provides different methods for the computation of the selfenergy
=#

"""
calc_Σ_MF_dir

Calculate the selfenergy Σ directly from a full propagator 'G' and an inverse bare propagator 'G0inv'.
"""
function calc_Σ_MF_dir(G::Vector{ComplexF64}, G0inv::Vector{ComplexF64}) ::Vector{ComplexF64}
    return G0inv .- 1. ./ G
end

function calc_Σ_MF_aIE(G_aux::Vector{ComplexF64}, G::Vector{ComplexF64}) ::Vector{ComplexF64}
    return G_aux ./ G
end
function calc_Σ_MF_sIE(G_QQ_aux::Vector{ComplexF64}, G_QF_aux::Vector{ComplexF64}, G_FQ_aux::Vector{ComplexF64}, G::Vector{ComplexF64}, Σ_H) ::Vector{ComplexF64}
    return G_QQ_aux .+ Σ_H .- G_QF_aux ./ G .* G_FQ_aux
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
    Σ_vec       ::Vector{ComplexF64},   # vector of Σ data
    sizeK23     ::NTuple{N1,Int},        # size of K2/core data
    ωconvMat    ::Matrix{Int},           # The rows describe how the fermionic frequencies depend on the frequency parametrization.
                                        # Convention: The first row always corresponds to a bosonic frequency. The other rows to fermionic ones.
    is_incoming ::NTuple{N2,Bool}        # True if the fermionic leg is incoming aka daggered
    ) where{N1, N2}
    # prepare stuff to return:
    Σs_return = Vector{Array{ComplexF64,size(ωconvMat)[2]}}(undef, size(ωconvMat)[1])

    # precompute a normal and a slanted version:
    N_Σ = length(Σ_vec)
    N_bos = sizeK23[1]
    N_fer = sizeK23[2]
    i_Σ_straight_l = div(N_Σ - N_fer, 2) + 1
    i_Σ_straight_h = N_Σ - i_Σ_straight_l + 1
    Σ_straight = Σ_vec[i_Σ_straight_l:i_Σ_straight_h]
    #println("size of Σ_straight", size(Σ_straight))
    Σ_slanted = Matrix{ComplexF64}(undef, N_bos, N_fer)
    iωbos0 = div(N_bos+1,2)
    for iωbos in 1:N_bos
        shift = -iωbos0+iωbos
        Σ_slanted[iωbos,:] .= Σ_vec[i_Σ_straight_l+shift:i_Σ_straight_h+shift]
    end

    # iterate through ωconvMat and compute the correct thing:
    for i in 1:size(ωconvMat)[1]
        pos_ferm_freq = argmax(abs.(ωconvMat[i,2:end]))+1
        if ωconvMat[i,:1] == 0 # the i-th column directly corresponds to ONE fermionic frequency
            Σtmp = deepcopy(Σ_straight)
            if ωconvMat[i,pos_ferm_freq] == -1
                reverse!(Σtmp)
            end
            new_shape = ones(Int, size(ωconvMat)[2])
            new_shape[pos_ferm_freq] = N_fer
            
        else                   # the i-th column is a combination of ONE fermionic and ONE bosonic frequency 
            Σtmp = deepcopy(Σ_slanted)
            if ωconvMat[i,pos_ferm_freq] == -1
                Σtmp = reverse(Σtmp, dims=2)
            end
            if ωconvMat[i,1] == -1
                Σtmp = reverse(Σtmp, dims=1)
            end
            new_shape = ones(Int, size(ωconvMat)[2])
            new_shape[1] = N_bos
            new_shape[pos_ferm_freq] = N_fer
            
        end
        if is_incoming[i]
            reverse!(Σtmp)
        end
        Σtmp = reshape(Σtmp, (new_shape...,))
        Σs_return[i] = Σtmp
    end

    return Σs_return
end