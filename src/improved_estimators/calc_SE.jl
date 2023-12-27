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