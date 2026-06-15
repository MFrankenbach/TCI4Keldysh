using MAT
using Plots
using LinearAlgebra
using HDF5
using Combinatorics
using LaTeXStrings
using TCI4Keldysh

#=
Compare freshly computed Keldysh core vertex with MuNRG result.
=#

"""
Compare MuNRG Matsubara vertices with TCI4Keldysh.
CAREFUL: Need channel="pNRG" for p-channel to get a consistent frequency convention
"""
function check_V_full_MF(Nhalf=2^4;channel="t", use_ΣaIE=true, spin::Int=1, basepath="SIAM_u=0.50")
    PSFpath = joinpath(TCI4Keldysh.datadir(), basepath, "PSF_nz=4_conn_zavg/")
    Vpath = joinpath(TCI4Keldysh.datadir(), basepath, "V_MF_" * TCI4Keldysh.channel_translate(channel))

    # Γfull data
    full_file = "V_MF_sym.mat"
    CF = nothing
    Γfull_ref = nothing
    ωs_ext = nothing
    matopen(joinpath(Vpath, full_file), "r") do f
        CF = read(f, "CF")
        CFdat = read(f, "CFdat")
        Γfull_ref = CFdat["Ggrid"][spin]
        # bosonic grid comes last in the data
        ωs_ext = ntuple(i -> imag.(vec(vec(CFdat["ogrid"])[4-i])), 3)
    end
    # bosonic grid comes last in the data
    Γfull_ref = permutedims(Γfull_ref, (3,1,2))
    @show size.(ωs_ext)
    @show size(Γfull_ref)

    # Σ data
    ωs_Σ = nothing
    Σ_file = "SE_MF_1.mat"
    matopen(joinpath(Vpath, Σ_file), "r") do f
        CF = read(f, "CF")
        CFdat = read(f, "CFdat")
        ωs_Σ_ = vec(vec(CFdat["ogrid"])[1])
        @assert norm(real.(ωs_Σ_)) <= 1.e-10
        ωs_Σ = imag.(ωs_Σ_)
    end

    @show size(ωs_Σ)
    @show typeof(ωs_Σ)

    # TCI4Keldysh calculation
    if channel in ["p","pQFT"]
        error("Need pNRG for the p-channel")
    end

    T = TCI4Keldysh.dir_to_T(PSFpath)
    om_small = TCI4Keldysh.MF_npoint_grid(T, Nhalf, 3)
    om_sig = TCI4Keldysh.MF_grid(T, 2*Nhalf, true)

    # Γ full
    sgntrafo = 1
    ωconvMat = sgntrafo * TCI4Keldysh.channel_trafo(channel)
    @time testval = if use_ΣaIE
        TCI4Keldysh.compute_Γfull_symmetric_estimator(
            "MF", PSFpath;
            T=T, channel=channel, flavor_idx=spin, ωs_ext=om_small
            )
    else # use sIE for self-energy
        error("sIE not supported for full vertex")
        Σ_calc_sIE = TCI4Keldysh.calc_Σ_MF_sIE(PSFpath, om_sig; flavor_idx=spin, T=T)
        TCI4Keldysh.compute_Γfull_symmetric_estimator(
            "MF", PSFpath, Σ_calc_sIE;
            ωs_ext=om_small, T=T, ωconvMat=ωconvMat, flavor_idx=spin
            )
    end

    # calulation DONE

    scfun = x -> real(x)

    slice = [div(length(om_small[1]), 2)+1, :, :]
    @show om_small[1][slice[1]]
    heatmap(scfun.(testval[slice...]); right_margin=10Plots.mm)
    title!("Γfull TCI4Keldysh")
    file_prefix = "V_MF_$(channel)_spincomponent$(spin)_"
    savefig(joinpath(dirname(@__FILE__), "output", "plots", file_prefix * "gam.pdf"))


    window_half = div(length(om_small[2]), 2)
    data_half = div(length(ωs_ext[2]), 2)
    window_slice = data_half-window_half+1:data_half+window_half
    slice_ref = [div(length(ωs_ext[1]), 2)+1, window_slice, window_slice]
    @show ωs_ext[1][slice_ref[1]]
    heatmap(scfun.(Γfull_ref[slice_ref...]); right_margin=10Plots.mm)
    title!("Γfull reference")
    savefig(joinpath(dirname(@__FILE__), "output", "plots", file_prefix * "ref.pdf"))

    # compare quantitatively
    window = (data_half-window_half+1:data_half+window_half+1, data_half-window_half+1:data_half+window_half, data_half-window_half+1:data_half+window_half)
    # Γ(ω,ν,ν')=Γ*(-ω,-ν,-ν')
    diff = if sgntrafo==1
        # need to reverse because the MuNRG channel transformations differ from ours by a global minus sign
        Γfull_ref[window...] .- reverse(testval)
    elseif sgntrafo==-1
        Γfull_ref[window...] .- testval
    else
        error("Invalid value $(sgntrafo) of transformation prefactor")
    end
    maxdiff = maximum(abs.(diff)) 
    amaxdiff = argmax(abs.(diff)) 
    @show maximum(abs.(Γfull_ref))
    @show amaxdiff
    @show diff[amaxdiff]
    @show testval[amaxdiff]
    @show Γfull_ref[window...][amaxdiff]
    printstyled("---- Max. abs. deviation: $(maxdiff) (Γfull value: $(testval[amaxdiff]))\n"; color=:blue)
    printstyled("---- Max. rel. abs. deviation: $(maxdiff/maximum(abs.(Γfull_ref))) (max. Γfull value: $(maximum(abs.(Γfull_ref))))\n"; color=:blue)
    scfun = x -> abs(x)
    heatmap(scfun.(Γfull_ref[slice_ref...] .+ testval[slice...]); right_margin=10Plots.mm)
    savefig(joinpath(dirname(@__FILE__), "output", "plots", file_prefix * "diff.pdf"))
    return maxdiff
end

check_V_full_MF(channel="t", use_ΣaIE=true, spin=1)
check_V_full_MF(channel="t", use_ΣaIE=true, spin=2)