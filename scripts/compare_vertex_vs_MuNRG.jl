using MAT
using Plots
using LinearAlgebra
using QuanticsTCI
import TensorCrossInterpolation as TCI

#=
Compare our MF/KF vertices with MuNRG results.
Careful: Results depend on which estimators (symmetric, left-asymmetric, right-asymmetric) are used for the self-energies
=#

function check_V_MF_CFdat()
    Vpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/V_MF_pp")

    for file in readdir(Vpath; join=true)
        matopen(file, "r") do f
            try
                keys(f)
            catch
                keys(f)
            end
            CFdat = read(f, "CFdat")
            println("For file: $file")
            @show size(CFdat["Ggrid"][1])
            @show size.(CFdat["ogrid"])
            println("")
        end
    end
end

"""
Compare MuNRG Matsubara vertices with TCI4Keldysh.
CAREFUL: Need channel="pNRG" for p-channel to get a consistent frequency convention
"""
function check_V_MF(Nhalf=2^4;channel="t")
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    Vpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50", "V_MF_" * TCI4Keldysh.channel_translate(channel))

    # Γcore data
    core_file = "V_MF_U4.mat"
    CF = nothing
    Γcore_ref = nothing
    ωs_ext = nothing
    spin = 1
    matopen(joinpath(Vpath, core_file), "r") do f
        CF = read(f, "CF")
        CFdat = read(f, "CFdat")
        Γcore_ref = CFdat["Ggrid"][spin]
        # bosonic grid comes last in the data
        ωs_ext = ntuple(i -> imag.(vec(vec(CFdat["ogrid"])[4-i])), 3)
    end
    # bosonic grid comes last in the data
    Γcore_ref = permutedims(Γcore_ref, (3,2,1))
    @show size.(ωs_ext)
    @show size(Γcore_ref)

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

    T = TCI4Keldysh.dir_to_T(PSFpath)
    om_small = TCI4Keldysh.MF_npoint_grid(T, Nhalf, 3)
    om_sig = TCI4Keldysh.MF_grid(T, 2*Nhalf, true)

    # sIE self-energy
    U = 0.05
    G        = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["F1", "F1dag"]; T, flavor_idx=spin, ωs_ext=(om_sig,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_aux    = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q1", "F1dag"]; T, flavor_idx=spin, ωs_ext=(om_sig,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_QQ_aux = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q1", "Q1dag"]; T, flavor_idx=spin, ωs_ext=(om_sig,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_data      = TCI4Keldysh.precompute_all_values(G)
    G_aux_data  = TCI4Keldysh.precompute_all_values(G_aux)
    G_QQ_aux_data=TCI4Keldysh.precompute_all_values(G_QQ_aux)
    Σ_calc_sIE = TCI4Keldysh.calc_Σ_MF_sIE(G_QQ_aux_data, G_aux_data, G_aux_data, G_data, U/2)

    # Γ core
    ωconvMat = TCI4Keldysh.channel_trafo(channel)
    @time testval = TCI4Keldysh.compute_Γcore_symmetric_estimator(
        "MF", PSFpath*"4pt/", Σ_calc_sIE; ωs_ext=om_small, T=T, ωconvMat=ωconvMat, flavor_idx=spin
        )
    
    # calulation DONE

    scfun = real

    slice = [div(length(om_small[1]), 2)+1, :, :]
    @show om_small[1][slice[1]]
    heatmap(scfun.(testval[slice...]))
    title!("Γcore TCI4Keldysh")
    savefig("gam.png")


    window_half = div(length(om_small[2]), 2)
    data_half = div(length(ωs_ext[2]), 2)
    window_slice = data_half-window_half+1:data_half+window_half
    slice_ref = [div(length(ωs_ext[1]), 2)+1, window_slice, window_slice]
    @show ωs_ext[1][slice_ref[1]]
    heatmap(scfun.(Γcore_ref[slice_ref...]))
    title!("Γcore reference")
    savefig("ref.png")

    # compare quantitatively
    window = (data_half-window_half+1:data_half+window_half+1, data_half-window_half+1:data_half+window_half, data_half-window_half+1:data_half+window_half)
    # pht / ph: add results
    diff = Γcore_ref[window...] .+ testval
    maxdiff = maximum(abs.(diff)) 
    printstyled("---- Max. abs. deviation: $(maxdiff)\n"; color=:blue)
    heatmap(log10.(abs.(Γcore_ref[slice_ref...] .+ testval[slice...])))
    savefig("diff.png")
    return maxdiff
end

function check_V_MF_all()
   check_V_MF(2^4;channel="t") 
   check_V_MF(2^4;channel="a") 
   check_V_MF(2^4;channel="pNRG") 
end

"""
Check Keldysh vertex of TCI4Keldysh against MuNRG results.
MuNRG results have frequency grids of size 2n+1 symmetric around 0.0
"""
function check_V_KF(Nhalf=2^3; iK::Int=1, channel="t")
    base_path = "SIAM_u=0.50"
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    Vpath = joinpath(TCI4Keldysh.datadir(), base_path, "V_KF_" * TCI4Keldysh.channel_translate(channel))

    # Γcore data
    core_file = "V_KF_U4.mat"
    CF = nothing
    Γcore_ref = nothing
    ωs_ext = nothing
    spin = 1
    matopen(joinpath(Vpath, core_file), "r") do f
        CF = read(f, "CF")
        CFdat = read(f, "CFdat")
        Γcore_ref = CFdat["Ggrid"][spin]
        ωs_ext = ntuple(i -> real.(vec(vec(CFdat["ogrid"])[4-i])), 3)
    end
    iK_tuple = TCI4Keldysh.KF_idx(iK, 3)
    Γcore_ref = permutedims(Γcore_ref, (1,2,3, 4,5,6,7))
    @show size.(ωs_ext)
    @show size(Γcore_ref)

    # Σ data
    ωs_Σ = nothing
    Σ_file = "SE_KF_1.mat"
    matopen(joinpath(Vpath, Σ_file), "r") do f
        CFdat = read(f, "CFdat")
        ωs_Σ = vec(vec(CFdat["ogrid"])[1])
    end
    @show size(ωs_Σ)

    # test
    (γ, sigmak) = TCI4Keldysh.read_broadening_params(base_path; channel=channel)
    T = TCI4Keldysh.dir_to_T(PSFpath)
    ωconvMat = TCI4Keldysh.channel_trafo(channel)

    ωs_cen = [div(length(om), 2)+1 for om in ωs_ext]
    @show [ωs_ext[i][ωs_cen[i]] for i in eachindex(ωs_ext)]
    om_small = ntuple(i -> ωs_ext[i][ωs_cen[i] - Nhalf : ωs_cen[i] + Nhalf], 3)
    ω_Σ_cen = div(length(ωs_Σ), 2) + 1
    om_sig = ωs_Σ[ω_Σ_cen - 2*Nhalf : ω_Σ_cen + 2*Nhalf]

    Σ_ref = TCI4Keldysh.calc_Σ_KF_sIE_viaR(PSFpath, om_sig; T=T, flavor_idx=spin, sigmak, γ)
    testval = TCI4Keldysh.compute_Γcore_symmetric_estimator(
        "KF",
        PSFpath*"4pt/",
        Σ_ref
        ;
        T,
        flavor_idx = spin,
        ωs_ext = om_small,
        ωconvMat=ωconvMat,
        sigmak, γ
    )

    # plot
    window = ntuple(i -> ωs_cen[i] - Nhalf : ωs_cen[i] + Nhalf, 2)
    slice_ref = [div(length(ωs_ext[1]), 2)+1, window..., iK_tuple...]
    @show ωs_ext[1][slice_ref[1]]
    @show ωs_ext[2][slice_ref[2]]
    @show ωs_ext[3][slice_ref[3]]
    heatmap(abs.(Γcore_ref[slice_ref...]))
    savefig("ref.png")

    slice = [div(length(om_small[1]), 2)+1, :, :, iK_tuple...]
    @show om_small[1][slice[1]]
    @show om_small[2][slice[2]]
    @show om_small[3][slice[3]]
    heatmap(abs.(testval[slice...]))
    savefig("gam.png")
end

function compress_precomputed_V_KF(iK::Int, channel="t"; tcikwargs...)
    base_path = "SIAM_u=0.50"
    Vpath = joinpath(TCI4Keldysh.datadir(), base_path, "V_KF_" * TCI4Keldysh.channel_translate(channel))

    # Γcore data
    core_file = "V_KF_U4.mat"
    Γcore_ref = nothing
    spin = 1
    matopen(joinpath(Vpath, core_file), "r") do f
        CFdat = read(f, "CFdat")
        Γcore_ref = CFdat["Ggrid"][spin]
    end
    iK_tuple = TCI4Keldysh.KF_idx(iK, 3)
    Γcore_ref = permutedims(Γcore_ref, (1,2,3, 4,5,6,7))
    to_tci = TCI4Keldysh.zeropad_array(Γcore_ref[:,:,:,iK_tuple...])
    @show size(Γcore_ref)

    qtt, _, _ = quanticscrossinterpolate(to_tci; tcikwargs...)

    @show TCI.linkdims(qtt.tci)
end