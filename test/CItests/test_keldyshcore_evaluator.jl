using TCI4Keldysh
using Test


function test_Γcore_KF(
    iK::Int, flavor_idx, channel::String="a";
    R=3, tolerance=1.e-5, batched=false,
    unfoldingscheme=:interleaved,
    kwargs...
    )
    basepath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50")
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    ωconvMat = TCI4Keldysh.channel_trafo(channel)
    T = TCI4Keldysh.dir_to_T(PSFpath)
    broadening_kwargs = TCI4Keldysh.read_all_broadening_params(basepath; channel=channel)
    broadening_kwargs[:estep] = 5
    ωmax = 0.1
    D = 3
    iK_tuple = TCI4Keldysh.KF_idx(iK, D)

    # reference
    ωs_ext = TCI4Keldysh.KF_grid(ωmax, R, D)
    Σωgrid = TCI4Keldysh.KF_grid_fer(2*ωmax, R+1)
    (Σ_L,Σ_R) = TCI4Keldysh.calc_Σ_KF_aIE_viaR(PSFpath, Σωgrid; T=T, flavor_idx=flavor_idx, broadening_kwargs...)
    Γcore_ref = TCI4Keldysh.compute_Γcore_symmetric_estimator(
        "KF",
        PSFpath*"4pt/",
        # Σ_ref
        Σ_R
        ;
        Σ_calcL=Σ_L,
        T,
        flavor_idx = flavor_idx,
        ωs_ext = ωs_ext,
        ωconvMat=ωconvMat,
        broadening_kwargs...
    )

    # tci
    qtt = TCI4Keldysh.Γ_core_TCI_KF(
        PSFpath, R, iK, ωmax
        ; 
        T=T, ωconvMat=ωconvMat, flavor_idx=flavor_idx,
        tolerance=tolerance,
        unfoldingscheme=unfoldingscheme,
        batched=batched,
        broadening_kwargs...,
        kwargs...
        )

    # compare
    Γcore_tci = TCI4Keldysh.QTT_to_fatTensor(qtt, Base.OneTo.(fill(2^R, D)))
    diff = abs.(Γcore_tci .- Γcore_ref[1:2^R,:,:,iK_tuple...])
    maxref =  maximum(abs.(Γcore_ref))
    reldiff = diff ./ maxref
    @show maximum(reldiff)
    @test maximum(reldiff) < 3.0*tolerance
end

@testset begin
    for channel in ["a", "p", "t"]
        for iK in 1:15
            println("==== iK: $iK")
            for flavor_idx in 1:2
                test_Γcore_KF(iK, flavor_idx, channel; batched=true, unfoldingscheme=:fused, tolerance=1.e-8, R=3)
            end
        end
    end
end